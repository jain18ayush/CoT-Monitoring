#!/usr/bin/env python3
import argparse, json, os, re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------ Device ------------------
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ------------------ Model load ------------------
def load_model(model_name: str, device: str, hf_token: Optional[str] = None):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device == "cuda" else None,
        dtype=dtype,
        token=hf_token,
    )
    if device in ["cpu", "mps"]:
        model.to(device)
    model.eval()
    return model, tok

# ------------------ Layer band helpers ------------------
def choose_layers(model, band: str) -> List[int]:
    """
    band: 'all' | 'first_third' | 'middle_third' | 'last_third' | 'range:START:END' (END exclusive)
    """
    n = len(model.model.layers)
    if band == "all":
        return list(range(n))
    if band == "first_third":
        return list(range(0, n//3))
    if band == "middle_third":
        return list(range(n//3, 2*n//3))
    if band == "last_third":
        return list(range(2*n//3, n))
    if band.startswith("range:"):
        _, a, b = band.split(":")
        a, b = int(a), int(b)
        a = max(0, min(a, n))
        b = max(0, min(b, n))
        return list(range(a, b))
    # default
    return list(range(n//3, 2*n//3))

# ------------------ Scoring ------------------
def gather_answer_logprob(logits: torch.Tensor,
                          target_ids: torch.Tensor,
                          span: Dict[str, int]) -> float:
    """
    logits: [1, S, V]
    target_ids: [1, S]
    span: {"tok_start": i, "tok_end": j} (absolute indices into target_ids)
    Returns sum log p(target_ids[i..j-1] | prefix)
    """
    i, j = span["tok_start"], span["tok_end"]
    if i is None or j is None or i >= j:
        return float("nan")
    S = target_ids.shape[1]
    i = max(1, min(i, S-1))   # guard for shift
    j = max(i, min(j, S))
    # Next-token prediction: logits at t-1 predict token at t
    # Use positions [i-1 .. j-2] to predict [i .. j-1]
    logits_slice = logits[:, i-1:j-1, :]            # [1, L, V]
    targets_slice = target_ids[:, i:j]              # [1, L]
    logp = F.log_softmax(logits_slice, dim=-1).gather(-1, targets_slice.unsqueeze(-1)).squeeze(-1)
    return float(logp.sum().item())

# ------------------ Hooks (necessity: set null) ------------------
def make_set_null_hook(tok_slice: slice):
    """
    Returns a forward hook that zeros the residual stream on tok_slice.
    Works for blocks returning Tensor or tuple(Tensor, ...).
    """
    def hook(module, inp, out):
        if isinstance(out, tuple):
            hs = out[0]
            # defensive: clone since some modules reuse buffers
            hs = hs.clone()
            hs[:, tok_slice, :] = 0.0
            return (hs,) + out[1:]
        else:
            hs = out.clone()
            hs[:, tok_slice, :] = 0.0
            return hs
    return hook

def score_with_optional_patch(model, input_ids: torch.Tensor,
                              answer_span: Dict[str, int],
                              patch_hook=None, layers: List[int]=None) -> float:
    handles = []
    try:
        if patch_hook and layers:
            for li in layers:
                handles.append(model.model.layers[li].register_forward_hook(patch_hook))
        with torch.no_grad():
            out = model(input_ids=input_ids)
            logits = out.logits  # [1, S, V]
        return gather_answer_logprob(logits, input_ids, answer_span)
    finally:
        for h in handles:
            h.remove()

# ------------------ Main logic per record ------------------
def ablate_record_steps(model, record: Dict[str, Any], layers: List[int], device: str):
    """
    Yields per-step results for one record.
    """
    # Required fields sanity
    if "token_ids_full" not in record or "final_answer_span" not in record or "step_spans" not in record:
        return
    full_ids = record["token_ids_full"]
    ans_span = record["final_answer_span"]
    steps = record["step_spans"]

    # Make tensors
    input_ids = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)

    # Base score (no patch)
    base_logp = score_with_optional_patch(model, input_ids, ans_span, patch_hook=None, layers=None)

    for s in steps:
        s_idx = s.get("step_idx", None)
        a, b = s.get("tok_start_abs", None), s.get("tok_end_abs", None)
        # Validate indices
        if a is None or b is None or not (0 <= a < b <= len(full_ids)):
            continue
        tok_slice = slice(a, b)
        patch = make_set_null_hook(tok_slice)
        abl_logp = score_with_optional_patch(model, input_ids, ans_span, patch_hook=patch, layers=layers)
        yield {
            "id": record.get("id", ""),
            "attempt": record.get("attempt", -1),
            "model": record.get("model", ""),
            "step_idx": s_idx,
            "tok_start_abs": a,
            "tok_end_abs": b,
            "layers": layers,
            "base_logp_answer": base_logp,
            "ablated_logp_answer": abl_logp,
            "delta_logp_answer": base_logp - abl_logp,  # >0 => step necessary
        }

# ------------------ I/O ------------------
def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_jsonl(path: Path, records: List[Dict[str, Any]]):
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ------------------ CLI ------------------
def main():
    ap = argparse.ArgumentParser(description="Necessity ablation on saved CoT steps (teacher-forced).")
    ap.add_argument("--jsonl", type=str, required=True, help="Path to runs.jsonl (from your mining script).")
    ap.add_argument("--out", type=str, default="ablation_results.jsonl", help="Output JSONL (appended).")
    ap.add_argument("--model", type=str, required=True, help="HF model name (same family as used for mining).")
    ap.add_argument("--hf_token", type=str, default="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")
    ap.add_argument("--layers", type=str, default="middle_third",
                    help="Layer band: all|first_third|middle_third|last_third|range:START:END")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on number of records to process (0 = all).")
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")
    model, tok = load_model(args.model, device, args.hf_token)
    layer_list = choose_layers(model, args.layers)
    print(f"Patching layers: {layer_list}")

    src = Path(args.jsonl)
    dst = Path(args.out)
    n_done = 0
    buffer: List[Dict[str, Any]] = []

    for rec in read_jsonl(src):
        # Only ablate if we have a valid final answer span and at least one step
        fas = rec.get("final_answer_span", {})
        if not isinstance(fas, dict) or fas.get("tok_start", None) is None or fas.get("tok_end", None) is None:
            continue
        if not rec.get("step_spans"):
            continue

        for res in ablate_record_steps(model, rec, layer_list, device):
            buffer.append(res)

        # flush often to keep progress saved
        if len(buffer) >= 50:
            write_jsonl(dst, buffer)
            n_done += len(buffer)
            print(f"Saved {n_done} results → {dst}")
            buffer = []

        if args.limit and n_done >= args.limit:
            break

    # final flush
    if buffer:
        write_jsonl(dst, buffer)
        n_done += len(buffer)
    print(f"Done. Wrote {n_done} ablation results to {dst}")

if __name__ == "__main__":
    main()
