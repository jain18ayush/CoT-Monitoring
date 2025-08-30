#!/usr/bin/env python3
"""
Collect step→step influence matrices for all records in runs.jsonl.

For each record (one CoT trace):
  - Do ONE base forward pass (teacher-forced) to cache logits.
  - For EACH source step A, do ONE patched forward pass (zero residuals on A tokens across chosen layers).
  - For EACH destination step B, compute Δ_{A→B} = logp_B(base) - logp_B(patched).

Writes one JSON object per record to the output JSONL, including:
  - id, attempt, model
  - layer_band + actual layer indices used
  - step_indices, token spans, per-step base logp
  - matrix: list-of-lists (rows=source step, cols=destination step)

Usage:
  python collect_step2step.py \
    --jsonl-in MATS2025_WINTER_RESULTS/runs.jsonl \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --layer-band middle_third \
    --out MATS2025_WINTER_RESULTS/step2step.jsonl
"""

import argparse, json, os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------ device ------------------
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ------------------ model loading (dtype-safe) ------------------
def load_model(model_name: str, device: str, hf_token: Optional[str] = None):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    kwargs = dict(device_map="auto" if device == "cuda" else None)
    # new transformers prefers `dtype=`, older uses `torch_dtype=`
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, token=hf_token, **kwargs
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, token=hf_token, **kwargs
        )
    if device in ["cpu", "mps"]:
        model.to(device)
    model.eval()
    return model, tok

# ------------------ layer selection ------------------
def choose_layers(model, band: str) -> List[int]:
    """
    band: 'all'|'first_third'|'middle_third'|'last_third'|'range:START:END' (END exclusive)
    """
    n = len(model.model.layers)
    if band == "all": return list(range(n))
    if band == "first_third": return list(range(0, n//3))
    if band == "middle_third": return list(range(n//3, 2*n//3))
    if band == "last_third": return list(range(2*n//3, n))
    if band.startswith("range:"):
        try:
            _, a, b = band.split(":"); a, b = int(a), int(b)
        except Exception:
            a, b = 0, n
        a = max(0, min(a, n)); b = max(0, min(b, n))
        return list(range(a, b))
    return list(range(n//3, 2*n//3))  # default

# ------------------ scoring utils ------------------
def gather_span_logprob(logits: torch.Tensor,
                        target_ids: torch.Tensor,
                        span: Dict[str, int]) -> float:
    """
    logits: [1,S,V]; target_ids: [1,S]
    span = {"tok_start": i, "tok_end": j} absolute indices into target_ids
    Returns sum log p of tokens i..j-1 (next-token).
    """
    i, j = int(span["tok_start"]), int(span["tok_end"])
    if i >= j:
        return float("nan")
    S = target_ids.shape[1]
    i = max(1, min(i, S-1))  # shift guard
    j = max(i, min(j, S))
    logits_slice = logits[:, i-1:j-1, :]
    targets_slice = target_ids[:, i:j]
    logp = F.log_softmax(logits_slice, dim=-1).gather(-1, targets_slice.unsqueeze(-1)).squeeze(-1)
    return float(logp.sum().item())

def make_set_null_hook(tok_slice: slice):
    """
    Forward hook that zeros residual states on tok_slice at each chosen layer.
    Works for blocks returning Tensor or tuple(Tensor,...).
    """
    def hook(module, inp, out):
        if isinstance(out, tuple):
            hs = out[0].clone()
            hs[:, tok_slice, :] = 0.0
            return (hs,) + out[1:]
        else:
            hs = out.clone()
            hs[:, tok_slice, :] = 0.0
            return hs
    return hook

# ------------------ core collection per record ------------------
def influence_matrix_for_record(model,
                                record: Dict[str, Any],
                                layers: List[int],
                                device: str) -> Optional[Dict[str, Any]]:
    """
    Compute step→step Δ matrix for one record.
    Returns a dict with metadata + matrix, or None if the record is invalid.
    """
    # Validate required fields
    if "token_ids_full" not in record or "step_spans" not in record:
        return None
    steps_all = record["step_spans"] or []
    # keep only steps with valid spans
    steps = []
    for s in steps_all:
        a = s.get("tok_start_abs"); b = s.get("tok_end_abs")
        if a is None or b is None: continue
        if isinstance(a, float): a = int(a)
        if isinstance(b, float): b = int(b)
        if a < 0 or b <= a: continue
        steps.append({**s, "tok_start_abs": int(a), "tok_end_abs": int(b), "step_idx": int(s.get("step_idx", len(steps)+1))})
    if not steps:
        return None
    # order by step_idx (1..N)
    steps = sorted(steps, key=lambda s: s["step_idx"])
    n = len(steps)

    # Prepare tensors
    full_ids = torch.tensor(record["token_ids_full"], dtype=torch.long, device=device).unsqueeze(0)

    # BASE logits (1 forward)
    with torch.no_grad():
        base_logits = model(full_ids).logits

    # Base logp per destination step (optional but useful)
    base_logp_per_step = []
    for B in steps:
        span_B = {"tok_start": B["tok_start_abs"], "tok_end": B["tok_end_abs"]}
        base_logp_per_step.append(gather_span_logprob(base_logits, full_ids, span_B))

    # Influence matrix M (rows=source A, cols=dest B)
    M = [[0.0 for _ in range(n)] for __ in range(n)]

    for i, A in enumerate(steps):
        # register hooks for A across chosen layers
        tok_slice_A = slice(A["tok_start_abs"], A["tok_end_abs"])
        handles = [model.model.layers[li].register_forward_hook(make_set_null_hook(tok_slice_A)) for li in layers]
        try:
            with torch.no_grad():
                ab_logits = model(full_ids).logits
            # fill row i
            for j, B in enumerate(steps):
                span_B = {"tok_start": B["tok_start_abs"], "tok_end": B["tok_end_abs"]}
                base_B = base_logp_per_step[j]
                abl_B  = gather_span_logprob(ab_logits, full_ids, span_B)
                M[i][j] = float(base_B - abl_B)  # Δ_{A→B}
        finally:
            for h in handles: h.remove()

    out = {
        "id": record.get("id", ""),
        "attempt": record.get("attempt", -1),
        "model": record.get("model", ""),
        "n_steps": n,
        "layer_band_used": None,          # filled by caller if helpful
        "layers": layers,                 # actual indices
        "step_indices": [s["step_idx"] for s in steps],
        "step_tok_spans": [[s["tok_start_abs"], s["tok_end_abs"]] for s in steps],
        "base_logp_per_step": base_logp_per_step,
        "matrix": M,                      # rows=source step, cols=dest step
    }
    return out

# ------------------ JSONL I/O ------------------
def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue

def write_jsonl(path: Path, records: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ------------------ CLI ------------------
def main():
    ap = argparse.ArgumentParser(description="Collect step→step influence matrices over all records.")
    ap.add_argument("--jsonl-in", type=str, required=True, help="Path to runs.jsonl (from your generation pass).")
    ap.add_argument("--out", type=str, required=True, help="Output JSONL for matrices (will be appended).")
    ap.add_argument("--model", type=str, required=True, help="HF model name used for teacher-forced scoring.")
    ap.add_argument("--hf-token", type=str, default="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")
    ap.add_argument("--layer-band", type=str, default="middle_third",
                    help="Layer band: all|first_third|middle_third|last_third|range:START:END")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit of records (0 = all).")
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")
    model, tok = load_model(args.model, device, args.hf_token)
    layers = choose_layers(model, args.layer_band)
    print(f"Layers used ({args.layer_band}): {layers}")

    src = Path(args.jsonl_in)
    dst = Path(args.out)

    buffer: List[Dict[str, Any]] = []
    n_written = 0

    for k, rec in enumerate(read_jsonl(src), 1):
        # skip records without steps
        if not rec.get("step_spans"):
            continue
        try:
            res = influence_matrix_for_record(model, rec, layers, device)
        except RuntimeError as e:
            print(f"[warn] RuntimeError on id={rec.get('id')} attempt={rec.get('attempt')}: {e}")
            res = None
        except Exception as e:
            print(f"[warn] Error on id={rec.get('id')} attempt={rec.get('attempt')}: {e}")
            res = None

        if res is None:
            continue

        res["layer_band_used"] = args.layer_band
        buffer.append(res)

        if len(buffer) >= 20:  # flush periodically
            write_jsonl(dst, buffer)
            n_written += len(buffer)
            print(f"[flush] wrote {n_written} matrices → {dst}")
            buffer = []

        if args.limit and n_written >= args.limit:
            break

    if buffer:
        write_jsonl(dst, buffer)
        n_written += len(buffer)

    print(f"Done. Wrote {n_written} records to {dst}")

if __name__ == "__main__":
    main()
