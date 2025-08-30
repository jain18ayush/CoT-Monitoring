#!/usr/bin/env python3
"""
Ablate saved CoT steps (teacher-forced), log deltas, and visualize what was zeroed.

Input:
  --jsonl  path to runs.jsonl (from your mining script)

Output:
  --out    ablation_results.jsonl (appended)
  optional HTML files per step if --highlight-dir is set.

Example:
  python ablate_steps.py \
    --jsonl MATS2025_WINTER_RESULTS/runs.jsonl \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --layers middle_third \
    --highlight-dir MATS2025_WINTER_RESULTS/highlights \
    --print-ansi
"""

import argparse, json, os, re, html
from pathlib import Path
from typing import List, Dict, Any, Optional

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
        torch_dtype=dtype,
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
        try:
            _, a, b = band.split(":")
            a, b = int(a), int(b)
        except Exception:
            a, b = 0, n
        a = max(0, min(a, n))
        b = max(0, min(b, n))
        return list(range(a, b))
    return list(range(n//3, 2*n//3))

# ------------------ Scoring ------------------
def gather_answer_logprob(logits: torch.Tensor,
                          target_ids: torch.Tensor,
                          span: Dict[str, int]) -> float:
    """
    logits: [1, S, V]
    target_ids: [1, S]
    span: {"tok_start": i, "tok_end": j} (absolute indices)
    Returns sum log p(target_ids[i..j-1] | prefix).
    """
    i, j = int(span["tok_start"]), int(span["tok_end"])
    if i >= j:
        return float("nan")
    S = target_ids.shape[1]
    i = max(1, min(i, S - 1))  # guard for shift
    j = max(i, min(j, S))
    # Next-token prediction: logits at t-1 predict token at t
    logits_slice = logits[:, i-1:j-1, :]            # [1, L, V]
    targets_slice = target_ids[:, i:j]              # [1, L]
    logp = F.log_softmax(logits_slice, dim=-1).gather(-1, targets_slice.unsqueeze(-1)).squeeze(-1)
    return float(logp.sum().item())

# ------------------ Hooks (necessity: set null) ------------------
def make_set_null_hook(tok_slice: slice):
    """
    Forward hook that zeros the residual stream on tok_slice for each chosen layer.
    Works for blocks returning Tensor or tuple(Tensor, ...).
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
            logits = out.logits
        return gather_answer_logprob(logits, input_ids, answer_span)
    finally:
        for h in handles:
            h.remove()

# ------------------ JSONL I/O ------------------
def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)

def write_jsonl(path: Path, records: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ------------------ Highlight helpers ------------------
def _ansi_wrap(s, color="yellow", bold=True):
    colors = {"red":31,"green":32,"yellow":33,"blue":34,"magenta":35,"cyan":36,"white":37}
    c = colors.get(color, 33)
    pre = f"\x1b[{1 if bold else 0};{c}m"
    post = "\x1b[0m"
    return f"{pre}{s}{post}"

def print_step_highlight_ansi(record: dict, step_idx: int, pad_context: int = 100):
    """
    Terminal preview: highlight the ablated step using character spans (relative to completion).
    """
    text = record.get("text","")
    spans = sorted(record.get("step_spans", []), key=lambda s: s["char_start"])
    target = None
    for s in spans:
        if s.get("step_idx") == step_idx:
            target = s
            break
    if not target:
        print(f"[highlight] step_idx={step_idx} not found for {record.get('id')}")
        return

    a, b = int(target["char_start"]), int(target["char_end"])
    start = max(0, a - pad_context)
    end   = min(len(text), b + pad_context)

    before = text[start:a]
    core   = text[a:b]
    after  = text[b:end]

    print(f"[{record.get('id')}] attempt={record.get('attempt')} step={step_idx} "
          f"(chars {a}:{b}, toks {target['tok_start_abs']}:{target['tok_end_abs']})")
    print(before + _ansi_wrap(core, "yellow", True) + after)
    print("-"*80)

def save_step_highlight_html(record: dict, step_idx: int, layers: List[int], out_dir: str):
    """
    Save an HTML file with all steps lightly highlighted; the ablated one is emphasized.
    """
    out_dirp = Path(out_dir)
    out_dirp.mkdir(parents=True, exist_ok=True)

    text = record.get("text","")
    spans = sorted(record.get("step_spans", []), key=lambda s: s["char_start"])
    if not spans:
        return None

    # Build HTML by slicing the completion text on character spans
    html_parts = []
    pos = 0
    for s in spans:
        cs, ce = int(s["char_start"]), int(s["char_end"])
        # plain segment
        if cs > pos:
            html_parts.append(f"<span class='plain'>{html.escape(text[pos:cs])}</span>")
        # step segment
        cls = "step"
        if s.get("step_idx") == step_idx:
            cls += " zeroed"
        label = f"[Step {s.get('step_idx','?')}]"
        body = text[cs:ce]
        html_parts.append(
            f"<span class='{cls}' data-tok='{s['tok_start_abs']}:{s['tok_end_abs']}'>"
            f"<span class='label'>{html.escape(label)}</span>"
            f"<span class='tok'>[{s['tok_start_abs']}:{s['tok_end_abs']})</span> "
            f"{html.escape(body)}"
            f"</span>"
        )
        pos = ce
    if pos < len(text):
        html_parts.append(f"<span class='plain'>{html.escape(text[pos:])}</span>")

    # final answer line (optional)
    m = re.search(r"(Final Answer:\s*[^\n\r]+)", text, flags=re.IGNORECASE)
    fa_html = f"<div class='final-answer'>{html.escape(m.group(1))}</div>" if m else ""

    layers_str = ",".join(map(str, layers))
    meta_html = (
        f"<div class='meta'>"
        f"<div><b>ID</b>: {html.escape(str(record.get('id','')))}</div>"
        f"<div><b>Attempt</b>: {html.escape(str(record.get('attempt','')))}</div>"
        f"<div><b>Model</b>: {html.escape(str(record.get('model','')))}</div>"
        f"<div><b>Patched layers</b>: {html.escape(layers_str)}</div>"
        f"</div>"
    )

    css = """
    <style>
    body { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; line-height: 1.45; padding: 16px; }
    .plain { color: #222; white-space: pre-wrap; }
    .step  { background: rgba(0, 128, 255, 0.10); border-radius: 4px; padding: 0 2px; }
    .step .label { color: #0366d6; font-weight: 600; margin-right: 6px; }
    .step .tok { color: #666; font-size: 12px; margin-left: 6px; }
    .step.zeroed { background: rgba(255, 0, 0, 0.15); outline: 2px solid rgba(255,0,0,0.35); }
    .final-answer { margin-top: 12px; padding: 6px 8px; background: rgba(0,200,0,0.12); border-left: 3px solid #0a0; }
    .meta { margin-bottom: 12px; padding: 8px; background: #fafafa; border: 1px solid #eee; border-radius: 6px; }
    </style>
    """

    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(str(record.get('id','')))} step {step_idx}</title>{css}</head>
<body>
{meta_html}
<div class="content">{''.join(html_parts)}</div>
{fa_html}
</body></html>"""

    fname = f"{record.get('id','sample')}_att{record.get('attempt',0)}_step{step_idx}.html"
    out_path = out_dirp / fname
    with out_path.open("w", encoding="utf-8") as f:
        f.write(html_doc)
    return out_path.as_posix()

# ------------------ Main ablation per record ------------------
def ablate_record_steps(model, record: Dict[str, Any], layers: List[int], device: str,
                        print_ansi: bool=False, highlight_dir: Optional[str]=None):
    """
    Yields per-step results for one record and (optionally) prints/saves highlights.
    """
    if "token_ids_full" not in record or "final_answer_span" not in record or "step_spans" not in record:
        return
    full_ids = record["token_ids_full"]
    ans_span = record["final_answer_span"]
    steps = record["step_spans"]
    if not steps:
        return

    # Make tensors
    input_ids = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)

    # Base score (no patch)
    base_logp = score_with_optional_patch(model, input_ids, ans_span, patch_hook=None, layers=None)

    for s in steps:
        s_idx = s.get("step_idx", None)
        a, b = s.get("tok_start_abs", None), s.get("tok_end_abs", None)
        if s_idx is None or a is None or b is None or not (0 <= a < b <= len(full_ids)):
            continue

        tok_slice = slice(a, b)
        patch = make_set_null_hook(tok_slice)
        abl_logp = score_with_optional_patch(model, input_ids, ans_span, patch_hook=patch, layers=layers)

        # OPTIONAL: visualization
        if print_ansi:
            print_step_highlight_ansi(record, s_idx, pad_context=100)
        if highlight_dir:
            saved = save_step_highlight_html(record, s_idx, layers, highlight_dir)
            if saved:
                print(f"[highlight] saved → {saved}")

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
            "delta_logp_answer": base_logp - abl_logp,  # >0 => step necessary (under replay)
        }

import json, torch, torch.nn.functional as F
from typing import Dict, List, Any

# --------- scoring utilities (generic span, not just final answer) ---------
def gather_span_logprob(logits: torch.Tensor,
                        target_ids: torch.Tensor,
                        span: Dict[str, int]) -> float:
    """
    logits: [1, S, V]; target_ids: [1, S]
    span: {"tok_start": i, "tok_end": j} absolute indices into target_ids
    Returns sum log p(target_ids[i..j-1] | prefix).
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
    """Zero the residual stream on tok_slice at the chosen layer(s)."""
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

def score_span_with_patch(model,
                          input_ids: torch.Tensor,
                          span: Dict[str, int],
                          patch_hook=None,
                          layers: List[int]=None) -> float:
    """Teacher-forced run; optionally patches residuals on given layers; scores span log-prob."""
    handles = []
    try:
        if patch_hook and layers:
            for li in layers:
                handles.append(model.model.layers[li].register_forward_hook(patch_hook))
        with torch.no_grad():
            out = model(input_ids=input_ids)
            logits = out.logits
        return gather_span_logprob(logits, input_ids, span)
    finally:
        for h in handles: h.remove()

# --------- convenience: choose layers (same as before) ---------
def choose_layers(model, band: str="middle_third") -> List[int]:
    n = len(model.model.layers)
    if band == "all": return list(range(n))
    if band == "first_third": return list(range(0, n//3))
    if band == "middle_third": return list(range(n//3, 2*n//3))
    if band == "last_third": return list(range(2*n//3, n))
    if band.startswith("range:"):
        _, a, b = band.split(":"); a, b = int(a), int(b)
        return list(range(max(0,a), min(n,b)))
    return list(range(n//3, 2*n//3))

# ---------  A → B effect for a single record  ---------
def step_to_step_delta(model,
                       record: Dict[str, Any],
                       src_step_idx: int,
                       dst_step_idx: int,
                       layers: List[int],
                       device: str="cuda") -> float:
    """
    Δ_{A→B}: sum log p_B(base) - sum log p_B(ablated) where ablation = zero A residuals on 'layers'.
    """
    steps = record["step_spans"]
    # find src/dst spans
    src = next(s for s in steps if s["step_idx"] == src_step_idx)
    dst = next(s for s in steps if s["step_idx"] == dst_step_idx)

    tok_slice_A = slice(int(src["tok_start_abs"]), int(src["tok_end_abs"]))
    span_B = {"tok_start": int(dst["tok_start_abs"]), "tok_end": int(dst["tok_end_abs"])}

    full_ids = torch.tensor(record["token_ids_full"], dtype=torch.long, device=device).unsqueeze(0)

    base = score_span_with_patch(model, full_ids, span_B, patch_hook=None, layers=None)
    patch = make_set_null_hook(tok_slice_A)
    ablated = score_span_with_patch(model, full_ids, span_B, patch_hook=patch, layers=layers)

    return base - ablated  # >0 ⇒ A supports B

# ---------  Per-layer curve for A → B  ---------
def step_to_step_by_layer(model, record, src_step_idx, dst_step_idx, layer_band="middle_third", device="cuda"):
    layers = choose_layers(model, layer_band)
    steps = record["step_spans"]
    src = next(s for s in steps if s["step_idx"] == src_step_idx)
    dst = next(s for s in steps if s["step_idx"] == dst_step_idx)

    tok_slice_A = slice(int(src["tok_start_abs"]), int(src["tok_end_abs"]))
    span_B = {"tok_start": int(dst["tok_start_abs"]), "tok_end": int(dst["tok_end_abs"])}
    full_ids = torch.tensor(record["token_ids_full"], dtype=torch.long, device=device).unsqueeze(0)

    base = score_span_with_patch(model, full_ids, span_B, None, None)
    deltas = []
    for li in layers:
        ablated = score_span_with_patch(model, full_ids, span_B, make_set_null_hook(tok_slice_A), [li])
        deltas.append((li, base - ablated))
    return deltas  # list of (layer_index, delta)

# ---------  Influence matrix for one trace (all A,B)  ---------
import numpy as np
def influence_matrix_for_record(model, record, layer_band="middle_third", device="cuda"):
    layers = choose_layers(model, layer_band)
    full_ids = torch.tensor(record["token_ids_full"], dtype=torch.long, device=device).unsqueeze(0)

    steps = sorted(record["step_spans"], key=lambda s: s["step_idx"])
    n = len(steps)
    M = np.zeros((n, n), dtype=float)

    # Precompute base logp for every B (one forward pass)
    with torch.no_grad():
        logits = model(full_ids).logits
    for j, B in enumerate(steps):
        span_B = {"tok_start": int(B["tok_start_abs"]), "tok_end": int(B["tok_end_abs"])}
        M[j, j] = gather_span_logprob(logits, full_ids, span_B)  # store base for diag if you want

    # Now do A→B by patching A (one patch per A) and re-scoring each B
    for i, A in enumerate(steps):
        tok_slice_A = slice(int(A["tok_start_abs"]), int(A["tok_end_abs"]))
        # one forward with patch A (on chosen layers)
        handles = [model.model.layers[li].register_forward_hook(make_set_null_hook(tok_slice_A)) for li in layers]
        with torch.no_grad():
            logits_ab = model(full_ids).logits
        for j, B in enumerate(steps):
            span_B = {"tok_start": int(B["tok_start_abs"]), "tok_end": int(B["tok_end_abs"])}
            base_B = gather_span_logprob(logits, full_ids, span_B)
            abl_B  = gather_span_logprob(logits_ab, full_ids, span_B)
            M[i, j] = base_B - abl_B    # row i = source step, col j = destination step
        for h in handles: h.remove()

    return M  # shape [n_steps, n_steps]

