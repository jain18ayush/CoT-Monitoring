
import json, os, math, hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PLOTS_DIR = Path("plots")
ABL_DIR = PLOTS_DIR / "ablations"
S2S_DIR = PLOTS_DIR / "step2step"
for d in [PLOTS_DIR, ABL_DIR, S2S_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def exists(p: str) -> bool:
    try:
        return Path(p).exists()
    except Exception:
        return False

def read_jsonl(path: str):
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs

def detect_ablation_schema(rec: Dict[str, Any]) -> bool:
    keys = set(rec.keys())
    return any(k in keys for k in ["delta_logp_answer", "ablated_logp_answer"]) and "step_idx" in keys

def detect_step2step_schema(rec: Dict[str, Any]) -> bool:
    keys = set(rec.keys())
    return "matrix" in keys and isinstance(rec.get("matrix"), (list, tuple))

def pick_first_existing(candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if exists(c):
            return c
    return None

def plot_ablation(ablation_path: str):
    recs = read_jsonl(ablation_path)
    if not recs:
        print(f"[Ablation] No records in {ablation_path}")
        return
    if not detect_ablation_schema(recs[0]):
        if not any(detect_ablation_schema(r) for r in recs[:20]):
            print(f"[Ablation] {ablation_path} doesn't look like ablation schema.")
            return

    rows = []
    for r in recs:
        if not detect_ablation_schema(r):
            continue
        rows.append({
            "step_idx": r.get("step_idx"),
            "delta_logp_answer": r.get("delta_logp_answer"),
            "base_logp_answer": r.get("base_logp_answer"),
            "ablated_logp_answer": r.get("ablated_logp_answer"),
            "layers": r.get("layers"),
            "run_id": r.get("run_id") or r.get("trace_id") or r.get("uid")
        })
    import pandas as pd
    df = pd.DataFrame(rows).dropna(subset=["delta_logp_answer"])
    if df.empty:
        print("[Ablation] DataFrame is empty after parsing.")
        return

    # Hist
    plt.figure()
    plt.hist(df["delta_logp_answer"].values, bins=50)
    plt.title("Ablation Δ log-prob (final answer)")
    plt.xlabel("Δ log-prob (base − ablated)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(ABL_DIR / "ablation_delta_hist.png")
    plt.close()

    # Boxplot by step
    top_steps = df["step_idx"].value_counts().sort_index().index.tolist()
    top_steps = top_steps[:20] if len(top_steps) > 20 else top_steps
    groups = [df.loc[df["step_idx"] == s, "delta_logp_answer"].values for s in top_steps]
    if groups:
        plt.figure()
        plt.boxplot(groups, labels=[str(s) for s in top_steps], showfliers=False)
        plt.title("Δ by step index (boxplot)")
        plt.xlabel("step_idx")
        plt.ylabel("Δ log-prob of final answer")
        plt.tight_layout()
        plt.savefig(ABL_DIR / "ablation_delta_box_by_step.png")
        plt.close()

    # Scatter
    plt.figure()
    plt.scatter(df["step_idx"].values, df["delta_logp_answer"].values, s=6)
    plt.title("Δ vs step index")
    plt.xlabel("step_idx")
    plt.ylabel("Δ log-prob (final answer)")
    plt.tight_layout()
    plt.savefig(ABL_DIR / "ablation_delta_scatter_step.png")
    plt.close()

    df.sort_values("delta_logp_answer", ascending=False).head(50).to_csv(ABL_DIR / "top_50_deltas_positive.csv", index=False)
    df.sort_values("delta_logp_answer", ascending=True).head(50).to_csv(ABL_DIR / "top_50_deltas_negative.csv", index=False)

def plot_matrix(ax, M: np.ndarray, title: str):
    im = ax.imshow(M, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Destination step (j)")
    ax.set_ylabel("Source step (i)")

def trace_hash(rec: Dict[str, Any]) -> str:
    h = hashlib.sha1(json.dumps(rec, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    return h

def upper_lower_sums(M: np.ndarray):
    n = M.shape[0]
    upper = 0.0
    lower = 0.0
    for i in range(n):
        for j in range(n):
            if i < j:
                upper += M[i, j]
            elif i > j:
                lower += M[i, j]
    return float(upper), float(lower)

def inflow_outflow(M: np.ndarray):
    outflow = M.sum(axis=1) - np.diag(M)
    inflow = M.sum(axis=0) - np.diag(M)
    return inflow, outflow

def plot_step2step(s2s_path: str):
    recs = read_jsonl(s2s_path)
    if not recs:
        print(f"[Step2Step] No records in {s2s_path}")
        return
    if not detect_step2step_schema(recs[0]):
        if not any(detect_step2step_schema(r) for r in recs[:20]):
            print(f"[Step2Step] {s2s_path} doesn't look like step→step schema.")
            return

    parsed = []
    for idx, r in enumerate(recs):
        if not detect_step2step_schema(r):
            continue
        M = np.array(r["matrix"], dtype=float)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            continue
        parsed.append({"idx": idx, "n": int(M.shape[0]), "M": M, "hash": trace_hash(r)})
    if not parsed:
        print("[Step2Step] No valid square matrices found.")
        return

    # Per-trace heatmaps (first 12)
    for t in parsed[:12]:
        fig, ax = plt.subplots()
        plot_matrix(ax, t["M"], f"Step→Step Δ (trace {t['idx']} | n={t['n']})")
        plt.tight_layout()
        plt.savefig(S2S_DIR / f"heatmap_trace{t['idx']}_n{t['n']}_{t['hash']}.png")
        plt.close(fig)

    # Aggregate by most common n
    sizes = pd.Series([t["n"] for t in parsed])
    common_n = int(sizes.mode().iloc[0])
    group = [t for t in parsed if t["n"] == common_n]
    if group:
        stack = np.stack([t["M"] for t in group], axis=0)
        mean_M = np.nanmean(stack, axis=0)
        fig, ax = plt.subplots()
        plot_matrix(ax, mean_M, f"Average Step→Step Δ (n={common_n}, k={len(group)})")
        plt.tight_layout()
        plt.savefig(S2S_DIR / f"avg_heatmap_n{common_n}.png")
        plt.close(fig)

        # Inflow/Outflow for first
        inflow, outflow = inflow_outflow(group[0]["M"])
        plt.figure()
        plt.bar(np.arange(len(outflow)), outflow)
        plt.title(f"Outflow by step (trace {group[0]['idx']}, n={common_n})")
        plt.xlabel("step")
        plt.ylabel("Σ_j Δ[i→j]")
        plt.tight_layout()
        plt.savefig(S2S_DIR / f"outflow_trace{group[0]['idx']}_n{common_n}.png")
        plt.close()

        plt.figure()
        plt.bar(np.arange(len(inflow)), inflow)
        plt.title(f"Inflow by step (trace {group[0]['idx']}, n={common_n})")
        plt.xlabel("step")
        plt.ylabel("Σ_i Δ[i→j]")
        plt.tight_layout()
        plt.savefig(S2S_DIR / f"inflow_trace{group[0]['idx']}_n{common_n}.png")
        plt.close()

    # Directionality summary
    rows = []
    for t in parsed:
        up, low = upper_lower_sums(t["M"])
        denom = abs(up) + abs(low)
        share = (up / denom) if denom > 0 else np.nan
        rows.append({"trace_idx": t["idx"], "n": t["n"], "upper_sum": up, "lower_sum": low, "upper_share": share})
    pd.DataFrame(rows).to_csv(S2S_DIR / "directionality_summary.csv", index=False)

def main(
    ablation_candidates=None,
    step2step_candidates=None
):
    if ablation_candidates is None:
        ablation_candidates = [
            "ablation.jsonl",
        ]
    if step2step_candidates is None:
        step2step_candidates = [
            "causal.jsonl",
        ]

    abl = pick_first_existing(ablation_candidates)
    s2s = pick_first_existing(step2step_candidates)

    print("# Plotting Causal CoT files")
    if abl:
        print(f"[Ablation] Using: {abl}")
        plot_ablation(abl)
    else:
        print("[Ablation] No file found. Skipping.")

    if s2s:
        print(f"[Step2Step] Using: {s2s}")
        plot_step2step(s2s)
    else:
        print("[Step2Step] No file found. Skipping.")

    print(f"\nOutputs saved under: {PLOTS_DIR.resolve()}")

if __name__ == "__main__":
    main()
