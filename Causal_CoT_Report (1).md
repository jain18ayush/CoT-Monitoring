# Causal CoT Experiment Report
**Teacher-forced, step→step (causal) analysis using residual zeroing**
Date: (generated now)

---
## Methods (recap)
- **Teacher forcing (replay):** model runs on exactly the saved tokens, fixing step boundaries.
- **Necessity ablation:** zero residual stream on the source step's token span (chosen layer band), then measure Δ log-prob on target tokens.
- **Step→Step matrix:** entry M[i,j] = Δ on step j when step i is zeroed. Upper-triangular (i<j) is forward influence; lower-triangular (i>j) would be backward (should be ~0 for causal decoder).

## Data & Files
- Step→step input: `/mnt/data/causal.jsonl` (10 traces; step counts = [3, 5, 6, 7, 11])
- Plots directory: `/mnt/data/plots`
- Directionality summary: [directionality_summary.csv](plots/step2step/directionality_summary.csv)

## Results
### Directionality
- **Forward dominance:** mean upper-share = **1.0000**, median = **1.0000**; fraction of traces with upper-share>0.55 = **1.0000**.
- **Backward mass:** mean pos fraction in lower triangle = **0.0000** (all lower entries were zero in this dataset).
- Average heatmap (most common n): [view](plots/step2step/avg_heatmap_n3.png)

### Proximity (Lag j−i)
- Edge **fractions** by lag (j−i): lag1=0.2919, lag2=0.2298, lag3=0.1677, lag4=0.1242 (then tapering).
- Mean **strength** by lag: lag1≈**25.9073**, lag2≈**2.1279**, lag3≈**1.6419**, lag4≈**1.2745**.

### Flow centrality (position vs inflow/outflow)
- Corr(step index, inflow) ≈ **0.7490** (later steps receive more).
- Corr(step index, outflow) ≈ **-0.7671** (earlier steps send more).
- Example inflow chart: [view](plots/step2step/inflow_trace0_n3.png)
- Example outflow chart: [view](plots/step2step/outflow_trace0_n3.png)

### Magnitudes
- Mean |Δ| per matrix (overall): **132.6264** (units: summed token log-prob deltas per destination step).
- By step count n: n=3: 259.6629, n=5: 116.0914, n=6: 108.4100, n=7: 37.4251, n=11: 31.1036
## Ablation vs Final Answer
- Source file: `ablation_results_v2 (1).jsonl` with **57** per-step ablations.
- **Summary:** mean Δ = **12.7699**, median = **1.3135**, std = **25.5867**; frac(+): **0.6667**, frac(−): **0.3333**.
- Histogram: [view](plots/ablations/ablation_delta_hist.png)
- Boxplot by step: [view](plots/ablations/ablation_delta_box_by_step.png)
- Scatter Δ vs step: [view](plots/ablations/ablation_delta_scatter_step.png)
- Top +Δ (50): [CSV](plots/ablations/top_50_deltas_positive.csv)
- Top −Δ (50): [CSV](plots/ablations/top_50_deltas_negative.csv)
- Per-step medians (top 10): [CSV](plots/ablations/per_step_medians_top10.csv)

## Interpretation
- The matrices are **strongly upper-triangular**: earlier steps support later steps; no measurable late→early effect (as expected for a causal decoder under teacher forcing).
- **Short-range edges dominate**: most helpful influence is from a step to the next step (lag=1), with rapidly diminishing strength at longer lags.
- **Role structure**: early steps behave as **sources** (high outflow), later steps are **sinks/aggregators** (high inflow).

## Verification & Limitations
- Sanity checks to keep running for each new batch: (i) lower-triangle near-zero; (ii) span alignment visualizations; (iii) no-op ≈ base; (iv) random-span and wrong-layer ≈ tiny effects.
- Magnitudes depend on token counts per step and the chosen layer band; consider normalizing Δ by tokens in the destination step for cross-trace comparability.
- Current report aggregates only the step→step matrices; final-answer ablations are pending the correct file.

## Next Steps
- Add **final-answer** ablation plots and correlate with step→step centrality.
- Compute **indirect vs direct** support by comparing A→final with path A→(late steps)→final.
- Explore **layer bands** (early/mid/late) to localize where edges live; produce per-band heatmaps.
- Normalize Δ by destination token count; add variance estimates via repeated runs.