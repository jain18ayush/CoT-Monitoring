SYSTEM ROLE
You are an autonomous research engineer. Build and run a small pipeline that mines (correct, incorrect) Chain-of-Thought (CoT) pairs for the SAME prompt, with strict step formatting and full reproducibility.

OBJECTIVE
Produce ≥20 problems where the SAME prompt yields both:
  (A) one CORRECT CoT trace and
  (B) one INCORRECT CoT trace,
and save all artifacts (token IDs, step spans, decode params, seeds) for later activation-patching experiments.

CONSTRAINTS
- Total runtime target: ≤15 hours.
- Use an open HF causal LM with accessible internals: default `meta-llama/Meta-Llama-3.1-8B-Instruct`; fallback to `mistralai/Mistral-7B-Instruct-v0.3`; if memory is tight, fallback to `Qwen/Qwen2.5-3B-Instruct`.
- Batch size 1; max_new_tokens 256; top_p fixed at 0.95.
- Residual stream patching will happen later; for now just mine pairs + record spans.
- Reproducibility is mandatory: log model name, SHA/commit if available, decode params, and RNG seed for every attempt.

DATA
- Prefer GSM8K (main) or SVAMP via `datasets`. Keep items with numeric gold answers (easier scoring).
- Build `data/problems.sample.json` with 30–60 candidates; you will mine pairs until ≥20 pairs are found.

PROMPT TEMPLATE (must use exactly)
"""
You are a careful solver. Solve with explicit steps.

Format STRICTLY:
Step 1: ...
Step 2: ...
(add steps as needed)
Final Answer: <one token/number/string>

Problem: {QUESTION}
"""

PAIR-MINING PROCEDURE (per problem)
1) Generation schedule (same prompt each time):
   - Greedy: T=0.0 (1 attempt).
   - High variance: T ∈ {0.9, 1.0}, two attempts each.
   - Mid variance: T ∈ {0.7, 0.5}, two attempts each.
   - Stop as soon as you have one CORRECT and one INCORRECT trace, or after 8 attempts; if none, skip the problem.
2) Use fixed seeds: seed = hash(problem_id, attempt_index) (record exact integer).
3) Parse each output:
   - Split steps by regex `^Step\s+\d+:\s*` (multiline).
   - Extract final answer by regex `Final Answer:\s*(.+)`.
   - Normalize answers: trim/strip punctuation; if both numeric, compare numerically; else case-insensitive string compare.
4) A “pair found” = one attempt marked correct, one attempt marked incorrect, for the SAME prompt.

WHAT TO SAVE
- `results/runs.jsonl`: one line per attempt with fields:
  {
    "id": "<problem id>",
    "attempt": <int>,
    "temperature": <float>,
    "top_p": 0.95,
    "seed": <int>,
    "model": "<repo/name>",
    "prompt": "...",
    "text": "... FULL CoT ...",
    "final_answer": "...",
    "correct": true/false,
    "parse_ok": true/false,
    "token_ids": [ints],
    "step_spans": [
      {"step_idx": 1, "char_start": a, "char_end": b, "tok_start": t0, "tok_end": t1, "text": "..."},
      ...
    ],
    "final_answer_span": {"tok_start": u, "tok_end": v}
  }
  Notes:
  - Record both char offsets and token spans. Token spans must align within the full output tokenization.
  - If parsing fails, set parse_ok=false and correct=false but still log the run.

- `results/pairs.jsonl`: one line per problem with a found pair:
  {
    "id": "...",
    "problem": "...",
    "gold_answer": "...",
    "correct_ref": {"attempt": k1, "runs_offset": <byte_offset_or_index>},
    "incorrect_ref": {"attempt": k2, "runs_offset": <byte_offset_or_index>},
    "steps_selected": {
      "early": {"from": "correct", "step_idx": <int>},
      "late":  {"from": "correct", "step_idx": <int>}
    },
    "decode_config": {"top_p": 0.95, "max_new_tokens": 256}
  }
  Selection rule: pick exactly TWO steps from the CORRECT trace—(i) the first substantive step (skip empty fluff), and (ii) the last reasoning step immediately before “Final Answer:”.

- `results/pairs_summary.csv`: columns [id, n_attempts, early_step_idx, late_step_idx, correct_attempt, incorrect_attempt].

ENGINEERING CHECKLIST
- Implement generator with reproducible seeding (Torch `Generator`).
- Implement step/answer parsing and numeric/string normalization.
- Tokenization + spans: store full output tokens; compute per-step token ranges by re-encoding step substrings and aligning; also store char offsets as a fallback.
- Robustness: if a trace lacks “Final Answer:” or any “Step i:”, mark parse_ok=false and treat as incorrect.
- Speed: prefer BF16/FP16; use 4-bit load if VRAM limited; if OOM, automatically fallback to the next smaller model and note it in logs.

EXIT / ACCEPTANCE CRITERIA
- ≥20 problems with valid pairs in `results/pairs.jsonl`.
- Average ≤8 attempts/problem among paired items.
- All paired runs reproducible by re-running with the logged model, prompt, T, top_p, and seed.
- Print a final summary table and write it to `results/summary.json`:
  {
    "pairs_found": N,
    "problems_processed": M,
    "avg_attempts_per_paired_item": ...,
    "temperatures_by_outcome": {"correct": {...}, "incorrect": {...}}
  }

DELIVERABLES
- Code (Python 3.10+) with a CLI `mine_pairs.py`:
  usage:
    python mine_pairs.py --problems data/problems.sample.json \
      --model meta-llama/Meta-Llama-3.1-8B-Instruct \
      --outdir results/ --max-tries 8 --min-pairs 20 \
      --temperature-seq 0.0,0.9,1.0,0.7,0.5 --top-p 0.95 --max-new-tokens 256
- `data/problems.sample.json` built from GSM8K or SVAMP (20–60 numeric-answer items).
- `results/runs.jsonl`, `results/pairs.jsonl`, `results/pairs_summary.csv`, `results/summary.json`, and a brief `README.md` with how to reproduce a specific pair.

BEGIN NOW. If a step is infeasible due to environment limits, degrade gracefully (smaller model; fewer items) but keep the SAME protocol and output schema. Do not change the prompt template.
