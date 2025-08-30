import json
import random
from pathlib import Path
from typing import List, Dict, Any

import datasets


def normalize_numeric_answer(text: str) -> str:
    t = text.strip()
    # Strip trailing punctuation and commas
    while t and t[-1] in ",.;:\n\t " :
        t = t[:-1]
    # Remove surrounding quotes
    if (t.startswith("\"") and t.endswith("\"")) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1]
    return t


def extract_numeric_from_gsm8k_answer(ans: str) -> str:
    # GSM8K answer format often ends with "#### <number>"
    if "####" in ans:
        num = ans.split("####")[-1].strip()
        return normalize_numeric_answer(num)
    return normalize_numeric_answer(ans)


def load_gsm8k_split(split: str = "train") -> List[Dict[str, Any]]:
    ds = datasets.load_dataset("gsm8k", "main", split=split)
    items: List[Dict[str, Any]] = []
    for ex in ds:
        question = ex.get("question", "").strip()
        answer = ex.get("answer", "").strip()
        gold = extract_numeric_from_gsm8k_answer(answer)
        # Keep items with numeric gold answers
        if question and gold and any(ch.isdigit() for ch in gold):
            items.append({
                "id": ex.get("id", f"gsm8k-{split}-{len(items)}"),
                "problem": question,
                "gold_answer": gold,
                "source": "gsm8k/main",
            })
    return items


def main(n_samples: int = 40, out_path: str = "data/problems.sample.json", seed: int = 123):
    random.seed(seed)
    items = load_gsm8k_split("train")
    random.shuffle(items)
    selected = items[:n_samples]

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(selected)} problems to {out_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--out", type=str, default="data/problems.sample.json")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    main(n_samples=args.n, out_path=args.out, seed=args.seed)



