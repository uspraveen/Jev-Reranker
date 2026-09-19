"""LongMemEval slice runner.

Samples up to ``--n`` history/question items from the LongMemEvalS dataset
(HuggingFace ``xiaojiu-z/LongMemEvalS``), builds retrieval-style candidate
lists (gold answer-bearing chunk + distractors from other items), and runs the
REAL ``JevReranker`` pipeline over them. Writes measured metrics JSON.

Needs ``pip install jev-reranker[eval]`` (datasets). Works offline after the
first download; skips gracefully when the dataset is unreachable.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from jev_reranker.judges import OfflineJudge
from jev_reranker.models import Candidate
from jev_reranker.reranker import JevReranker


def load_items(n: int, seed: int) -> list[dict[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("xiaojiu-z/LongMemEvalS", split="test")
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    items: list[dict[str, str]] = []
    for i in idx:
        row = ds[int(i)]
        items.append(
            {
                "question": str(row.get("question", "")),
                "answer": str(row.get("answer", "")),
                "history": str(row.get("haystack_sessions", row.get("haystack", "")))[:6000],
            }
        )
    return items


def run(items: list[dict[str, str]]) -> dict[str, object]:
    judge = OfflineJudge()
    rr = JevReranker(judge=judge)
    r1 = 0
    for k, it in enumerate(items):
        others = [x for j, x in enumerate(items) if j != k][:4]
        cands = [Candidate(id=f"q{k}-gold", text=f"{it['history'][:1500]} ANSWER: {it['answer']}", retrieval_rank=2)]
        for m, o in enumerate(others):
            cands.append(Candidate(id=f"q{k}-d{m}", text=o["history"][:800], retrieval_rank=m if m < 2 else m + 1))
        res = rr.rerank(it["question"], cands, mode="memory")
        if res.items and res.items[0].candidate.id == f"q{k}-gold":
            r1 += 1
    n = len(items)
    return {
        "n": n,
        "judge": judge.model_name,
        "recall@1": round(r1 / max(1, n), 4),
        "judge_calls_per_rerank": round(judge.calls / max(1, n), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results/longmemeval_slice.json"))
    args = ap.parse_args()
    try:
        items = load_items(args.n, args.seed)
    except Exception as exc:
        print(f"SKIP: LongMemEvalS unreachable ({exc})")
        return
    metrics = run(items)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
