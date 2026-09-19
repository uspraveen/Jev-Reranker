"""Reranker Arena demo: retrieval-order vs Jev-Reranker side by side.

Usage:  streamlit run demo/arena.py   (or: python demo/arena.py --cli)
Stdlib-only CLI fallback included so the demo works without extra deps.
"""

from __future__ import annotations

import argparse
import json

from jev_reranker.judges import OfflineJudge
from jev_reranker.models import Candidate
from jev_reranker.reranker import JevReranker

SAMPLES = [
    {
        "query": "how long do refunds take to settle?",
        "candidates": [
            "Our patio furniture sale ends Sunday.",
            "Refunds settle within 5 business days.",
            "OUTDATED deprecated old version: refunds settle in 30 days.",
            "The refund FAQ table of contents.",
        ],
    },
    {
        "query": "search API rate limit?",
        "candidates": [
            "Search API allows 120 requests per minute per key.",
            "However, that is wrong: the search API never allows more than 5 requests.",
            "API keys are created in the dashboard.",
        ],
    },
]


def run_demo(query: str, texts: list[str]) -> dict[str, object]:
    rr = JevReranker(judge=OfflineJudge())
    cands = [Candidate(id=f"c{i}", text=t, retrieval_rank=i) for i, t in enumerate(texts)]
    res = rr.rerank(query, cands, mode="memory")
    return {
        "query": query,
        "judge": rr.client.judge.model_name,
        "retrieval_order": [c.id for c in cands],
        "reranked": [
            {"id": it.candidate.id, "label": it.label.value, "value": round(it.value, 3), "text": it.candidate.text}
            for it in res.items
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", action="store_true")
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()
    if args.cli:
        s = SAMPLES[args.sample % len(SAMPLES)]
        print(json.dumps(run_demo(s["query"], s["candidates"]), indent=2))
        return
    try:
        import streamlit as st

        st.title("Reranker Arena — retrieval order vs Jev-Reranker")
        idx = st.selectbox("sample", range(len(SAMPLES)), format_func=lambda i: SAMPLES[i]["query"])
        s = SAMPLES[int(idx)]
        query = st.text_input("query", s["query"])
        texts = st.text_area("candidates (one per line)", "\n".join(s["candidates"])).splitlines()
        if st.button("Rerank"):
            out = run_demo(query, [t for t in texts if t.strip()])
            st.json(out)
    except ImportError:
        s = SAMPLES[0]
        print(json.dumps(run_demo(s["query"], s["candidates"]), indent=2))


if __name__ == "__main__":
    main()
