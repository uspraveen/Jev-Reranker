"""Reranker Arena demo: retrieval-order vs Jev-Reranker side by side.

Usage:
  streamlit run demo/arena.py          # interactive UI
  python demo/arena.py --cli           # stdlib CLI
  python demo/arena.py --html out.html # static HTML report (no deps)

Uses live Jev when TYPESAFE_API_KEY is set, else the offline lexical judge
(clearly labeled). Shows per-memory head bars (relevance/utility/superseded/
conflict) and a stats block: judge calls, latency, context tokens.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from jev_reranker.judges import OfflineJudge
from jev_reranker.models import Candidate
from jev_reranker.reranker import JevReranker, candidate_tokens

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
    {
        "query": "who is oncall for the database?",
        "candidates": [
            "Priya is the primary oncall for postgres-cluster-2 (updated 2026-09-10).",
            "Marcus was the primary oncall for postgres-cluster-2 (2024 rotation).",
            "postgres-cluster-2 backups run nightly at 02:00 UTC.",
            "no longer valid, replaced: escalations page the old DBA list (2023).",
        ],
    },
]


def build_judge() -> Any:
    import os

    if os.environ.get("TYPESAFE_API_KEY"):
        from jev_reranker.judges import LiveJevJudge

        return LiveJevJudge()
    return OfflineJudge()


def run_demo(query: str, texts: list[str], budget_tokens: int = 120) -> dict[str, Any]:
    rr = JevReranker(judge=build_judge())
    cands = [Candidate(id=f"c{i}", text=t, retrieval_rank=i) for i, t in enumerate(texts)]

    t0 = time.perf_counter()
    res = rr.rerank(query, cands, mode="memory")
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    t1 = time.perf_counter()
    sel = rr.select(query, cands, budget_tokens=budget_tokens)
    sel_ms = round((time.perf_counter() - t1) * 1000, 1)

    tokens_before = sum(candidate_tokens(c) for c in cands)
    return {
        "query": query,
        "judge": rr.client.judge.model_name,
        "judge_calls": rr.judge_calls,
        "latency_ms": latency_ms,
        "selection_latency_ms": sel_ms,
        "usage": res.usage,
        "context": {
            "budget_tokens": budget_tokens,
            "tokens_before": tokens_before,
            "tokens_selected": sel.total_tokens,
            "selected_ids": [it.candidate.id for it in sel.selected],
            "suppressed_near_duplicates": [it.candidate.id for it in sel.suppressed_near_duplicates],
        },
        "retrieval_order": [c.id for c in cands],
        "reranked": [
            {
                "id": it.candidate.id,
                "label": it.label.value,
                "value": round(it.value, 3),
                "heads": {
                    "relevance": round(it.judgments.relevance / 3, 2),
                    "utility": round(it.judgments.utility / 3, 2),
                    "superseded": round(it.judgments.superseded, 2),
                    "conflict": round(it.judgments.conflict, 2),
                },
                "text": it.candidate.text,
            }
            for it in res.items
        ],
    }


HEAD_COLORS = {
    "relevance": "#2563eb",
    "utility": "#0891b2",
    "superseded": "#d97706",
    "conflict": "#dc2626",
}
LABEL_COLORS = {
    "USE": "#16a34a",
    "KEEP": "#65a30d",
    "DROP": "#6b7280",
    "STALE": "#d97706",
    "CONFLICT": "#dc2626",
    "UNCERTAIN": "#7c3aed",
}


def render_html(sample_results: list[dict[str, Any]]) -> str:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Reranker Arena - Jev-Reranker</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:2rem;background:#f8fafc;color:#0f172a}",
        "h1{font-size:1.4rem} h2{font-size:1.1rem;margin-top:2rem}",
        ".card{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:1rem;margin:0.6rem 0}",
        ".bar-row{display:flex;align-items:center;margin:2px 0}",
        ".bar-label{width:6.5rem;font-size:0.8rem;color:#475569}",
        ".bar-track{flex:1;background:#eef2f7;border-radius:4px;height:12px;position:relative}",
        ".bar-fill{height:12px;border-radius:4px}",
        ".label-chip{display:inline-block;padding:1px 8px;border-radius:99px;"
        "color:#fff;font-size:0.75rem;font-weight:600}",
        ".stats{background:#0f172a;color:#e2e8f0;border-radius:8px;padding:0.8rem 1.2rem;"
        "font-family:ui-monospace,monospace;font-size:0.85rem}",
        ".text{font-size:0.85rem;color:#334155;margin-top:4px}",
        "</style></head><body>",
        "<h1>Reranker Arena - retrieval order vs Jev-Reranker</h1>",
    ]
    for r in sample_results:
        parts.append(f"<h2>Query: {r['query']}</h2>")
        parts.append(
            f"<div class='stats'>judge: {r['judge']} | judge calls: {r['judge_calls']} | "
            f"rerank latency: {r['latency_ms']} ms | selection latency: {r['selection_latency_ms']} ms | "
            f"tokens: {r['context']['tokens_selected']} selected / {r['context']['tokens_before']} available "
            f"(budget {r['context']['budget_tokens']}) | input tokens: {r['usage'].get('input_tokens', 0)} | "
            f"output tokens: {r['usage'].get('output_tokens', 0)}</div>"
        )
        by_id = {x["id"]: x for x in r["reranked"]}
        for cid in r["retrieval_order"]:
            item = by_id.get(cid)
            if item is None:
                parts.append(f"<div class='card'><span class='label-chip' style='background:#6b7280'>EXCLUDED</span> "
                             f"<span class='text'>{cid} (outside top-k)</span></div>")
                continue
            chip = f"<span class='label-chip' style='background:{LABEL_COLORS[item['label']]}'>{item['label']}</span>"
            bars = "".join(
                f"<div class='bar-row'><div class='bar-label'>{name}</div>"
                f"<div class='bar-track'><div class='bar-fill' style='width:{val * 100:.0f}%;"
                f"background:{HEAD_COLORS[name]}'></div></div><div style='width:2.5rem;text-align:right;"
                f"font-size:0.75rem'>{val:.2f}</div></div>"
                for name, val in item["heads"].items()
            )
            parts.append(
                f"<div class='card'><b>{cid}</b> {chip} value={item['value']}<div class='bars'>{bars}</div>"
                f"<div class='text'>{item['text']}</div></div>"
            )
        sel_ids = r["context"]["selected_ids"]
        parts.append(f"<p>Context selection ({r['context']['tokens_selected']} tok): {', '.join(sel_ids)}</p>")
    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", action="store_true")
    ap.add_argument("--html", type=Path, default=None, help="write a static HTML report and exit")
    ap.add_argument("--sample", type=int, default=None)
    args = ap.parse_args()

    if args.html is not None:
        results = [run_demo(s["query"], s["candidates"]) for s in SAMPLES]
        args.html.write_text(render_html(results), encoding="utf-8")
        print(f"wrote {args.html}")
        return
    if args.cli:
        idx = args.sample if args.sample is not None else 0
        s = SAMPLES[idx % len(SAMPLES)]
        print(json.dumps(run_demo(s["query"], s["candidates"]), indent=2))
        return
    try:
        import streamlit as st

        st.title("Reranker Arena - retrieval order vs Jev-Reranker")
        idx = st.selectbox("sample", range(len(SAMPLES)), format_func=lambda i: SAMPLES[i]["query"])
        s = SAMPLES[int(idx)]
        query = st.text_input("query", s["query"])
        budget = st.slider("context budget (tokens)", 40, 400, 120)
        texts = st.text_area("candidates (one per line)", "\n".join(s["candidates"])).splitlines()
        if st.button("Rerank"):
            out = run_demo(query, [t for t in texts if t.strip()], budget_tokens=budget)
            for x in out["reranked"]:
                st.markdown(f"**{x['id']}** `{x['label']}` value={x['value']} - {x['text']}")
                for name, val in x["heads"].items():
                    st.progress(val, text=f"{name} {val:.2f}")
            st.json({k: v for k, v in out.items() if k != "reranked"})
    except ImportError:
        s = SAMPLES[0]
        print(json.dumps(run_demo(s["query"], s["candidates"]), indent=2))


if __name__ == "__main__":
    main()
