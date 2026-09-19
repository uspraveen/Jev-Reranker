"""Token-budget context selection: fit the best context into N tokens."""

from jev_reranker import Candidate, JevReranker

cands = [
    Candidate(id=f"m{i}", text=text, retrieval_rank=i)
    for i, text in enumerate(
        [
            "Refunds settle within 5 business days, confirmed 2026-09-15.",
            "The full refund policy document with twenty paragraphs of legal background. " * 6,
            "Support email is support@example.com.",
            "OUTDATED deprecated old version: refunds took 30 days in 2023.",
            "Chargebacks must be filed within 120 days of the transaction date.",
        ]
    )
]

rr = JevReranker()
sel = rr.select("summarize the refund policy for a customer", cands, budget_tokens=150)
print(f"budget={sel.budget_tokens} used={sel.total_tokens}")
for it in sel.selected:
    print(f"  IN  [{it.label.value}] value={it.value:+.3f} {it.candidate.text[:80]}")
for it in sel.excluded:
    print(f"  OUT [{it.label.value}] {it.candidate.id}")
