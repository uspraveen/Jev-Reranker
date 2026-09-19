"""30-second quickstart: rerank with Jev-Reranker.

Needs TYPESAFE_API_KEY for live Jev; otherwise runs on the offline judge
(clearly labeled) so the mechanics are identical.
"""

from jev_reranker import JevReranker

rr = JevReranker()  # LiveJevJudge iff TYPESAFE_API_KEY is set, else OfflineJudge

result = rr.rerank_texts(
    "how long do refunds take to settle?",
    [
        "Our patio furniture sale ends Sunday.",
        "Refunds settle within 5 business days.",
        "OUTDATED deprecated old version: refunds settle in 30 days.",
    ],
    mode="memory",
    top_k=3,
)

for item in result.items:
    print(f"{item.rank + 1}. [{item.label.value:9s}] value={item.value:+.3f}  {item.candidate.text}")

print(f"\nrequest_id={result.request_id} cached={result.cached} "
      f"fallback={result.fallback_used} judge={result.trace.get('judge')}")
