"""Generic retrieval: swap the rubric to your domain, read relevance_score.

The policy math is domain-agnostic; the questions encode domain meaning.
Pick a built-in rubric (agent_memory | generic_retrieval | code_search) or
pass a custom Rubric. Runs fully offline (no TYPESAFE_API_KEY -> OfflineJudge).
"""

from jev_reranker import BUILTIN_RUBRICS, CohereCompatReranker, JevReranker, NoulHeadRubric, Rubric, ScoreHeadRubric

DOCS = [
    "kubectl rollout status deploy/payments --timeout=120s",
    "Payments refunds settle within 5 business days.",
    "Our office lunch menu rotates daily.",
]

# 1. Built-in rubric preset - same pipeline, domain wording.
rr = JevReranker(rubric="code_search")
res = rr.rerank("how do I check a k8s rollout?", DOCS, mode="relevance")
for it in res.items:
    print("preset:", it.rank, round(it.relevance_score, 3), it.candidate.text[:50])

# 2. Fully custom rubric - one dict per head.
legal = Rubric(
    name="legal",
    rel=ScoreHeadRubric(
        instructions="Is candidate [cid] controlling authority for the query?",
        criteria=["off point", "background", "persuasive", "controlling"],
    ),
    util=ScoreHeadRubric(
        instructions="How actionable is candidate [cid] for the query?",
        criteria=["no holdings", "background only", "on-point dicta", "directly on point"],
    ),
    sup=NoulHeadRubric(
        instructions="Was candidate [cid] overruled or superseded by newer authority?",
        criteria={"true": "Newer authority controls", "false": "Still good law"},
    ),
    con=NoulHeadRubric(
        instructions="Does candidate [cid] conflict with the query or other candidates?",
        criteria={"true": "Contradicted elsewhere", "false": "Consistent"},
    ),
)
rr_legal = JevReranker(rubric=legal)
print("custom rubric id:", rr_legal.client.rubric.id)

# 3. Hosted-API shape: positional index + relevance_score in [0, 1].
api = CohereCompatReranker(reranker=JevReranker(rubric="generic_retrieval"))
body = api.rerank("how do refunds work?", DOCS, top_n=2)
print("cohere-shape:", body["results"])

# Serve the same contract over HTTP instead:
#   pip install "jev-reranker[server]"
#   uvicorn jev_reranker.server:app --port 8494
print("built-ins:", sorted(BUILTIN_RUBRICS))
