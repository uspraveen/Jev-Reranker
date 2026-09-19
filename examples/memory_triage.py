"""Agent-memory triage: label STALE/CONFLICT/UNCERTAIN memories in one Jev call."""

from jev_reranker import Candidate, JevReranker

memories = [
    Candidate(id="m1", text="Deploy freeze every Friday 18:00 UTC.", retrieval_rank=0),
    Candidate(id="m2", text="OUTDATED deprecated old version: deploy freeze starts Thursday.", retrieval_rank=1),
    Candidate(id="m3", text="However, that is wrong: there is no deploy freeze at all.", retrieval_rank=2),
    Candidate(id="m4", text="Rollback runbook lives at /docs/rollback.", retrieval_rank=3),
]

rr = JevReranker()
result = rr.rerank("when is the deploy freeze?", memories, mode="memory")

usable = [it for it in result.items if it.label.value in ("USE", "KEEP")]
flagged = [it for it in result.items if it.label.value in ("STALE", "CONFLICT", "UNCERTAIN")]

print("USABLE:")
for it in usable:
    print(f"  [{it.label.value}] {it.candidate.text}")
print("FLAGGED:")
for it in flagged:
    print(f"  [{it.label.value}] {it.candidate.id}: {it.candidate.text}")
