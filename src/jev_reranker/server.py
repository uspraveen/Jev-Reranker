"""Optional HTTP surface: a small FastAPI app speaking the hosted-rerank shape.

    pip install "jev-reranker[server]"
    uvicorn jev_reranker.server:app --port 8494

Endpoints:
- ``POST /rerank`` - Cohere-compatible: ``{query, documents, top_n}`` ->
  ``{id, results: [{index, relevance_score, document}], meta}``
- ``POST /select`` - token-budget selection: ``{query, documents, budget_tokens}``
- ``GET  /health`` - judge/model info

The judge is chosen at startup like ``JevReranker()``: live Jev iff
``TYPESAFE_API_KEY`` is set, else the offline judge (``/health`` reports
which one is serving - offline responses are never presented as Jev output).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from jev_reranker import __version__
from jev_reranker.integrations.cohere_compat import CohereCompatReranker, Doc


class RerankRequest(BaseModel):
    query: str
    documents: list[Doc]
    top_n: int | None = None


class SelectRequest(BaseModel):
    query: str
    documents: list[Doc]
    budget_tokens: int = Field(gt=0)


def create_app(reranker: CohereCompatReranker | None = None) -> FastAPI:
    rr = reranker or CohereCompatReranker()
    app = FastAPI(title="Jev-Reranker", version=__version__)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": rr.model, "judge": rr.judge_name}

    @app.post("/rerank")
    async def rerank(req: RerankRequest) -> dict[str, Any]:
        return await rr.arerank(req.query, req.documents, top_n=req.top_n)

    @app.post("/select")
    async def select(req: SelectRequest) -> dict[str, Any]:
        return await rr.aselect(req.query, req.documents, budget_tokens=req.budget_tokens)

    return app


app = create_app()
