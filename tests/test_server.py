"""Server tests: Cohere-shaped /rerank, token-budget /select, /health over the offline judge."""

from fastapi.testclient import TestClient

from jev_reranker.integrations.cohere_compat import CohereCompatReranker
from jev_reranker.judges import OfflineJudge
from jev_reranker.reranker import JevReranker
from jev_reranker.server import create_app


def _client() -> TestClient:
    rr = CohereCompatReranker(reranker=JevReranker(judge=OfflineJudge()))
    return TestClient(create_app(rr))


def test_health_reports_judge() -> None:
    with _client() as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["judge"] == "OfflineJudge"


def test_rerank_contract_shape() -> None:
    docs = ["lunch menu pasta", "payments deploy runbook kubectl", "deploy payments to prod"]
    with _client() as client:
        body = client.post(
            "/rerank", json={"query": "how do I deploy payments?", "documents": docs, "top_n": 2}
        ).json()
    assert len(body["results"]) == 2
    scores = [r["relevance_score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)
    for r in body["results"]:
        assert set(r) == {"index", "relevance_score", "document"}
        assert isinstance(r["index"], int)
        assert 0 <= r["index"] < len(docs)
        assert r["document"]["text"] == docs[r["index"]]
        assert 0.0 <= r["relevance_score"] <= 1.0
    assert body["meta"]["jev"]["mode"] == "relevance"
    assert len(body["meta"]["jev"]["items"]) == 2


def test_rerank_accepts_document_dicts() -> None:
    docs = [{"text": "deploy payments via kubectl"}, {"text": "lunch menu"}]
    with _client() as client:
        body = client.post("/rerank", json={"query": "deploy payments?", "documents": docs}).json()
    assert len(body["results"]) == 2
    assert body["results"][0]["document"]["text"] == "deploy payments via kubectl"


def test_select_contract_shape() -> None:
    docs = ["Refunds settle within 5 business days.", "The sky is blue.", "Refunds settle in five business days."]
    with _client() as client:
        body = client.post(
            "/select",
            json={"query": "how long do refunds take?", "documents": docs, "budget_tokens": 200},
        ).json()
    assert body["total_tokens"] <= body["budget_tokens"]
    for entry in body["selected"]:
        assert {"index", "text", "label", "value"} <= set(entry)
