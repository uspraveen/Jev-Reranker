"""Reranker adapters: one interface, six systems.

Every adapter gets the SAME (query, candidate texts in candidate order) and
returns ``Scores`` = {docid: float score} plus honest metadata (latency,
token usage, errors). NDCG is invariant to monotonic score transforms, so
adapters may return raw logits/scores; only order matters.

Errors are NEVER silently dropped: a failed query is recorded as an error in
the JSONL and reported in the results. Retries: exponential backoff honoring
Retry-After on 429/5xx for the hosted APIs; Jev uses the SDK RetryPolicy.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


class AdapterError(RuntimeError):
    pass


class Scores(dict[str, float]):
    pass


class CallMeta(dict[str, Any]):
    pass


def _backoff_call(fn: Callable[[], Any], attempts: int = 5, base: float = 2.0, pace: Callable[[], None] | None = None) -> Any:
    """Call fn() retrying on 429/5xx with exponential backoff (honors Retry-After).

    ``pace`` runs before EVERY attempt, including the first: rate limits count
    retried requests, so pacing only between queries still trips them.
    """
    import requests

    last: Exception | None = None
    for i in range(attempts):
        if pace is not None:
            pace()
        try:
            resp = fn()
            if resp.status_code in (429, 500, 502, 503, 504, 529):
                wait = float(resp.headers.get("Retry-After", base**i))
                time.sleep(min(wait, 60.0))
                last = AdapterError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                continue
            if resp.status_code != 200:
                raise AdapterError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            return resp
        except AdapterError:
            raise
        except requests.RequestException as exc:  # network blips
            last = exc
            time.sleep(min(base**i, 60.0))
    raise AdapterError(f"exhausted retries: {last}")


class BaseAdapter:
    name: str = "base"

    def rerank(self, query: str, doc_ids: list[str], texts: list[str]) -> tuple[Scores, CallMeta]:
        raise NotImplementedError

    def close(self) -> None:  # noqa: B027
        pass


class BM25Adapter(BaseAdapter):
    """Floor: keep BM25 retrieval order (score = -rank, ties impossible)."""

    name = "bm25"

    def rerank(self, query: str, doc_ids: list[str], texts: list[str]) -> tuple[Scores, CallMeta]:
        t0 = time.perf_counter()
        scores = Scores({d: -float(i) for i, d in enumerate(doc_ids)})
        return scores, CallMeta(latency_ms=round((time.perf_counter() - t0) * 1000, 2), usage={})


class JevAdapter(BaseAdapter):
    """Jev-Reranker relevance mode (product path, generic_retrieval rubric)."""

    name = "jev-latest"

    def __init__(self, api_key: str) -> None:
        from jev_reranker import JevReranker
        from jev_reranker.judges import LiveJevJudge

        self.rr = JevReranker(judge=LiveJevJudge(api_key=api_key), rubric="generic_retrieval")
        self.calls = 0

    def rerank(self, query: str, doc_ids: list[str], texts: list[str]) -> tuple[Scores, CallMeta]:
        from jev_reranker import Candidate

        cands = [Candidate(id=d, text=t, retrieval_rank=i) for i, (d, t) in enumerate(zip(doc_ids, texts, strict=True))]
        res = self.rr.rerank(query, cands, mode="relevance")
        self.calls += 1
        if res.fallback_used:
            raise AdapterError("jev fallback_used (judge unreachable)")
        scores = Scores({it.candidate.id: it.relevance_score for it in res.items})
        return scores, CallMeta(
            latency_ms=res.latency_ms,
            usage=res.usage,
            n_questions=len(doc_ids),
            fallback_used=False,
            dropped_by_token_budget=res.trace.get("dropped_by_token_budget"),
        )

    def close(self) -> None:
        pass  # sync judge; nothing persistent


class VoyageAdapter(BaseAdapter):
    def __init__(self, api_key: str, model: str = "rerank-2.5", min_interval_s: float = 21.0) -> None:
        import requests

        self.http = requests.Session()
        self.http.headers.update({"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        self.model = model
        self.name = model
        self.min_interval_s = min_interval_s  # trial keys: 3 RPM
        self._last_call = 0.0
        self.total_tokens = 0

    def _pace(self) -> None:
        wait = self.min_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def rerank(self, query: str, doc_ids: list[str], texts: list[str]) -> tuple[Scores, CallMeta]:
        t0 = time.perf_counter()
        resp = _backoff_call(
            lambda: self.http.post(
                "https://api.voyageai.com/v1/rerank",
                json={"model": self.model, "query": query, "documents": texts, "top_k": len(texts)},
                timeout=60,
            ),
            pace=self._pace,
        )
        body = resp.json()
        self.total_tokens += int(body.get("usage", {}).get("total_tokens", 0))
        scores = Scores({doc_ids[int(r["index"])]: float(r["relevance_score"]) for r in body["data"]})
        return scores, CallMeta(
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            usage=body.get("usage", {}),
        )


class CohereAdapter(BaseAdapter):
    def __init__(self, api_key: str, model: str, min_interval_s: float = 7.0) -> None:
        import requests

        self.http = requests.Session()
        self.http.headers.update({"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        self.model = model
        self.name = model
        self.min_interval_s = min_interval_s  # trial-key pacing
        self._last_call = 0.0
        self.search_units = 0

    def _pace(self) -> None:
        wait = self.min_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def rerank(self, query: str, doc_ids: list[str], texts: list[str]) -> tuple[Scores, CallMeta]:
        t0 = time.perf_counter()
        resp = _backoff_call(
            lambda: self.http.post(
                "https://api.cohere.com/v2/rerank",
                json={"model": self.model, "query": query, "documents": texts, "top_n": len(texts)},
                timeout=60,
            ),
            pace=self._pace,
        )
        body = resp.json()
        self.search_units += int(body.get("meta", {}).get("billed_units", {}).get("search_units", 0))
        scores = Scores({doc_ids[int(r["index"])]: float(r["relevance_score"]) for r in body["results"]})
        return scores, CallMeta(
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            usage=body.get("meta", {}).get("billed_units", {}),
        )


class CrossEncoderAdapter(BaseAdapter):
    """Open-weights cross-encoder, CPU, raw logits as scores (plain transformers)."""

    name = "cross-encoder"

    def __init__(self, model_name: str, batch_size: int = 32, max_length: int = 512) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.model_name = model_name
        self.name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.pairs = 0

    def rerank(self, query: str, doc_ids: list[str], texts: list[str]) -> tuple[Scores, CallMeta]:
        t0 = time.perf_counter()
        all_scores: list[float] = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                enc = self.tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                logits = self.model(**enc).logits[:, 0]
                all_scores.extend(float(s) for s in logits)
        self.pairs += len(texts)
        scores = Scores({d: s for d, s in zip(doc_ids, all_scores, strict=True)})
        return scores, CallMeta(
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            usage={"n_pairs": len(texts)},
        )


class LlamaCppRerankAdapter(BaseAdapter):
    """Open-weights reranker served by llama.cpp's /v1/rerank (TEI-compatible).

    Used for the self-hosted lane on musl hosts where torch wheels are
    unavailable; the GGUF quantization is stated in the results.
    """

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:8899") -> None:
        import requests

        self.http = requests.Session()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = model

    def rerank(self, query: str, doc_ids: list[str], texts: list[str]) -> tuple[Scores, CallMeta]:
        t0 = time.perf_counter()
        resp = _backoff_call(
            lambda: self.http.post(
                f"{self.base_url}/v1/rerank",
                json={"model": self.model, "query": query, "documents": texts, "top_n": len(texts)},
                timeout=300,
            )
        )
        body = resp.json()
        results = body.get("results") or body.get("data") or []
        scores = Scores({doc_ids[int(r["index"])]: float(r["relevance_score"]) for r in results})
        return scores, CallMeta(
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            usage={"n_pairs": len(texts)},
        )


def build_adapters(
    systems: list[str],
    keys: dict[str, str],
    cohere_model: str = "rerank-v4.0-pro",
    cohere_min_interval_s: float = 7.0,
    ce_model: str = "BAAI/bge-reranker-base",
) -> list[BaseAdapter]:
    out: list[BaseAdapter] = []
    for s in systems:
        if s == "bm25":
            out.append(BM25Adapter())
        elif s == "jev-latest":
            out.append(JevAdapter(keys["TYPESAFE_API_KEY"]))
        elif s == "voyage-2.5":
            out.append(VoyageAdapter(keys["VOYAGE_API_KEY"], "rerank-2.5", min_interval_s=21.0))
        elif s == "voyage-2.5-lite":
            out.append(VoyageAdapter(keys["VOYAGE_API_KEY"], "rerank-2.5-lite", min_interval_s=21.0))
        elif s == "cohere":
            out.append(CohereAdapter(keys["COHERE_API_KEY"], cohere_model, cohere_min_interval_s))
        elif s == "bge-reranker-base":
            out.append(CrossEncoderAdapter(ce_model))
        elif s == "bge-reranker-v2-m3-gguf":
            out.append(LlamaCppRerankAdapter("bge-reranker-v2-m3"))
        else:
            raise ValueError(f"unknown system {s!r}")
    return out
