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
    """Open-weights cross-encoder, GPU if available, raw logits as scores."""

    name = "cross-encoder"

    def __init__(self, model_name: str, batch_size: int = 32, max_length: int = 512,
                 trust_remote_code: bool = False, torch_dtype: str | None = None) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        dtype = getattr(torch, torch_dtype) if torch_dtype else None
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, torch_dtype=dtype
        ).to(self.device)
        if getattr(self.model.config, "pad_token_id", None) is None:
            # Qwen3-derived classifiers raise on batch>1 without a pad token id.
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None                 else self.tokenizer.eos_token_id
            self.model.config.pad_token_id = pad_id
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
                ).to(self.device)
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


class ZerankAdapter(BaseAdapter):
    """zeroentropy zerank-1 via its own CrossEncoder.predict path.

    The checkpoint stores no classification head — their remote code reads
    yes/no token logits from the LM head (scaled /5). Feeding pairs through
    plain AutoModelForSequenceClassification scores with a RANDOMLY
    INITIALIZED head (verified: fresh-head warning + below-floor NDCG), so
    this adapter must go through their predict(). Their 15k-token batch
    budget OOMs a 46 GB card at fp32, so we monkey-patch the module constant
    down and load bf16.
    """

    def __init__(self, model_name: str = "zeroentropy/zerank-1-reranker", token_budget: int = 3000) -> None:
        import sys
        import torch
        from sentence_transformers import CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(
            model_name,
            trust_remote_code=True,
            device=device,
            automodel_args={"torch_dtype": torch.bfloat16},
        )
        for mod in list(sys.modules.values()):
            if hasattr(mod, "PER_DEVICE_BATCH_SIZE_TOKENS"):
                mod.PER_DEVICE_BATCH_SIZE_TOKENS = token_budget
        self.model_name = model_name
        self.name = "zerank-1"
        self.pairs = 0

    def rerank(self, query: str, doc_ids: list[str], texts: list[str]) -> tuple[Scores, CallMeta]:
        import time as _time

        t0 = _time.perf_counter()
        raw = self.model.predict([(query, t) for t in texts])
        self.pairs += len(texts)
        scores = Scores({d: float(v) for d, v in zip(doc_ids, raw, strict=True)})
        return scores, CallMeta(
            latency_ms=round((_time.perf_counter() - t0) * 1000, 2),
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
            import os

            out.append(CrossEncoderAdapter(os.environ.get("CE_MODEL", ce_model)))
        elif s == "bge-reranker-v2-m3-gguf":
            out.append(LlamaCppRerankAdapter("bge-reranker-v2-m3"))
        elif s == "qwen3-reranker-0.6b":
            out.append(Qwen3RerankAdapter("Qwen/Qwen3-Reranker-0.6B"))
        elif s == "zerank-1":
            out.append(ZerankAdapter("zeroentropy/zerank-1-reranker"))
        else:
            raise ValueError(f"unknown system {s!r}")
    return out


class Qwen3RerankAdapter(BaseAdapter):
    """Qwen3-Reranker: causal LM scored via P(yes)/(P(yes)+P(no)) at the last token.

    Reference format from the Qwen3-Reranker model card; left padding so the
    final position is the real last token. max_length 512 to match the other
    lanes' truncation.
    """

    INSTRUCTION = "Given a query, retrieve relevant documents that answer the query"
    PREFIX = (
        '<|im_start|>system\nJudge whether the Document meets the requirements based on '
        'the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
        '<|im_end|>\n<|im_start|>user\n'
    )
    SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-0.6B", batch_size: int = 16, max_length: int = 512) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        self.model.eval()
        self.model_name = model_name
        self.name = model_name.split("/")[-1]
        self.batch_size = batch_size
        self.max_length = max_length
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix_tokens = self.tokenizer.encode(self.PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.SUFFIX, add_special_tokens=False)
        self.pairs = 0

    def _format(self, query: str, doc: str) -> str:
        return f"<Instruct>: {self.INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}"

    def rerank(self, query: str, doc_ids: list[str], texts: list[str]) -> tuple[Scores, CallMeta]:
        t0 = time.perf_counter()
        scores: dict[str, float] = {}
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                chunk = [self._format(query, t) for t in texts[i : i + self.batch_size]]
                inputs = self.tokenizer(
                    chunk,
                    padding=False,
                    truncation="longest_first",
                    return_attention_mask=False,
                    max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
                )
                inputs["input_ids"] = [self.prefix_tokens + ids + self.suffix_tokens for ids in inputs["input_ids"]]
                batch = self.tokenizer.pad(inputs, padding=True, return_tensors="pt").to(self.device)
                logits = self.model(**batch).logits[:, -1, :]
                stacked = self.torch.stack([logits[:, self.token_false_id], logits[:, self.token_true_id]], dim=1)
                probs = self.torch.nn.functional.log_softmax(stacked, dim=1)[:, 1].exp().tolist()
                for d, s in zip(doc_ids[i : i + self.batch_size], probs, strict=True):
                    scores[d] = float(s)
        self.pairs += len(texts)
        return Scores(scores), CallMeta(
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            usage={"n_pairs": len(texts)},
        )
