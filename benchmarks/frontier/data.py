"""BEIR data loading + BM25 candidate generation.

Loads the official BEIR repo files from huggingface.co/datasets/BeIR/<name>
(parquet shards for corpus/queries, TSV qrels from the BeIR/<name>-qrels
repo). pyarrow ships musllinux wheels, so this works on the Alpine/musl
benchmark host without a compiler.
Candidates come from bm25s with a lean tokenizer (lowercase, no stemmer) -
deliberately simple because the floor is shared identically by every system;
published BM25 numbers used heavier pipelines and will differ (noted in the
README).
"""

from __future__ import annotations

import csv
import io
import json
import random
from pathlib import Path

import requests

DATASETS = ("scifact", "nfcorpus", "fiqa")

_HF = "https://huggingface.co/datasets/BeIR/{name}/resolve/main/{file}"
_TREE = "https://huggingface.co/api/datasets/BeIR/{name}/tree/main/{dir}"
_QREL_SPLITS = ("test.tsv", "dev.tsv", "train.tsv")


def _get(url: str) -> requests.Response:
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp


def _parquet_shards(name: str, subdir: str) -> list[str]:
    """Paths of all .parquet shards under BeIR/<name>/<subdir>/."""
    tree = _get(_TREE.format(name=name, dir=subdir)).json()
    return [str(f["path"]) for f in tree if str(f.get("path", "")).endswith(".parquet")]


def load_beir(name: str, cache_dir: Path) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]]]:
    """Return (corpus {docid: text}, queries {qid: text}, qrels {qid: {docid: grade}})."""
    cache_file = cache_dir / f"{name}_loaded.json"
    if cache_file.exists():
        blob = json.loads(cache_file.read_text(encoding="utf-8"))
        return blob["corpus"], blob["queries"], blob["qrels"]

    import pyarrow.parquet as pq

    def _tables(name: str, subdir: str):
        for shard in _parquet_shards(name, subdir):
            yield pq.read_table(io.BytesIO(_get(_HF.format(name=name, file=shard)).content))

    corpus: dict[str, str] = {}
    for table in _tables(name, "corpus"):
        cols = table.column_names
        ids = table.column("_id").to_pylist()
        titles = table.column("title").to_pylist() if "title" in cols else [""] * table.num_rows
        texts = table.column("text").to_pylist() if "text" in cols else [""] * table.num_rows
        for did, title, text in zip(ids, titles, texts, strict=True):
            combined = (title + " " + text).strip() if title else text
            corpus[str(did)] = combined

    queries: dict[str, str] = {}
    for table in _tables(name, "queries"):
        ids = table.column("_id").to_pylist()
        texts = table.column("text").to_pylist() if "text" in table.column_names else [""] * table.num_rows
        for qid, text in zip(ids, texts, strict=True):
            queries[str(qid)] = text

    qrels_text = None
    for split in _QREL_SPLITS:
        resp = _get(_HF.format(name=f"{name}-qrels", file=split))
        if resp.ok:
            qrels_text = resp.text
            break
    if qrels_text is None:
        raise RuntimeError(f"no qrels found for {name}")

    qrels: dict[str, dict[str, int]] = {}
    reader = csv.DictReader(io.StringIO(qrels_text), delimiter="\t")
    if reader.fieldnames and "query-id" not in reader.fieldnames:
        reader = csv.DictReader(io.StringIO(qrels_text), delimiter="\t",
                                fieldnames=["query-id", "corpus-id", "score"])
    for row in reader:
        qid, did, score = row.get("query-id"), row.get("corpus-id"), row.get("score")
        if qid is None or did is None or score is None:
            continue
        qrels.setdefault(qid, {})[did] = int(score)

    usable = {qid: rels for qid, rels in qrels.items() if qid in queries and rels}
    qrels = usable
    queries = {qid: queries[qid] for qid in usable}

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps({"corpus": corpus, "queries": queries, "qrels": qrels}, ensure_ascii=False),
        encoding="utf-8",
    )
    return corpus, queries, qrels


def sample_queries(qrels: dict[str, dict[str, int]], n: int, seed: int = 42) -> list[str]:
    """Seeded subsample of qids (sorted first for determinism). n<=0 -> all."""
    qids = sorted(qrels)
    if 0 < n < len(qids):
        rng = random.Random(seed)
        qids = sorted(rng.sample(qids, n))
    return qids


def bm25_candidates(
    corpus: dict[str, str], queries: dict[str, str], k: int, cache_file: Path
) -> dict[str, list[str]]:
    """BM25 top-k docids per query (cached; deterministic given corpus+queries).

    A partial cache (e.g. written by a smoke run) is EXTENDED, not trusted:
    any requested qid missing from the cache is retrieved and merged.
    """
    cached: dict[str, list[str]] = {}
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
    missing_qids = [q for q in sorted(queries) if q not in cached]
    if not missing_qids:
        return cached
    import bm25s

    doc_ids = sorted(corpus)
    tokenizer = bm25s.tokenization.Tokenizer(lower=True, stopwords="english", stemmer=None)
    corpus_tokens = tokenizer.tokenize([corpus[d] for d in doc_ids], show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    q_tokens = tokenizer.tokenize([queries[q] for q in missing_qids], show_progress=False)
    docs_idx, scores = retriever.retrieve(q_tokens, k=min(k, len(doc_ids)), show_progress=False)
    for qid, row in zip(missing_qids, docs_idx, strict=True):
        cached[qid] = [doc_ids[int(i)] for i in row]
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cached), encoding="utf-8")
    return cached
