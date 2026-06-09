"""Download + cache the GraphRAG-Bench dataset from HuggingFace."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from khora_graphrag_bench.datasets.converters import graphrag_bench_to_dataset
from khora_graphrag_bench.datasets.schema import GraphRAGDataset

logger = logging.getLogger(__name__)


# Published GraphRAG-Bench novel-domain questions.
GRAPHRAG_BENCH_NOVEL_URL = (
    "https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench/resolve/main/Datasets/Questions/novel_questions.json"
)

# Published GraphRAG-Bench novel-domain source corpus: the full novels the
# questions were authored against. The knowledge graph is built from these
# documents (matching GraphRAG-Bench's methodology), not from per-question
# evidence.
GRAPHRAG_BENCH_NOVEL_CORPUS_URL = (
    "https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench/resolve/main/Datasets/Corpus/novel.json"
)

# Entity / relationship type allowlists that GraphRAG-Bench's reference
# implementation uses for the novel-domain corpus. Passing these to Khora's
# extraction pipeline (instead of generic defaults) measurably improves graph
# quality and downstream r_score.
NOVEL_ENTITY_TYPES = ["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "CONCEPT", "OBJECT", "CREATURE"]
NOVEL_RELATIONSHIP_TYPES = [
    "INTERACTS_WITH",
    "LOCATED_IN",
    "CAUSES",
    "SYMBOLIZES",
    "OPPOSES",
    "PART_OF",
    "BELONGS_TO",
    "TRANSFORMS_INTO",
    "RELATES_TO",
]


def _fetch_cached(url: str, path: Path, force_download: bool, label: str) -> Any:
    """Download ``url`` to ``path`` (unless cached) and return the parsed JSON."""
    if force_download or not path.exists():
        logger.info("Downloading GraphRAG-Bench %s to %s", label, path)
        with urlopen(url) as resp:  # noqa: S310 — fixed HF URL
            path.write_bytes(resp.read())
    return json.loads(path.read_text())


def load_graphrag_bench(
    cache_dir: str | Path = ".cache/khora-graphrag-bench/datasets",
    force_download: bool = False,
) -> GraphRAGDataset:
    """Load the GraphRAG-Bench novel dataset (questions + full source corpus).

    Two files are downloaded and cached under ``cache_dir``:
      - ``novel_questions.json`` — the 2010 questions with gold evidence.
      - ``novel_corpus.json`` — the 20 full source novels the questions were
        authored against.

    The knowledge graph is built from the full novels (matching GraphRAG-Bench's
    own methodology — retrieval runs over the novels, not over per-question
    evidence). Subsequent calls read from disk unless ``force_download`` is true.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    raw = _fetch_cached(GRAPHRAG_BENCH_NOVEL_URL, cache / "novel_questions.json", force_download, "novel questions")
    if not isinstance(raw, list):
        raise ValueError(f"GraphRAG-Bench questions JSON must be a list, got {type(raw).__name__}")

    corpus = _fetch_cached(GRAPHRAG_BENCH_NOVEL_CORPUS_URL, cache / "novel_corpus.json", force_download, "novel corpus")
    if not isinstance(corpus, list):
        raise ValueError(f"GraphRAG-Bench corpus JSON must be a list, got {type(corpus).__name__}")

    return graphrag_bench_to_dataset(
        raw,
        corpus=corpus,
        name="graphrag_bench_novel",
        entity_types=NOVEL_ENTITY_TYPES,
        relationship_types=NOVEL_RELATIONSHIP_TYPES,
    )
