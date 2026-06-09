"""Convert the raw GraphRAG-Bench JSON into our typed dataset model."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from khora_graphrag_bench.datasets.schema import DatasetDocument, GraphRAGDataset, GraphRAGQuestion

logger = logging.getLogger(__name__)


def _hash_content(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# Map of GraphRAG-Bench `question_type` (human-readable) to our difficulty enum.
_DIFFICULTY = {
    "fact retrieval": "fact_retrieval",
    "complex reasoning": "complex_reasoning",
    "contextual summarize": "contextual_summarization",
    "contextual summarization": "contextual_summarization",
    "creative generation": "creative_generation",
}

# Map to the scoring-code enum (drives which scorer is applied).
_QUESTION_TYPE_CODE = {
    "fact retrieval": "FB",
    "complex reasoning": "OE",
    "contextual summarize": "OE",
    "contextual summarization": "OE",
    "creative generation": "OE",
}


def graphrag_bench_to_dataset(
    raw: list[dict[str, Any]],
    corpus: list[dict[str, Any]] | None = None,
    name: str = "graphrag_bench_novel",
    entity_types: list[str] | None = None,
    relationship_types: list[str] | None = None,
) -> GraphRAGDataset:
    """Convert the published GraphRAG-Bench JSON into a typed ``GraphRAGDataset``.

    Question records (``novel_questions.json``) carry:
      - ``id`` (e.g. "Novel-73586ddc")
      - ``source`` (corpus identifier, e.g. "Novel-44557")
      - ``question`` / ``answer`` (gold)
      - ``question_type`` ("Fact Retrieval" | "Complex Reasoning" | …)
      - ``evidence`` (gold supporting sentences) / ``evidence_triple``

    ``corpus`` is the source-document set (``novel.json``): one record per novel,
    ``{"corpus_name": <source>, "context": <full novel text>}``. The knowledge
    graph is built from these full texts — matching GraphRAG-Bench's own
    methodology, where retrieval runs over the novels, not over per-question
    evidence. Each question's ``relevant_doc_ids`` points at the novel it was
    authored from (``question.source`` == ``corpus_name``).

    If ``corpus`` is omitted, or a question's ``source`` has no matching corpus
    entry, that question's document is reconstructed from its ``evidence`` as a
    logged last resort. That makes retrieval trivial and is NOT comparable to the
    published benchmark — it exists only so the harness degrades gracefully
    instead of dropping questions.
    """
    documents: list[DatasetDocument] = []
    questions: list[GraphRAGQuestion] = []
    corpus_doc_ids: set[str] = set()
    fallback_doc_ids: set[str] = set()

    # Build the document corpus from the full source novels (one doc per novel).
    for record in corpus or []:
        corpus_name = record.get("corpus_name") or record.get("source") or "unknown"
        content = record.get("context", "")
        if not content or corpus_name in corpus_doc_ids:
            continue
        corpus_doc_ids.add(corpus_name)
        documents.append(
            DatasetDocument(
                doc_id=corpus_name,
                content=content,
                title=corpus_name,
                metadata={"source_textbook": corpus_name},
            )
        )

    for record in raw:
        source = record.get("source", "unknown")
        evidence_list = record.get("evidence", [])

        if source in corpus_doc_ids:
            # The question is answered from its source novel: retrieval must find
            # the supporting passages within that full-text graph.
            relevant_doc_ids = [source]
        else:
            # Last resort: no matching novel in the corpus. Reconstruct a document
            # from this question's evidence so it isn't silently dropped. This is
            # the answer key — retrieval is trivial and not benchmark-comparable.
            context = record.get("context", "") or " ".join(evidence_list)
            doc_id = f"{source}_{_hash_content(context)[:12]}" if context else source
            if context and doc_id not in fallback_doc_ids:
                fallback_doc_ids.add(doc_id)
                documents.append(
                    DatasetDocument(
                        doc_id=doc_id,
                        content=context,
                        title=source,
                        metadata={"source_textbook": source, "reconstructed_from_evidence": True},
                    )
                )
            relevant_doc_ids = [doc_id] if context else []

        question_id = record.get("id", f"q_{len(questions)}")
        question_text = record.get("question", "")
        if not question_text:
            continue

        raw_qtype = str(record.get("question_type", "")).lower().strip()
        difficulty = _DIFFICULTY.get(raw_qtype, "fact_retrieval")
        qtype_code = _QUESTION_TYPE_CODE.get(raw_qtype, "OE")
        gold_answer = record.get("answer", record.get("gold_answer", ""))

        questions.append(
            GraphRAGQuestion(
                question_id=question_id,
                question=question_text,
                question_type=qtype_code,  # type: ignore[arg-type]
                difficulty=difficulty,  # type: ignore[arg-type]
                discipline=source,
                source_textbook=source,
                gold_answer=gold_answer,
                evidence=list(evidence_list),
                relevant_doc_ids=relevant_doc_ids,
                metadata={
                    "evidence_relations": record.get("evidence_triple", []),
                },
            )
        )

    if not corpus_doc_ids:
        logger.warning(
            "graphrag_bench: no source corpus provided — ALL documents were reconstructed "
            "from per-question evidence. Retrieval-side metrics are NOT comparable to the "
            "published GraphRAG-Bench numbers."
        )
    elif fallback_doc_ids:
        logger.warning(
            "graphrag_bench: %d question source(s) had no matching corpus novel; their "
            "documents were reconstructed from evidence (trivial retrieval, not comparable).",
            len(fallback_doc_ids),
        )

    return GraphRAGDataset(
        name=name,
        version="1.0",
        description=(
            "GraphRAG-Bench (ICLR'26). College-level questions over full source novels, "
            "with gold evidence triples for graph-RAG evaluation. "
            "https://github.com/GraphRAG-Bench/GraphRAG-Benchmark"
        ),
        documents=documents,
        questions=questions,
        entity_types=entity_types or [],
        relationship_types=relationship_types or [],
    )
