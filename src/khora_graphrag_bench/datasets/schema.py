"""Pydantic models for the GraphRAG-Bench dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class DatasetDocument(BaseModel):
    """A document in the dataset (becomes input to ``build_graph``)."""

    doc_id: str
    content: str
    title: str = ""
    metadata: dict = Field(default_factory=dict)


class GraphRAGQuestion(BaseModel):
    """A GraphRAG-Bench question (one of five types across four difficulties)."""

    question_id: str
    question: str
    # Question type — drives which scoring function applies.
    #   MC: multiple choice (exact letter match)
    #   MS: multi-select (graduated Jaccard)
    #   TF: true/false (normalised exact match)
    #   FB: fill-in-the-blank (LLM-judged factuality + semantic sim)
    #   OE: open-ended (LLM-judged factuality + semantic sim)
    question_type: Literal["MC", "MS", "TF", "FB", "OE"]
    # Difficulty level — drives which auxiliary metrics are applicable.
    difficulty: Literal[
        "fact_retrieval",
        "complex_reasoning",
        "contextual_summarization",
        "creative_generation",
    ]
    discipline: str = ""
    source_textbook: str = ""
    gold_answer: str
    options: list[str] | None = None
    # Gold supporting facts. Used for r_score and evidence_recall.
    evidence: list[str] = Field(default_factory=list)
    # Document IDs (from the dataset's documents list) whose content was used
    # as the evidence for this question. Used by the sampler to retain only
    # questions whose specific evidence chunks survived doc sampling
    # (`set(relevant_doc_ids).issubset(retained_ids)`).
    relevant_doc_ids: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class GraphRAGDataset(BaseModel):
    """A GraphRAG-Bench dataset: documents + multi-type questions + optional ontology."""

    name: str
    version: str = "1.0"
    description: str = ""
    documents: list[DatasetDocument] = Field(default_factory=list)
    questions: list[GraphRAGQuestion] = Field(default_factory=list)
    # Optional entity/relationship type allowlists for Khora's extraction
    # pipeline. The dataset's published ontology is loaded into these so the
    # graph build uses the same type vocabulary the paper expects.
    entity_types: list[str] = Field(default_factory=list)
    relationship_types: list[str] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> GraphRAGDataset:
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    def compute_hash(self) -> str:
        """Short SHA-256 fingerprint of dataset contents (for reproducibility)."""
        content = json.dumps(
            {
                "documents": [d.model_dump() for d in self.documents],
                "questions": [q.model_dump() for q in self.questions],
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]
