"""Unit tests for ``khora_graphrag_bench.datasets.schema``."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from khora_graphrag_bench.datasets.schema import (
    DatasetDocument,
    GraphRAGDataset,
    GraphRAGQuestion,
)


class TestDatasetDocument:
    def test_minimal(self) -> None:
        doc = DatasetDocument(doc_id="d1", content="hello")
        assert doc.doc_id == "d1"
        assert doc.content == "hello"
        assert doc.title == ""
        assert doc.metadata == {}

    def test_full(self) -> None:
        doc = DatasetDocument(doc_id="d2", content="body", title="T", metadata={"k": "v"})
        assert doc.title == "T"
        assert doc.metadata == {"k": "v"}

    def test_metadata_default_is_independent(self) -> None:
        a = DatasetDocument(doc_id="a", content="x")
        b = DatasetDocument(doc_id="b", content="y")
        a.metadata["only_a"] = 1
        assert b.metadata == {}

    def test_missing_required_doc_id(self) -> None:
        with pytest.raises(ValidationError):
            DatasetDocument(content="no id")

    def test_missing_required_content(self) -> None:
        with pytest.raises(ValidationError):
            DatasetDocument(doc_id="d")


class TestGraphRAGQuestion:
    def _valid_kwargs(self, **overrides: object) -> dict:
        base = {
            "question_id": "q1",
            "question": "Who?",
            "question_type": "MC",
            "difficulty": "fact_retrieval",
            "gold_answer": "A",
        }
        base.update(overrides)
        return base

    def test_minimal_defaults(self) -> None:
        q = GraphRAGQuestion(**self._valid_kwargs())
        assert q.discipline == ""
        assert q.source_textbook == ""
        assert q.options is None
        assert q.evidence == []
        assert q.relevant_doc_ids == []
        assert q.metadata == {}

    @pytest.mark.parametrize("qtype", ["MC", "MS", "TF", "FB", "OE"])
    def test_all_question_types_accepted(self, qtype: str) -> None:
        q = GraphRAGQuestion(**self._valid_kwargs(question_type=qtype))
        assert q.question_type == qtype

    @pytest.mark.parametrize(
        "difficulty",
        [
            "fact_retrieval",
            "complex_reasoning",
            "contextual_summarization",
            "creative_generation",
        ],
    )
    def test_all_difficulties_accepted(self, difficulty: str) -> None:
        q = GraphRAGQuestion(**self._valid_kwargs(difficulty=difficulty))
        assert q.difficulty == difficulty

    def test_invalid_question_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GraphRAGQuestion(**self._valid_kwargs(question_type="ZZ"))

    def test_invalid_difficulty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GraphRAGQuestion(**self._valid_kwargs(difficulty="impossible"))

    def test_missing_gold_answer_rejected(self) -> None:
        kwargs = self._valid_kwargs()
        del kwargs["gold_answer"]
        with pytest.raises(ValidationError):
            GraphRAGQuestion(**kwargs)

    def test_options_and_lists(self) -> None:
        q = GraphRAGQuestion(
            **self._valid_kwargs(
                options=["A", "B", "C"],
                evidence=["e1", "e2"],
                relevant_doc_ids=["d1"],
            )
        )
        assert q.options == ["A", "B", "C"]
        assert q.evidence == ["e1", "e2"]
        assert q.relevant_doc_ids == ["d1"]


class TestGraphRAGDataset:
    def _doc(self, doc_id: str = "d1") -> DatasetDocument:
        return DatasetDocument(doc_id=doc_id, content="content of " + doc_id)

    def _question(self, qid: str = "q1") -> GraphRAGQuestion:
        return GraphRAGQuestion(
            question_id=qid,
            question="Who?",
            question_type="OE",
            difficulty="fact_retrieval",
            gold_answer="A",
        )

    def test_minimal_defaults(self) -> None:
        ds = GraphRAGDataset(name="ds")
        assert ds.version == "1.0"
        assert ds.description == ""
        assert ds.documents == []
        assert ds.questions == []
        assert ds.entity_types == []
        assert ds.relationship_types == []

    def test_missing_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GraphRAGDataset()

    def test_full_construction(self) -> None:
        ds = GraphRAGDataset(
            name="ds",
            documents=[self._doc()],
            questions=[self._question()],
            entity_types=["PERSON"],
            relationship_types=["RELATES_TO"],
        )
        assert ds.documents[0].doc_id == "d1"
        assert ds.questions[0].question_id == "q1"
        assert ds.entity_types == ["PERSON"]

    def test_nested_dicts_coerced_to_models(self) -> None:
        ds = GraphRAGDataset(
            name="ds",
            documents=[{"doc_id": "d9", "content": "x"}],
            questions=[
                {
                    "question_id": "q9",
                    "question": "Q?",
                    "question_type": "TF",
                    "difficulty": "complex_reasoning",
                    "gold_answer": "true",
                }
            ],
        )
        assert isinstance(ds.documents[0], DatasetDocument)
        assert isinstance(ds.questions[0], GraphRAGQuestion)

    def test_save_then_load_roundtrip(self, tmp_path) -> None:
        ds = GraphRAGDataset(
            name="ds",
            documents=[self._doc(), self._doc("d2")],
            questions=[self._question()],
            entity_types=["PERSON"],
        )
        path = tmp_path / "nested" / "dataset.json"
        ds.save(path)
        assert path.exists()
        loaded = GraphRAGDataset.load(path)
        assert loaded == ds

    def test_save_creates_parent_dirs(self, tmp_path) -> None:
        ds = GraphRAGDataset(name="ds")
        path = tmp_path / "a" / "b" / "c" / "ds.json"
        ds.save(path)
        assert path.exists()

    def test_save_writes_indented_json(self, tmp_path) -> None:
        ds = GraphRAGDataset(name="ds")
        path = tmp_path / "ds.json"
        ds.save(path)
        text = path.read_text()
        # indent=2 produces newlines + leading spaces.
        assert "\n  " in text
        assert json.loads(text)["name"] == "ds"

    def test_load_accepts_str_path(self, tmp_path) -> None:
        ds = GraphRAGDataset(name="ds")
        path = tmp_path / "ds.json"
        ds.save(path)
        loaded = GraphRAGDataset.load(str(path))
        assert loaded.name == "ds"


class TestComputeHash:
    def _ds(self, **overrides: object) -> GraphRAGDataset:
        kwargs = {
            "name": "ds",
            "documents": [DatasetDocument(doc_id="d1", content="hello")],
            "questions": [
                GraphRAGQuestion(
                    question_id="q1",
                    question="Q?",
                    question_type="OE",
                    difficulty="fact_retrieval",
                    gold_answer="A",
                )
            ],
        }
        kwargs.update(overrides)
        return GraphRAGDataset(**kwargs)

    def test_hash_is_16_char_hex(self) -> None:
        h = self._ds().compute_hash()
        assert len(h) == 16
        int(h, 16)  # raises if not hex

    def test_hash_is_deterministic(self) -> None:
        assert self._ds().compute_hash() == self._ds().compute_hash()

    def test_hash_changes_with_documents(self) -> None:
        a = self._ds()
        b = self._ds(documents=[DatasetDocument(doc_id="d1", content="different")])
        assert a.compute_hash() != b.compute_hash()

    def test_hash_changes_with_questions(self) -> None:
        a = self._ds()
        b = self._ds(
            questions=[
                GraphRAGQuestion(
                    question_id="q2",
                    question="Other?",
                    question_type="OE",
                    difficulty="fact_retrieval",
                    gold_answer="B",
                )
            ]
        )
        assert a.compute_hash() != b.compute_hash()

    def test_hash_ignores_name_and_metadata_fields(self) -> None:
        # compute_hash only fingerprints documents + questions.
        a = self._ds(name="alpha", description="x", version="1.0")
        b = self._ds(name="beta", description="y", version="9.9")
        assert a.compute_hash() == b.compute_hash()

    def test_hash_ignores_entity_types(self) -> None:
        a = self._ds(entity_types=["PERSON"])
        b = self._ds(entity_types=["LOCATION", "EVENT"])
        assert a.compute_hash() == b.compute_hash()

    def test_empty_dataset_hash(self) -> None:
        h = GraphRAGDataset(name="empty").compute_hash()
        assert len(h) == 16
