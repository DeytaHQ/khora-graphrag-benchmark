"""Unit tests for ``khora_graphrag_bench.datasets.converters``."""

from __future__ import annotations

import hashlib

from khora_graphrag_bench.datasets.converters import (
    _hash_content,
    graphrag_bench_to_dataset,
)
from khora_graphrag_bench.datasets.schema import GraphRAGDataset


def _question_record(**overrides: object) -> dict:
    base = {
        "id": "Novel-abc123",
        "source": "Novel-44557",
        "question": "Who is the protagonist?",
        "answer": "Alice",
        "question_type": "Fact Retrieval",
        "evidence": ["Alice is the hero.", "She leads the party."],
        "evidence_triple": [["Alice", "is", "hero"]],
    }
    base.update(overrides)
    return base


def _corpus_record(corpus_name: str = "Novel-44557", context: str = "Full novel text about Alice.") -> dict:
    return {"corpus_name": corpus_name, "context": context}


class TestHashContent:
    def test_matches_sha256(self) -> None:
        assert _hash_content("hello") == hashlib.sha256(b"hello").hexdigest()

    def test_deterministic(self) -> None:
        assert _hash_content("x") == _hash_content("x")

    def test_differs_for_different_input(self) -> None:
        assert _hash_content("a") != _hash_content("b")


class TestCorpusDocuments:
    def test_builds_one_doc_per_novel(self) -> None:
        corpus = [_corpus_record("Novel-1", "text 1"), _corpus_record("Novel-2", "text 2")]
        ds = graphrag_bench_to_dataset([], corpus=corpus)
        assert {d.doc_id for d in ds.documents} == {"Novel-1", "Novel-2"}
        doc1 = next(d for d in ds.documents if d.doc_id == "Novel-1")
        assert doc1.content == "text 1"
        assert doc1.title == "Novel-1"
        assert doc1.metadata == {"source_textbook": "Novel-1"}

    def test_source_fallback_key_used_for_corpus_name(self) -> None:
        corpus = [{"source": "Novel-X", "context": "body"}]
        ds = graphrag_bench_to_dataset([], corpus=corpus)
        assert ds.documents[0].doc_id == "Novel-X"

    def test_unknown_corpus_name_when_no_keys(self) -> None:
        corpus = [{"context": "body"}]
        ds = graphrag_bench_to_dataset([], corpus=corpus)
        assert ds.documents[0].doc_id == "unknown"

    def test_skips_empty_context(self) -> None:
        corpus = [_corpus_record("Novel-1", ""), _corpus_record("Novel-2", "ok")]
        ds = graphrag_bench_to_dataset([], corpus=corpus)
        assert [d.doc_id for d in ds.documents] == ["Novel-2"]

    def test_dedupes_repeated_corpus_name(self) -> None:
        corpus = [_corpus_record("Novel-1", "first"), _corpus_record("Novel-1", "second")]
        ds = graphrag_bench_to_dataset([], corpus=corpus)
        assert len(ds.documents) == 1
        # First wins; the duplicate is skipped.
        assert ds.documents[0].content == "first"


class TestQuestionConversionWithCorpus:
    def test_question_linked_to_source_novel(self) -> None:
        corpus = [_corpus_record("Novel-44557")]
        ds = graphrag_bench_to_dataset([_question_record()], corpus=corpus)
        assert len(ds.questions) == 1
        q = ds.questions[0]
        assert q.question_id == "Novel-abc123"
        assert q.relevant_doc_ids == ["Novel-44557"]
        assert q.discipline == "Novel-44557"
        assert q.source_textbook == "Novel-44557"
        assert q.gold_answer == "Alice"
        assert q.evidence == ["Alice is the hero.", "She leads the party."]
        assert q.metadata["evidence_relations"] == [["Alice", "is", "hero"]]

    def test_no_extra_documents_when_source_matches(self) -> None:
        corpus = [_corpus_record("Novel-44557")]
        ds = graphrag_bench_to_dataset([_question_record()], corpus=corpus)
        # Only the corpus doc; no reconstructed fallback.
        assert [d.doc_id for d in ds.documents] == ["Novel-44557"]


class TestQuestionTypeMapping:
    def _convert_qtype(self, raw_qtype: str) -> tuple[str, str]:
        ds = graphrag_bench_to_dataset([_question_record(question_type=raw_qtype)])
        q = ds.questions[0]
        return q.question_type, q.difficulty

    def test_fact_retrieval(self) -> None:
        assert self._convert_qtype("Fact Retrieval") == ("FB", "fact_retrieval")

    def test_complex_reasoning(self) -> None:
        assert self._convert_qtype("Complex Reasoning") == ("OE", "complex_reasoning")

    def test_contextual_summarize(self) -> None:
        assert self._convert_qtype("Contextual Summarize") == ("OE", "contextual_summarization")

    def test_contextual_summarization(self) -> None:
        assert self._convert_qtype("Contextual Summarization") == ("OE", "contextual_summarization")

    def test_creative_generation(self) -> None:
        assert self._convert_qtype("Creative Generation") == ("OE", "creative_generation")

    def test_case_insensitive(self) -> None:
        assert self._convert_qtype("FACT RETRIEVAL") == ("FB", "fact_retrieval")
        assert self._convert_qtype("Fact retrieval") == ("FB", "fact_retrieval")

    def test_leading_trailing_whitespace_stripped(self) -> None:
        assert self._convert_qtype("  complex reasoning  ") == ("OE", "complex_reasoning")

    def test_internal_whitespace_not_collapsed(self) -> None:
        # The converter only lower()/strip()s; extra internal spaces do not match
        # the lookup table, so it falls through to the default.
        assert self._convert_qtype("fact  retrieval") == ("OE", "fact_retrieval")

    def test_unknown_type_defaults(self) -> None:
        assert self._convert_qtype("Something Else") == ("OE", "fact_retrieval")

    def test_missing_type_defaults(self) -> None:
        ds = graphrag_bench_to_dataset([_question_record(question_type=None)])
        q = ds.questions[0]
        assert (q.question_type, q.difficulty) == ("OE", "fact_retrieval")


class TestFallbackReconstruction:
    def test_reconstructs_doc_from_context_when_no_corpus(self) -> None:
        rec = _question_record(context="explicit context", source="Novel-99")
        ds = graphrag_bench_to_dataset([rec])
        assert len(ds.documents) == 1
        doc = ds.documents[0]
        assert doc.content == "explicit context"
        assert doc.title == "Novel-99"
        assert doc.metadata["reconstructed_from_evidence"] is True
        # doc_id is "<source>_<12-hex>".
        assert doc.doc_id.startswith("Novel-99_")
        assert ds.questions[0].relevant_doc_ids == [doc.doc_id]

    def test_reconstructs_from_joined_evidence_when_no_context(self) -> None:
        rec = _question_record(source="Novel-99", evidence=["a.", "b."])
        rec.pop("context", None)
        ds = graphrag_bench_to_dataset([rec])
        assert ds.documents[0].content == "a. b."

    def test_no_context_no_evidence_yields_empty_doc_ids(self) -> None:
        rec = _question_record(source="Novel-99", evidence=[])
        rec.pop("context", None)
        ds = graphrag_bench_to_dataset([rec])
        assert ds.documents == []
        assert ds.questions[0].relevant_doc_ids == []

    def test_identical_fallback_context_deduped(self) -> None:
        rec1 = _question_record(id="q1", source="Novel-99", context="same body")
        rec2 = _question_record(id="q2", source="Novel-99", context="same body")
        ds = graphrag_bench_to_dataset([rec1, rec2])
        assert len(ds.documents) == 1
        # Both questions point at the single shared reconstructed doc.
        assert ds.questions[0].relevant_doc_ids == ds.questions[1].relevant_doc_ids


class TestGoldAnswerAndIds:
    def test_gold_answer_falls_back_to_gold_answer_key(self) -> None:
        rec = _question_record()
        del rec["answer"]
        rec["gold_answer"] = "from gold key"
        ds = graphrag_bench_to_dataset([rec])
        assert ds.questions[0].gold_answer == "from gold key"

    def test_missing_answer_defaults_empty(self) -> None:
        rec = _question_record()
        del rec["answer"]
        ds = graphrag_bench_to_dataset([rec])
        assert ds.questions[0].gold_answer == ""

    def test_missing_id_gets_positional_fallback(self) -> None:
        rec1 = _question_record(question="Q1?")
        del rec1["id"]
        rec2 = _question_record(question="Q2?")
        del rec2["id"]
        ds = graphrag_bench_to_dataset([rec1, rec2])
        # First question id uses len(questions)==0, second is 1.
        assert ds.questions[0].question_id == "q_0"
        assert ds.questions[1].question_id == "q_1"


class TestSkippingAndEmpty:
    def test_question_without_text_is_skipped(self) -> None:
        rec = _question_record(question="")
        ds = graphrag_bench_to_dataset([rec])
        assert ds.questions == []

    def test_missing_question_key_is_skipped(self) -> None:
        rec = _question_record()
        del rec["question"]
        ds = graphrag_bench_to_dataset([rec])
        assert ds.questions == []

    def test_empty_inputs(self) -> None:
        ds = graphrag_bench_to_dataset([])
        assert ds.documents == []
        assert ds.questions == []

    def test_corpus_none_treated_as_empty(self) -> None:
        ds = graphrag_bench_to_dataset([], corpus=None)
        assert ds.documents == []


class TestDatasetMetadata:
    def test_returns_dataset_instance(self) -> None:
        ds = graphrag_bench_to_dataset([])
        assert isinstance(ds, GraphRAGDataset)

    def test_name_default(self) -> None:
        assert graphrag_bench_to_dataset([]).name == "graphrag_bench_novel"

    def test_custom_name(self) -> None:
        assert graphrag_bench_to_dataset([], name="custom").name == "custom"

    def test_version_and_description(self) -> None:
        ds = graphrag_bench_to_dataset([])
        assert ds.version == "1.0"
        assert "GraphRAG-Bench" in ds.description

    def test_entity_and_relationship_types_passed_through(self) -> None:
        ds = graphrag_bench_to_dataset(
            [],
            entity_types=["PERSON", "LOCATION"],
            relationship_types=["RELATES_TO"],
        )
        assert ds.entity_types == ["PERSON", "LOCATION"]
        assert ds.relationship_types == ["RELATES_TO"]

    def test_types_default_to_empty_lists(self) -> None:
        ds = graphrag_bench_to_dataset([])
        assert ds.entity_types == []
        assert ds.relationship_types == []


class TestLoggingWarnings:
    def test_warns_when_no_corpus(self, caplog) -> None:
        with caplog.at_level("WARNING"):
            graphrag_bench_to_dataset([_question_record(context="ctx")])
        assert any("no source corpus provided" in r.message for r in caplog.records)

    def test_warns_when_some_sources_have_no_corpus(self, caplog) -> None:
        corpus = [_corpus_record("Novel-1")]
        matched = _question_record(id="m", source="Novel-1")
        unmatched = _question_record(id="u", source="Novel-2", context="ctx")
        with caplog.at_level("WARNING"):
            graphrag_bench_to_dataset([matched, unmatched], corpus=corpus)
        assert any("no matching corpus novel" in r.message for r in caplog.records)

    def test_no_warning_when_all_matched(self, caplog) -> None:
        corpus = [_corpus_record("Novel-44557")]
        with caplog.at_level("WARNING"):
            graphrag_bench_to_dataset([_question_record()], corpus=corpus)
        assert not [r for r in caplog.records if r.levelname == "WARNING"]
