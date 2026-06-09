"""Unit tests for ``khora_graphrag_bench.datasets.loader``.

No network or download happens: ``urlopen`` is mocked and the parsing/conversion
of in-memory sample records is what gets exercised.
"""

from __future__ import annotations

import json
from unittest import mock

import pytest

from khora_graphrag_bench.datasets import loader
from khora_graphrag_bench.datasets.loader import (
    GRAPHRAG_BENCH_NOVEL_CORPUS_URL,
    GRAPHRAG_BENCH_NOVEL_URL,
    NOVEL_ENTITY_TYPES,
    NOVEL_RELATIONSHIP_TYPES,
    _fetch_cached,
    load_graphrag_bench,
)
from khora_graphrag_bench.datasets.schema import GraphRAGDataset

SAMPLE_QUESTIONS = [
    {
        "id": "Novel-q1",
        "source": "Novel-1",
        "question": "Who is the hero?",
        "answer": "Alice",
        "question_type": "Fact Retrieval",
        "evidence": ["Alice is the hero."],
        "evidence_triple": [["Alice", "is", "hero"]],
    }
]

SAMPLE_CORPUS = [{"corpus_name": "Novel-1", "context": "A full novel about Alice the hero."}]


class TestConstants:
    def test_novel_entity_types(self) -> None:
        assert NOVEL_ENTITY_TYPES == [
            "PERSON",
            "ORGANIZATION",
            "LOCATION",
            "EVENT",
            "CONCEPT",
            "OBJECT",
            "CREATURE",
        ]

    def test_novel_relationship_types(self) -> None:
        assert NOVEL_RELATIONSHIP_TYPES == [
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

    def test_type_lists_have_no_duplicates(self) -> None:
        assert len(NOVEL_ENTITY_TYPES) == len(set(NOVEL_ENTITY_TYPES))
        assert len(NOVEL_RELATIONSHIP_TYPES) == len(set(NOVEL_RELATIONSHIP_TYPES))

    def test_urls_point_at_huggingface(self) -> None:
        assert GRAPHRAG_BENCH_NOVEL_URL.startswith("https://huggingface.co/")
        assert GRAPHRAG_BENCH_NOVEL_URL.endswith("novel_questions.json")
        assert GRAPHRAG_BENCH_NOVEL_CORPUS_URL.endswith("novel.json")


def _fake_urlopen(payload: object):
    """Return a context-manager mock whose ``.read()`` yields ``payload`` as JSON bytes."""
    resp = mock.MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    cm = mock.MagicMock()
    cm.__enter__.return_value = resp
    return cm


class TestFetchCached:
    def test_downloads_when_file_missing(self, tmp_path) -> None:
        path = tmp_path / "data.json"
        with mock.patch.object(loader, "urlopen", return_value=_fake_urlopen([1, 2, 3])) as m:
            result = _fetch_cached("http://x", path, force_download=False, label="x")
        m.assert_called_once_with("http://x")
        assert result == [1, 2, 3]
        # File was written to cache.
        assert json.loads(path.read_text()) == [1, 2, 3]

    def test_reads_cache_without_download(self, tmp_path) -> None:
        path = tmp_path / "data.json"
        path.write_text(json.dumps({"cached": True}))
        with mock.patch.object(loader, "urlopen") as m:
            result = _fetch_cached("http://x", path, force_download=False, label="x")
        m.assert_not_called()
        assert result == {"cached": True}

    def test_force_download_overwrites_cache(self, tmp_path) -> None:
        path = tmp_path / "data.json"
        path.write_text(json.dumps(["stale"]))
        with mock.patch.object(loader, "urlopen", return_value=_fake_urlopen(["fresh"])) as m:
            result = _fetch_cached("http://x", path, force_download=True, label="x")
        m.assert_called_once()
        assert result == ["fresh"]
        assert json.loads(path.read_text()) == ["fresh"]


class TestLoadGraphragBench:
    def _patch_fetch(self, questions: object, corpus: object):
        """Patch ``_fetch_cached`` to return the right payload per URL (no IO/network)."""

        def side_effect(url, path, force_download, label):  # noqa: ANN001
            if url == GRAPHRAG_BENCH_NOVEL_URL:
                return questions
            if url == GRAPHRAG_BENCH_NOVEL_CORPUS_URL:
                return corpus
            raise AssertionError(f"unexpected url {url}")

        return mock.patch.object(loader, "_fetch_cached", side_effect=side_effect)

    def test_happy_path_builds_dataset(self, tmp_path) -> None:
        with self._patch_fetch(SAMPLE_QUESTIONS, SAMPLE_CORPUS):
            ds = load_graphrag_bench(cache_dir=tmp_path)
        assert isinstance(ds, GraphRAGDataset)
        assert ds.name == "graphrag_bench_novel"
        assert ds.entity_types == NOVEL_ENTITY_TYPES
        assert ds.relationship_types == NOVEL_RELATIONSHIP_TYPES
        assert [d.doc_id for d in ds.documents] == ["Novel-1"]
        assert ds.questions[0].question_id == "Novel-q1"
        assert ds.questions[0].relevant_doc_ids == ["Novel-1"]

    def test_creates_cache_dir(self, tmp_path) -> None:
        cache = tmp_path / "nested" / "cache"
        with self._patch_fetch(SAMPLE_QUESTIONS, SAMPLE_CORPUS):
            load_graphrag_bench(cache_dir=cache)
        assert cache.is_dir()

    def test_passes_force_download_through(self, tmp_path) -> None:
        seen_force: list[bool] = []

        def side_effect(url, path, force_download, label):  # noqa: ANN001
            seen_force.append(force_download)
            return SAMPLE_QUESTIONS if url == GRAPHRAG_BENCH_NOVEL_URL else SAMPLE_CORPUS

        with mock.patch.object(loader, "_fetch_cached", side_effect=side_effect):
            load_graphrag_bench(cache_dir=tmp_path, force_download=True)
        assert seen_force == [True, True]

    def test_fetches_expected_cache_paths(self, tmp_path) -> None:
        seen_paths: dict[str, str] = {}

        def side_effect(url, path, force_download, label):  # noqa: ANN001
            seen_paths[url] = path.name
            return SAMPLE_QUESTIONS if url == GRAPHRAG_BENCH_NOVEL_URL else SAMPLE_CORPUS

        with mock.patch.object(loader, "_fetch_cached", side_effect=side_effect):
            load_graphrag_bench(cache_dir=tmp_path)
        assert seen_paths[GRAPHRAG_BENCH_NOVEL_URL] == "novel_questions.json"
        assert seen_paths[GRAPHRAG_BENCH_NOVEL_CORPUS_URL] == "novel_corpus.json"

    def test_raises_when_questions_not_a_list(self, tmp_path) -> None:
        with self._patch_fetch({"not": "a list"}, SAMPLE_CORPUS):
            with pytest.raises(ValueError, match="questions JSON must be a list"):
                load_graphrag_bench(cache_dir=tmp_path)

    def test_raises_when_corpus_not_a_list(self, tmp_path) -> None:
        with self._patch_fetch(SAMPLE_QUESTIONS, {"not": "a list"}):
            with pytest.raises(ValueError, match="corpus JSON must be a list"):
                load_graphrag_bench(cache_dir=tmp_path)

    def test_no_network_call_made(self, tmp_path) -> None:
        # Guards that the whole load path never touches urlopen when _fetch_cached is mocked.
        with self._patch_fetch(SAMPLE_QUESTIONS, SAMPLE_CORPUS):
            with mock.patch.object(loader, "urlopen") as m:
                load_graphrag_bench(cache_dir=tmp_path)
        m.assert_not_called()

    def test_end_to_end_with_mocked_urlopen(self, tmp_path) -> None:
        # Exercise the real _fetch_cached + conversion with urlopen mocked (still no network).
        def fake(url, *a, **kw):  # noqa: ANN001
            payload = SAMPLE_QUESTIONS if url == GRAPHRAG_BENCH_NOVEL_URL else SAMPLE_CORPUS
            return _fake_urlopen(payload)

        with mock.patch.object(loader, "urlopen", side_effect=fake):
            ds = load_graphrag_bench(cache_dir=tmp_path)
        assert ds.questions[0].question_id == "Novel-q1"
        # Cache files were written.
        assert (tmp_path / "novel_questions.json").exists()
        assert (tmp_path / "novel_corpus.json").exists()
