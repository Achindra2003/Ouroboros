from __future__ import annotations

from ouroboros import memory
from ouroboros.memory import LexicalEmbedder, cosine_similarity, semantic_search


def test_lexical_embedder_normalized_and_dim():
    emb = LexicalEmbedder(dim=64)
    [vec] = emb.embed(["recursive self modeling"])
    assert len(vec) == 64
    assert abs(sum(v * v for v in vec) - 1.0) < 1e-9  # L2 normalized


def test_cosine_identical_text_is_one():
    emb = LexicalEmbedder()
    a, b = emb.embed(["the mind observes itself", "the mind observes itself"])
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-9


def test_cosine_disjoint_text_is_zero():
    emb = LexicalEmbedder()
    a, b = emb.embed(["consciousness recursion"], )[0], emb.embed(["weather umbrella"])[0]
    assert cosine_similarity(a, b) == 0.0


def test_semantic_search_ranks_relevant_first():
    emb = LexicalEmbedder()
    memories = [
        "Consciousness arises from recursive self-modeling",
        "The weather affects mood significantly",
    ]
    results = semantic_search("What is recursive self-modeling?", memories, k=2, embedder=emb)
    assert results[0][0] == memories[0]
    assert results[0][1] > 0
    # The unrelated memory should score lower (often zero).
    assert results[0][1] >= results[1][1]


def test_semantic_search_empty_memories():
    assert semantic_search("anything", [], embedder=LexicalEmbedder()) == []


def test_semantic_search_respects_k():
    emb = LexicalEmbedder()
    memories = [f"memory number {i} about thought" for i in range(10)]
    results = semantic_search("thought", memories, k=3, embedder=emb)
    assert len(results) == 3


def test_get_embedder_falls_back_to_lexical_without_sentence_transformers(monkeypatch):
    import sys

    memory.get_embedder.cache_clear()
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    embedder = memory.get_embedder()
    assert isinstance(embedder, LexicalEmbedder)
    memory.get_embedder.cache_clear()
