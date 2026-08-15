"""Tests for hierarchical memory categories and categorizer."""


from ctxforge.core.categories import CategoryAssignment, MemoryCategory
from ctxforge.core.memory import MemoryItem, MemorySource, MemoryType
from ctxforge.extraction.categorizer import MemoryCategorizer


def test_memory_category_defaults():
    cat = MemoryCategory(name="Travel")
    assert cat.category_id  # UUID generated
    assert cat.description == ""
    assert cat.summary is None
    assert cat.embedding is None


def test_category_assignment_creation():
    assignment = CategoryAssignment(
        memory_id="m1", category_id="c1", confidence=0.85
    )
    assert assignment.memory_id == "m1"
    assert assignment.category_id == "c1"
    assert assignment.confidence == 0.85


def _make_memory(embedding=None, memory_id="m1"):
    mem = MemoryItem(
        user_id="u1",
        content="test",
        type=MemoryType.SEMANTIC,
        source=MemorySource.AGENT_INFERENCE,
        embedding=embedding,
    )
    mem.memory_id = memory_id
    return mem


def test_categorizer_match():
    cat = MemoryCategory(name="Food", embedding=[1.0, 0.0, 0.0])
    categorizer = MemoryCategorizer(categories=[cat], similarity_threshold=0.5)
    memory = _make_memory(embedding=[0.9, 0.1, 0.0])
    assignments = categorizer.categorize(memory)
    assert len(assignments) == 1
    assert assignments[0].category_id == cat.category_id


def test_categorizer_no_match():
    cat = MemoryCategory(name="Food", embedding=[1.0, 0.0, 0.0])
    categorizer = MemoryCategorizer(categories=[cat], similarity_threshold=0.9)
    memory = _make_memory(embedding=[0.0, 1.0, 0.0])
    assignments = categorizer.categorize(memory)
    assert len(assignments) == 0


def test_categorizer_no_embedding():
    cat = MemoryCategory(name="Food", embedding=[1.0, 0.0, 0.0])
    categorizer = MemoryCategorizer(categories=[cat])
    memory = _make_memory(embedding=None)
    assignments = categorizer.categorize(memory)
    assert len(assignments) == 0


def test_auto_create_category():
    m1 = _make_memory(embedding=[1.0, 0.0, 0.0])
    m2 = _make_memory(embedding=[0.0, 1.0, 0.0])
    m3 = _make_memory(embedding=[0.0, 0.0, 1.0])

    categorizer = MemoryCategorizer()
    cat = categorizer.auto_create_category([m1, m2, m3], name="Mixed")
    assert cat.name == "Mixed"
    assert cat.embedding is not None
    # Centroid should be approximately [1/3, 1/3, 1/3]
    for v in cat.embedding:
        assert abs(v - 1 / 3) < 1e-9


def test_auto_create_category_no_embeddings():
    m1 = _make_memory(embedding=None)
    categorizer = MemoryCategorizer()
    cat = categorizer.auto_create_category([m1], name="Empty")
    assert cat.name == "Empty"
    assert cat.embedding is None
