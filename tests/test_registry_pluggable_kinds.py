"""Pluggable component kinds: vectorstore, graph_store, tokenizer,
expertise_retriever (+ middleware_factory) via register_*/get_* and the
generic register_component/get_component paths (P0-ext)."""

import pytest

from engine.registry import ComponentRegistry


@pytest.fixture()
def registry():
    return ComponentRegistry()


class _V:
    pass


class _G:
    pass


class _T:
    pass


class _E:
    pass


def test_new_kinds_register_get_list(registry):
    registry.register_vector_store("myvec")(_V)
    registry.register_graph_store("mygraph")(_G)
    registry.register_tokenizer("mytok")(_T)
    registry.register_expertise_retriever("myexp")(_E)
    assert registry.get_vector_store("MYVEC") is _V  # case-insensitive
    assert registry.get_graph_store("mygraph") is _G
    assert registry.get_tokenizer("mytok") is _T
    assert registry.get_expertise_retriever("myexp") is _E
    assert "myvec" in registry.list_vector_stores()
    assert "mygraph" in registry.list_graph_stores()
    assert "mytok" in registry.list_tokenizers()
    assert "myexp" in registry.list_expertise_retrievers()


def test_generic_component_paths(registry):
    registry.register_component("vector_store", "v1", _V)
    registry.register_component("vectorstore", "v2", _V)  # alias
    registry.register_component("graph_store", "g1", _G)
    registry.register_component("tokenizer", "t1", _T)
    registry.register_component("expertise_retriever", "e1", _E)
    assert registry.get_component("vector_store", "V1") is _V
    assert registry.get_component("vectorstore", "v2") is _V
    assert registry.get_component("graph_store", "g1") is _G
    assert registry.get_component("tokenizer", "t1") is _T
    assert registry.get_component("expertise_retriever", "e1") is _E
    assert registry.get_component("middleware_factory", "missing") is None
    assert registry.get_component("bogus-kind", "x") is None


def test_generic_component_decorator_and_class_path(registry):
    @registry.register_component("tokenizer", "deco")
    class _D:
        pass

    assert registry.get_tokenizer("deco") is _D
    registry.register_component_class_path(
        "graph_store", "cp", "engine.registry:ComponentRegistry"
    )
    from engine.registry import ComponentRegistry as CR

    assert registry.get_graph_store("cp") is CR


def test_unknown_component_type_raises(registry):
    with pytest.raises(ValueError, match="Unknown component type"):
        registry.register_component("nope", "x", _V)


def test_clear_covers_new_kinds(registry):
    registry.register_vector_store("v")(_V)
    registry.register_graph_store("g")(_G)
    registry.register_tokenizer("t")(_T)
    registry.register_expertise_retriever("e")(_E)
    registry.clear()
    assert registry.list_vector_stores() == []
    assert registry.list_graph_stores() == []
    assert registry.list_tokenizers() == []
    assert registry.list_expertise_retrievers() == []
