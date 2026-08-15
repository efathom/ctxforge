"""Tests for inline memory reference utilities."""

from ctxforge.utils.references import (
    build_reference_map,
    extract_references,
    format_as_citations,
    strip_references,
)


def test_extract_references_single():
    assert extract_references("See [ref:abc] for details") == ["abc"]


def test_extract_references_comma_separated():
    refs = extract_references("Check [ref:a,b,c] here")
    assert refs == ["a", "b", "c"]


def test_extract_references_multiple_markers():
    text = "First [ref:x1] and then [ref:y2,z3]."
    refs = extract_references(text)
    assert refs == ["x1", "y2", "z3"]


def test_extract_references_empty():
    assert extract_references("No references here.") == []


def test_extract_references_dedup():
    refs = extract_references("[ref:a] and [ref:a,b]")
    assert refs == ["a", "b"]


def test_strip_references():
    text = "Hello [ref:abc] world."
    assert strip_references(text) == "Hello world."


def test_strip_references_multiple():
    text = "A [ref:x] B [ref:y,z] ."
    result = strip_references(text)
    assert result == "A B."


def test_format_as_citations_numbered():
    memory_map = {"id1": "First fact", "id2": "Second fact"}
    result = format_as_citations(memory_map)
    assert "[1] (id1): First fact" in result
    assert "[2] (id2): Second fact" in result


def test_build_reference_map():
    text = "See [ref:a,b] for context."
    lookup = {"a": "Fact A", "b": "Fact B", "c": "Fact C"}
    result = build_reference_map(text, lookup)
    assert result == {"a": "Fact A", "b": "Fact B"}


def test_build_reference_map_missing():
    text = "See [ref:a,missing] for context."
    lookup = {"a": "Fact A"}
    result = build_reference_map(text, lookup)
    assert result == {"a": "Fact A"}
