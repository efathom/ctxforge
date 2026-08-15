"""
Tests for Phase 5: Visualization & Debugging.

Tests the visualization module for extractions.
"""

import os
import tempfile

from ctxforge.core.alignment_types import AlignmentStatus, CharSpan
from ctxforge.core.memory import MemoryType
from ctxforge.protocols.extractor import ExtractionCandidate
from ctxforge.protocols.graph import GraphEdge, GraphNode
from ctxforge.visualization import (
    HighlightedSpan,
    save_visualization,
    visualize_graph_extractions,
    visualize_memory_extractions,
)


class TestHighlightedSpan:
    """Tests for HighlightedSpan dataclass."""
    
    def test_basic_span(self):
        """Test basic span creation."""
        span = HighlightedSpan(
            start_pos=0,
            end_pos=10,
            label="Test",
            color="#D2E3FC",
            tooltip="Test tooltip",
        )
        
        assert span.start_pos == 0
        assert span.end_pos == 10
        assert span.length == 10
        assert span.label == "Test"
        assert span.color == "#D2E3FC"
    
    def test_span_with_data(self):
        """Test span with extra data."""
        span = HighlightedSpan(
            start_pos=5,
            end_pos=15,
            label="Entity",
            color="#C8E6C9",
            tooltip="Entity details",
            data={"id": "123", "type": "person"},
        )
        
        assert span.data == {"id": "123", "type": "person"}
        assert span.length == 10


class TestVisualizeMemoryExtractions:
    """Tests for visualize_memory_extractions function."""
    
    def test_empty_candidates(self):
        """Test visualization with no candidates."""
        html_output = visualize_memory_extractions(
            source_text="Hello world",
            candidates=[],
        )
        
        assert "<html>" in html_output
        assert "Hello world" in html_output
        assert "Memory Extractions" in html_output
    
    def test_with_aligned_candidates(self):
        """Test visualization with aligned candidates."""
        candidates = [
            ExtractionCandidate(
                content="coffee",
                memory_type=MemoryType.SEMANTIC,
                confidence=0.9,
                source_text="I love coffee every morning",
                source_span=CharSpan(start_pos=7, end_pos=13),  # "coffee"
                alignment_status=AlignmentStatus.MATCH_EXACT,
            ),
        ]
        
        html_output = visualize_memory_extractions(
            source_text="I love coffee every morning",
            candidates=candidates,
        )
        
        assert "<html>" in html_output
        assert "coffee" in html_output
        assert "semantic" in html_output
        assert "highlight" in html_output
    
    def test_candidates_without_spans(self):
        """Test that candidates without spans are skipped."""
        candidates = [
            ExtractionCandidate(
                content="User likes coffee",
                memory_type=MemoryType.SEMANTIC,
                confidence=0.9,
                source_text="I love coffee",
                # No source_span
            ),
        ]
        
        html_output = visualize_memory_extractions(
            source_text="I love coffee",
            candidates=candidates,
        )
        
        # Should still generate HTML
        assert "<html>" in html_output
        # But no highlight spans
        assert 'class="highlight"' not in html_output
    
    def test_multiple_types(self):
        """Test visualization with multiple memory types."""
        candidates = [
            ExtractionCandidate(
                content="coffee",
                memory_type=MemoryType.SEMANTIC,
                confidence=0.9,
                source_text="I love coffee and tea",
                source_span=CharSpan(start_pos=7, end_pos=13),
                alignment_status=AlignmentStatus.MATCH_EXACT,
            ),
            ExtractionCandidate(
                content="tea",
                memory_type=MemoryType.EPISODIC,
                confidence=0.8,
                source_text="I love coffee and tea",
                source_span=CharSpan(start_pos=18, end_pos=21),
                alignment_status=AlignmentStatus.MATCH_EXACT,
            ),
        ]
        
        html_output = visualize_memory_extractions(
            source_text="I love coffee and tea",
            candidates=candidates,
        )
        
        # Should have legend items for both types
        assert "semantic" in html_output
        assert "episodic" in html_output
    
    def test_custom_title(self):
        """Test custom title in visualization."""
        html_output = visualize_memory_extractions(
            source_text="Hello",
            candidates=[],
            title="My Custom Extraction View",
        )
        
        assert "My Custom Extraction View" in html_output


class TestVisualizeGraphExtractions:
    """Tests for visualize_graph_extractions function."""
    
    def test_empty_nodes_and_edges(self):
        """Test visualization with no nodes or edges."""
        html_output = visualize_graph_extractions(
            source_text="Hello world",
            nodes=[],
            edges=[],
            episode_id="ep-1",
        )
        
        assert "<html>" in html_output
        assert "Hello world" in html_output
    
    def test_with_nodes(self):
        """Test visualization with nodes."""
        nodes = [
            GraphNode(
                node_id="node-1",
                scope_id="scope-1",
                name="Alice",
                labels=["Person"],
                source_spans={"ep-1": CharSpan(start_pos=0, end_pos=5)},
                alignment_status=AlignmentStatus.MATCH_EXACT,
            ),
        ]
        
        html_output = visualize_graph_extractions(
            source_text="Alice works at Acme",
            nodes=nodes,
            edges=[],
            episode_id="ep-1",
        )
        
        assert "Alice" in html_output
        assert "Person" in html_output
        assert 'class="highlight"' in html_output
    
    def test_with_edges(self):
        """Test visualization with edges."""
        edges = [
            GraphEdge(
                edge_id="edge-1",
                scope_id="scope-1",
                source_node_id="node-1",
                target_node_id="node-2",
                edge_type="works_at",
                fact="Alice works at Acme",
                source_spans={"ep-1": CharSpan(start_pos=0, end_pos=19)},
                alignment_status=AlignmentStatus.MATCH_EXACT,
            ),
        ]
        
        html_output = visualize_graph_extractions(
            source_text="Alice works at Acme",
            nodes=[],
            edges=edges,
            episode_id="ep-1",
        )
        
        assert "works_at" in html_output
        assert 'class="highlight"' in html_output
    
    def test_different_episode_id(self):
        """Test that nodes/edges for different episodes are skipped."""
        nodes = [
            GraphNode(
                node_id="node-1",
                scope_id="scope-1",
                name="Alice",
                labels=["Person"],
                source_spans={"ep-2": CharSpan(start_pos=0, end_pos=5)},  # Different episode
            ),
        ]
        
        html_output = visualize_graph_extractions(
            source_text="Alice works at Acme",
            nodes=nodes,
            edges=[],
            episode_id="ep-1",  # Looking for ep-1
        )
        
        # Should not have highlights
        assert 'class="highlight"' not in html_output


class TestSaveVisualization:
    """Tests for save_visualization function."""
    
    def test_save_to_file(self):
        """Test saving visualization to file."""
        html_content = "<html><body>Test</body></html>"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.html")
            result = save_visualization(html_content, filepath)
            
            assert result.exists()
            assert result.read_text() == html_content
    
    def test_creates_parent_directories(self):
        """Test that parent directories are created."""
        html_content = "<html><body>Test</body></html>"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "subdir", "another", "test.html")
            result = save_visualization(html_content, filepath)
            
            assert result.exists()
            assert result.read_text() == html_content


class TestHTMLStructure:
    """Tests for HTML structure and escaping."""
    
    def test_html_escaping(self):
        """Test that special characters are properly escaped."""
        candidates = [
            ExtractionCandidate(
                content="<script>alert('xss')</script>",
                memory_type=MemoryType.SEMANTIC,
                confidence=0.9,
                source_text="Text with <script>alert('xss')</script>",
                source_span=CharSpan(start_pos=10, end_pos=39),
                alignment_status=AlignmentStatus.MATCH_EXACT,
            ),
        ]
        
        html_output = visualize_memory_extractions(
            source_text="Text with <script>alert('xss')</script>",
            candidates=candidates,
        )
        
        # User-provided script tags in content should be escaped
        # (Our own legitimate <script> tag for JS functionality is ok)
        assert "&lt;script&gt;" in html_output
        # The text-container div should contain escaped content
        assert "text-container" in html_output
        # The XSS attempt should be escaped in the tooltip and content
        assert "alert(&#x27;xss&#x27;)" in html_output or "alert('xss')" not in html_output.split('<script>')[0]
    
    def test_json_data_attribute(self):
        """Test that JSON data is properly escaped in data attributes."""
        candidates = [
            ExtractionCandidate(
                content="test",
                memory_type=MemoryType.SEMANTIC,
                confidence=0.9,
                source_text='Text with "quotes"',
                source_span=CharSpan(start_pos=0, end_pos=4),
                alignment_status=AlignmentStatus.MATCH_EXACT,
            ),
        ]
        
        html_output = visualize_memory_extractions(
            source_text='Text with "quotes"',
            candidates=candidates,
        )
        
        # Should have data-info attribute
        assert "data-info=" in html_output
    
    def test_css_styles_included(self):
        """Test that CSS styles are included."""
        html_output = visualize_memory_extractions(
            source_text="Hello",
            candidates=[],
        )
        
        assert "<style>" in html_output
        assert ".highlight" in html_output
        assert ".tooltip" in html_output
        assert ".legend" in html_output
    
    def test_javascript_included(self):
        """Test that JavaScript is included."""
        html_output = visualize_memory_extractions(
            source_text="Hello",
            candidates=[],
        )
        
        assert "<script>" in html_output
        assert "addEventListener" in html_output

