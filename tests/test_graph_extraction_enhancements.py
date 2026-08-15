"""
Tests for Phase 3: Graph Extraction Enhancements.

Tests the enhanced graph extraction with:
- Multi-pass extraction
- Source text alignment
- Source episode tracking
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ctxforge.core.alignment_types import AlignmentStatus, CharSpan
from ctxforge.graph.extraction.llm import GraphExtractionConfig, LLMGraphExtractor
from ctxforge.graph.ontology import GraphOntology
from ctxforge.protocols.graph import GraphEdge, GraphEpisode, GraphNode


class TestGraphNodeEnhancements:
    """Tests for GraphNode source grounding fields."""
    
    def test_source_grounding_fields_default(self):
        """Test source grounding fields have correct defaults."""
        node = GraphNode(
            node_id="node-1",
            scope_id="scope-1",
            name="Test Entity",
        )
        
        assert node.source_episode_ids == []
        assert node.source_spans == {}
        assert node.alignment_status is None
        assert node.extraction_confidence == 1.0
    
    def test_source_grounding_fields_set(self):
        """Test setting source grounding fields."""
        span = CharSpan(start_pos=10, end_pos=25)
        node = GraphNode(
            node_id="node-1",
            scope_id="scope-1",
            name="Test Entity",
            source_episode_ids=["ep-1", "ep-2"],
            source_spans={"ep-1": span},
            alignment_status=AlignmentStatus.MATCH_EXACT,
            extraction_confidence=0.95,
        )
        
        assert node.source_episode_ids == ["ep-1", "ep-2"]
        assert node.source_spans == {"ep-1": span}
        assert node.alignment_status == AlignmentStatus.MATCH_EXACT
        assert node.extraction_confidence == 0.95


class TestGraphEdgeEnhancements:
    """Tests for GraphEdge source grounding fields."""
    
    def test_source_grounding_fields_default(self):
        """Test source grounding fields have correct defaults."""
        edge = GraphEdge(
            edge_id="edge-1",
            scope_id="scope-1",
            source_node_id="node-1",
            target_node_id="node-2",
            edge_type="knows",
        )
        
        assert edge.source_episode_ids == []
        assert edge.source_spans == {}
        assert edge.alignment_status is None
        assert edge.extraction_confidence == 1.0
    
    def test_source_grounding_fields_set(self):
        """Test setting source grounding fields."""
        span = CharSpan(start_pos=5, end_pos=20)
        edge = GraphEdge(
            edge_id="edge-1",
            scope_id="scope-1",
            source_node_id="node-1",
            target_node_id="node-2",
            edge_type="knows",
            fact="Alice knows Bob",
            source_episode_ids=["ep-1"],
            source_spans={"ep-1": span},
            alignment_status=AlignmentStatus.MATCH_FUZZY,
            extraction_confidence=0.85,
        )
        
        assert edge.source_episode_ids == ["ep-1"]
        assert edge.source_spans == {"ep-1": span}
        assert edge.alignment_status == AlignmentStatus.MATCH_FUZZY
        assert edge.extraction_confidence == 0.85


class TestGraphExtractionConfig:
    """Tests for GraphExtractionConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = GraphExtractionConfig()
        
        assert config.extraction_passes == 1
        assert config.enable_alignment is True
        assert config.fuzzy_alignment_threshold == 0.75
        assert config.temperature == 0.0
        assert config.max_tokens == 1400
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = GraphExtractionConfig(
            extraction_passes=3,
            enable_alignment=False,
            fuzzy_alignment_threshold=0.9,
            temperature=0.2,
            max_tokens=2000,
        )
        
        assert config.extraction_passes == 3
        assert config.enable_alignment is False
        assert config.fuzzy_alignment_threshold == 0.9
        assert config.temperature == 0.2
        assert config.max_tokens == 2000


class TestLLMGraphExtractorEnhancements:
    """Tests for enhanced LLMGraphExtractor."""
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        provider = MagicMock()
        provider.chat = AsyncMock()
        return provider
    
    @pytest.fixture
    def sample_ontology(self):
        """Create a sample ontology."""
        return GraphOntology(
            entity_types={"Person": None, "Organization": None},
            edge_types={"works_at": None, "knows": None},
            allowed_edges={
                "works_at": [("Person", "Organization")],
                "knows": [("Person", "Person")],
            },
        )
    
    @pytest.fixture
    def sample_episodes(self):
        """Create sample episodes."""
        return [
            GraphEpisode(
                episode_id="ep-1",
                scope_id="scope-1",
                content="Alice works at Acme Corp. Bob knows Alice.",
                content_type="text",
            ),
        ]
    
    @pytest.mark.asyncio
    async def test_extract_with_alignment(self, mock_llm_provider, sample_ontology, sample_episodes):
        """Test extraction with alignment enabled."""
        mock_response = MagicMock()
        mock_response.content = '''
        {
            "entities": [
                {"name": "Alice", "entity_type": "Person"},
                {"name": "Acme Corp", "entity_type": "Organization"}
            ],
            "edges": [
                {"source_name": "Alice", "source_type": "Person", 
                 "edge_type": "works_at", "target_name": "Acme Corp", "target_type": "Organization",
                 "fact": "Alice works at Acme Corp"}
            ]
        }
        '''
        mock_llm_provider.chat.return_value = mock_response
        
        extractor = LLMGraphExtractor(mock_llm_provider)
        
        nodes, edges = await extractor.extract(
            scope_id="scope-1",
            episodes=sample_episodes,
            ontology=sample_ontology,
            enable_alignment=True,
        )
        
        # Should extract entities and edges
        assert len(nodes) >= 1
        assert len(edges) >= 1
        
        # Check that source episode IDs are set
        for node in nodes:
            assert len(node.source_episode_ids) > 0
        
        for edge in edges:
            assert len(edge.source_episode_ids) > 0
    
    @pytest.mark.asyncio
    async def test_extract_without_alignment(self, mock_llm_provider, sample_ontology, sample_episodes):
        """Test extraction with alignment disabled."""
        mock_response = MagicMock()
        mock_response.content = '''
        {
            "entities": [
                {"name": "Alice", "entity_type": "Person"}
            ],
            "edges": []
        }
        '''
        mock_llm_provider.chat.return_value = mock_response
        
        extractor = LLMGraphExtractor(mock_llm_provider)
        
        nodes, edges = await extractor.extract(
            scope_id="scope-1",
            episodes=sample_episodes,
            ontology=sample_ontology,
            enable_alignment=False,
        )
        
        # Should still extract
        assert len(nodes) >= 1
        
        # Without alignment, source_spans should be empty
        for node in nodes:
            assert node.source_spans == {}
    
    @pytest.mark.asyncio
    async def test_multi_pass_extraction(self, mock_llm_provider, sample_ontology, sample_episodes):
        """Test multi-pass extraction finds more entities."""
        call_count = 0
        
        async def mock_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            response = MagicMock()
            if call_count == 1:
                response.content = '''
                {
                    "entities": [{"name": "Alice", "entity_type": "Person"}],
                    "edges": []
                }
                '''
            else:
                response.content = '''
                {
                    "entities": [{"name": "Acme Corp", "entity_type": "Organization"}],
                    "edges": []
                }
                '''
            return response
        
        mock_llm_provider.chat = mock_chat
        
        extractor = LLMGraphExtractor(mock_llm_provider)
        
        nodes, edges = await extractor.extract(
            scope_id="scope-1",
            episodes=sample_episodes,
            ontology=sample_ontology,
            extraction_passes=2,
            enable_alignment=False,
        )
        
        # Should have called LLM twice
        assert call_count == 2
        
        # Should have found entities from both passes
        assert len(nodes) == 2
        names = [n.name for n in nodes]
        assert "Alice" in names
        assert "Acme Corp" in names
    
    @pytest.mark.asyncio
    async def test_multi_pass_deduplicates(self, mock_llm_provider, sample_ontology, sample_episodes):
        """Test that multi-pass deduplicates same entities."""
        async def mock_chat(*args, **kwargs):
            response = MagicMock()
            # Both passes return the same entity
            response.content = '''
            {
                "entities": [{"name": "Alice", "entity_type": "Person"}],
                "edges": []
            }
            '''
            return response
        
        mock_llm_provider.chat = mock_chat
        
        extractor = LLMGraphExtractor(mock_llm_provider)
        
        nodes, edges = await extractor.extract(
            scope_id="scope-1",
            episodes=sample_episodes,
            ontology=sample_ontology,
            extraction_passes=2,
            enable_alignment=False,
        )
        
        # Should deduplicate to just one
        assert len(nodes) == 1
        assert nodes[0].name == "Alice"
    
    @pytest.mark.asyncio
    async def test_empty_episodes(self, mock_llm_provider, sample_ontology):
        """Test extraction with empty episodes."""
        extractor = LLMGraphExtractor(mock_llm_provider)
        
        nodes, edges = await extractor.extract(
            scope_id="scope-1",
            episodes=[],
            ontology=sample_ontology,
        )
        
        assert nodes == []
        assert edges == []
    
    @pytest.mark.asyncio
    async def test_malformed_response(self, mock_llm_provider, sample_ontology, sample_episodes):
        """Test handling of malformed LLM response."""
        mock_response = MagicMock()
        mock_response.content = "This is not valid JSON"
        mock_llm_provider.chat.return_value = mock_response
        
        extractor = LLMGraphExtractor(mock_llm_provider)
        
        nodes, edges = await extractor.extract(
            scope_id="scope-1",
            episodes=sample_episodes,
            ontology=sample_ontology,
        )
        
        # Should return empty, not crash
        assert nodes == []
        assert edges == []
    
    @pytest.mark.asyncio
    async def test_custom_config(self, mock_llm_provider, sample_ontology, sample_episodes):
        """Test using custom configuration."""
        mock_response = MagicMock()
        mock_response.content = '{"entities": [], "edges": []}'
        mock_llm_provider.chat.return_value = mock_response
        
        config = GraphExtractionConfig(
            extraction_passes=3,
            enable_alignment=False,
        )
        extractor = LLMGraphExtractor(
            mock_llm_provider,
            default_config=config,
        )
        
        nodes, edges = await extractor.extract(
            scope_id="scope-1",
            episodes=sample_episodes,
            ontology=sample_ontology,
        )
        
        # Should have used the config (3 passes)
        assert mock_llm_provider.chat.call_count == 3

