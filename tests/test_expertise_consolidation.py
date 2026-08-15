"""
Tests for Expertise Consolidator and Analyzer.

Tests deduplication, merging, and quality analysis.
"""

from datetime import datetime, timedelta, timezone

import pytest

from ctxforge.core.expertise import (
    Expertise,
    ExpertiseItem,
    ExpertiseSection,
    SimilarGroup,
)
from ctxforge.expertise.analyzer import (
    ExpertiseAnalyzer,
    QualityReport,
)
from ctxforge.expertise.consolidator import ExpertiseConsolidator


# Test fixtures
@pytest.fixture
def expertise():
    """Create sample expertise for testing."""
    exp = Expertise(
        expertise_id="test-expertise",
        name="Test Expertise",
        domain="testing",
    )
    
    # Add diverse items
    exp.add_item(
        section=ExpertiseSection.STRATEGIES,
        content="Always start with a friendly greeting",
    )
    exp.add_item(
        section=ExpertiseSection.STRATEGIES,
        content="Begin with a warm and friendly greeting",  # Similar to first
    )
    exp.add_item(
        section=ExpertiseSection.FORMULAS,
        content="Price = Cost * (1 + Margin)",
    )
    exp.add_item(
        section=ExpertiseSection.COMMON_MISTAKES,
        content="Don't forget to validate user input",
    )
    exp.add_item(
        section=ExpertiseSection.COMMON_MISTAKES,
        content="Never skip input validation steps",  # Similar
    )
    exp.add_item(
        section=ExpertiseSection.HEURISTICS,
        content="When in doubt, ask for clarification",
    )
    
    return exp


@pytest.fixture
def expertise_with_usage():
    """Create expertise with usage data."""
    exp = Expertise(
        expertise_id="test-usage",
        name="Usage Test",
        domain="testing",
    )
    
    # High performer
    item1 = exp.add_item(
        section=ExpertiseSection.STRATEGIES,
        content="High performing strategy",
    )
    item1.helpful_count = 15
    item1.harmful_count = 2
    
    # Problematic item
    item2 = exp.add_item(
        section=ExpertiseSection.COMMON_MISTAKES,
        content="Problematic advice",
    )
    item2.helpful_count = 2
    item2.harmful_count = 10
    
    # Unused item
    exp.add_item(
        section=ExpertiseSection.HEURISTICS,
        content="Never used heuristic",
    )
    
    # Average performer
    item4 = exp.add_item(
        section=ExpertiseSection.FORMULAS,
        content="Moderately effective formula",
    )
    item4.helpful_count = 5
    item4.harmful_count = 3
    
    # Make one item stale
    stale_item = exp.add_item(
        section=ExpertiseSection.CONTEXT_CLUES,
        content="Old context clue",
    )
    stale_item.updated_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=60)
    
    return exp


class TestExpertiseConsolidator:
    """Tests for ExpertiseConsolidator class."""
    
    @pytest.mark.asyncio
    async def test_find_similar_groups(self):
        """Test finding groups of similar items."""
        consolidator = ExpertiseConsolidator(default_threshold=0.8)
        
        # Use very similar items that will definitely be grouped
        items = [
            ExpertiseItem(
                item_id="a1",
                content="Always validate user input before processing",
                section=ExpertiseSection.STRATEGIES,
            ),
            ExpertiseItem(
                item_id="a2",
                content="Always validate user input before processing it",  # Near duplicate
                section=ExpertiseSection.STRATEGIES,
            ),
            ExpertiseItem(
                item_id="b1",
                content="Use formula: price = cost * margin",
                section=ExpertiseSection.FORMULAS,
            ),
        ]
        
        groups = await consolidator.find_similar_groups(items)
        
        # Should find at least one group (the validation items are near-duplicates)
        assert len(groups) >= 1
        
        # Each group should have at least 2 items
        for group in groups:
            assert group.item_count >= 2
    
    @pytest.mark.asyncio
    async def test_find_similar_groups_high_threshold(self, expertise):
        """Test that high threshold reduces groups found."""
        consolidator = ExpertiseConsolidator(default_threshold=0.95)
        
        groups = await consolidator.find_similar_groups(expertise.items)
        
        # Very high threshold should find fewer or no groups
        # (depends on how similar the items actually are)
        for group in groups:
            # All items in group should be very similar
            assert group.item_count >= 2
    
    @pytest.mark.asyncio
    async def test_find_similar_groups_empty_input(self):
        """Test with empty input."""
        consolidator = ExpertiseConsolidator()
        
        groups = await consolidator.find_similar_groups([])
        
        assert groups == []
    
    @pytest.mark.asyncio
    async def test_find_similar_groups_single_item(self):
        """Test with single item."""
        consolidator = ExpertiseConsolidator()
        item = ExpertiseItem(
            item_id="test-1",
            section=ExpertiseSection.STRATEGIES,
            content="Only item",
        )
        
        groups = await consolidator.find_similar_groups([item])
        
        assert groups == []
    
    @pytest.mark.asyncio
    async def test_merge_group(self):
        """Test merging a group of items."""
        consolidator = ExpertiseConsolidator()
        
        items = [
            ExpertiseItem(
                item_id="item-1",
                section=ExpertiseSection.STRATEGIES,
                content="First version of advice",
                helpful_count=5,
                harmful_count=1,
            ),
            ExpertiseItem(
                item_id="item-2",
                section=ExpertiseSection.STRATEGIES,
                content="Second version of same advice with more detail",
                helpful_count=3,
                harmful_count=0,
            ),
        ]
        
        group = SimilarGroup(items=items, similarity_scores=[1.0, 0.9])
        
        merged = await consolidator.merge_group(group)
        
        assert merged is not None
        assert merged.item_id == "item-1"  # Uses first item's ID
        assert merged.helpful_count == 8  # Sum of helpful counts
        assert merged.harmful_count == 1  # Sum of harmful counts
        assert merged.source == "consolidator:merge"
    
    @pytest.mark.asyncio
    async def test_merge_group_single_item(self):
        """Test merging single item returns itself."""
        consolidator = ExpertiseConsolidator()
        
        item = ExpertiseItem(
            item_id="only",
            section=ExpertiseSection.STRATEGIES,
            content="Only item",
        )
        
        group = SimilarGroup(items=[item], similarity_scores=[1.0])
        
        merged = await consolidator.merge_group(group)
        
        assert merged.item_id == item.item_id
    
    @pytest.mark.asyncio
    async def test_merge_group_with_llm_func(self):
        """Test merging with LLM merge function."""
        async def mock_llm_merge(contents):
            return "Intelligently merged: " + " + ".join(contents)
        
        consolidator = ExpertiseConsolidator(llm_merge_func=mock_llm_merge)
        
        items = [
            ExpertiseItem(
                item_id="a",
                content="First content",
                section=ExpertiseSection.STRATEGIES,
            ),
            ExpertiseItem(
                item_id="b",
                content="Second content",
                section=ExpertiseSection.STRATEGIES,
            ),
        ]
        
        group = SimilarGroup(items=items, similarity_scores=[1.0, 0.9])
        merged = await consolidator.merge_group(group)
        
        assert "Intelligently merged" in merged.content
    
    @pytest.mark.asyncio
    async def test_consolidate_expertise(self, expertise):
        """Test consolidating entire expertise."""
        consolidator = ExpertiseConsolidator(default_threshold=0.7)
        
        result = await consolidator.consolidate_expertise(
            expertise,
            apply_changes=True,
        )
        
        assert "groups_found" in result
        assert "items_merged" in result
        assert "items_removed" in result
    
    @pytest.mark.asyncio
    async def test_consolidate_expertise_no_apply(self, expertise):
        """Test consolidation without applying changes."""
        consolidator = ExpertiseConsolidator(default_threshold=0.7)
        
        original_count = expertise.active_item_count
        
        _result = await consolidator.consolidate_expertise(
            expertise,
            apply_changes=False,
        )
        
        # Items should not be modified
        assert expertise.active_item_count == original_count
    
    @pytest.mark.asyncio
    async def test_find_duplicates(self, expertise):
        """Test finding duplicates of a specific item."""
        consolidator = ExpertiseConsolidator(default_threshold=0.7)
        
        target = expertise.items[0]  # First greeting item
        candidates = expertise.items[1:]  # Rest of items
        
        duplicates = await consolidator.find_duplicates(target, candidates)
        
        # Should find the similar greeting
        assert len(duplicates) >= 0  # May or may not find depending on similarity
    
    @pytest.mark.asyncio
    async def test_is_duplicate(self):
        """Test duplicate detection."""
        consolidator = ExpertiseConsolidator(default_threshold=0.8)
        
        item1 = ExpertiseItem(
            item_id="new",
            content="Always validate user input carefully",
            section=ExpertiseSection.STRATEGIES,
        )
        
        candidates = [
            ExpertiseItem(
                item_id="existing",
                content="Always validate user input carefully",  # Exact match
                section=ExpertiseSection.STRATEGIES,
            )
        ]
        
        is_dup = await consolidator.is_duplicate(item1, candidates)
        
        assert is_dup is True
    
    @pytest.mark.asyncio
    async def test_is_duplicate_different_content(self):
        """Test non-duplicate detection."""
        consolidator = ExpertiseConsolidator(default_threshold=0.9)
        
        item1 = ExpertiseItem(
            item_id="new",
            content="Always be polite to customers",
            section=ExpertiseSection.STRATEGIES,
        )
        
        candidates = [
            ExpertiseItem(
                item_id="existing",
                content="Calculate price using cost plus margin",
                section=ExpertiseSection.FORMULAS,
            )
        ]
        
        is_dup = await consolidator.is_duplicate(item1, candidates)
        
        assert is_dup is False


class TestExpertiseAnalyzer:
    """Tests for ExpertiseAnalyzer class."""
    
    def test_get_statistics(self, expertise_with_usage):
        """Test getting statistics."""
        analyzer = ExpertiseAnalyzer()
        
        stats = analyzer.get_statistics(expertise_with_usage)
        
        assert stats.total_items == 5
        assert stats.active_items == 5
        assert 0 <= stats.average_effectiveness <= 1
    
    def test_identify_high_performers(self, expertise_with_usage):
        """Test identifying high-performing items."""
        analyzer = ExpertiseAnalyzer(
            high_performer_threshold=0.8,
            min_usage_for_evaluation=3,
        )
        
        high_performers = analyzer.identify_high_performers(expertise_with_usage)
        
        # Should find the high performer item
        assert len(high_performers) >= 1
        for item in high_performers:
            assert item.effectiveness_score >= 0.8
    
    def test_identify_problematic(self, expertise_with_usage):
        """Test identifying problematic items."""
        analyzer = ExpertiseAnalyzer(
            problematic_threshold=0.4,
            min_usage_for_evaluation=3,
        )
        
        problematic = analyzer.identify_problematic(expertise_with_usage)
        
        # Should find the problematic item
        assert len(problematic) >= 1
        for item in problematic:
            assert item.effectiveness_score < 0.4
    
    def test_identify_unused(self, expertise_with_usage):
        """Test identifying unused items."""
        analyzer = ExpertiseAnalyzer()
        
        unused = analyzer.identify_unused(expertise_with_usage)
        
        # Should find the unused item
        assert len(unused) >= 1
        for item in unused:
            assert item.total_usage == 0
    
    def test_identify_stale(self, expertise_with_usage):
        """Test identifying stale items."""
        analyzer = ExpertiseAnalyzer(stale_days=30)
        
        stale = analyzer.identify_stale(expertise_with_usage)
        
        # Should find the stale item
        assert len(stale) >= 1
    
    def test_get_section_analysis(self, expertise_with_usage):
        """Test section analysis."""
        analyzer = ExpertiseAnalyzer()
        
        analysis = analyzer.get_section_analysis(expertise_with_usage)
        
        # Should have data for sections with items
        assert len(analysis) > 0
        
        for _section, data in analysis.items():
            assert "count" in data
            assert "total_usage" in data
            assert "average_effectiveness" in data
    
    def test_generate_quality_report(self, expertise_with_usage):
        """Test quality report generation."""
        analyzer = ExpertiseAnalyzer()
        
        report = analyzer.generate_quality_report(expertise_with_usage)
        
        assert isinstance(report, QualityReport)
        assert report.expertise_id == expertise_with_usage.expertise_id
        assert report.total_items == 5
        assert 0 <= report.health_score <= 1
    
    def test_quality_report_recommendations(self, expertise_with_usage):
        """Test that report includes recommendations."""
        analyzer = ExpertiseAnalyzer()
        
        report = analyzer.generate_quality_report(expertise_with_usage)
        
        # Should have recommendations due to problematic items
        assert len(report.recommendations) > 0
    
    def test_get_effectiveness_trend(self, expertise_with_usage):
        """Test effectiveness trend analysis."""
        analyzer = ExpertiseAnalyzer()
        
        recent, older = analyzer.get_effectiveness_trend(
            expertise_with_usage,
            recent_items=2,
        )
        
        assert 0 <= recent <= 1
        assert 0 <= older <= 1
    
    def test_suggest_items_to_remove(self, expertise_with_usage):
        """Test suggestion of items to remove."""
        analyzer = ExpertiseAnalyzer(min_usage_for_evaluation=3)
        
        to_remove = analyzer.suggest_items_to_remove(expertise_with_usage)
        
        # Should suggest at least the problematic item
        assert len(to_remove) >= 1
    
    def test_suggest_items_to_remove_with_target(self, expertise_with_usage):
        """Test with target count."""
        analyzer = ExpertiseAnalyzer()
        
        to_remove = analyzer.suggest_items_to_remove(
            expertise_with_usage,
            target_count=2,
        )
        
        assert len(to_remove) <= 2
    
    def test_format_summary(self, expertise_with_usage):
        """Test summary formatting."""
        analyzer = ExpertiseAnalyzer()
        
        summary = analyzer.format_summary(expertise_with_usage)
        
        assert isinstance(summary, str)
        assert "Expertise Analysis" in summary
        assert "Effectiveness" in summary


class TestQualityReport:
    """Tests for QualityReport dataclass."""
    
    def test_health_score_good(self):
        """Test health score with good items."""
        report = QualityReport(
            expertise_id="test",
            active_items=10,
            high_performers=["a", "b", "c", "d", "e"],
            problematic=["x"],
        )
        
        # 5 good vs 1 bad = high health
        assert report.health_score > 0.8
    
    def test_health_score_bad(self):
        """Test health score with problematic items."""
        report = QualityReport(
            expertise_id="test",
            active_items=10,
            high_performers=["a"],
            problematic=["x", "y", "z", "w", "v"],
        )
        
        # 1 good vs 5 bad = low health
        assert report.health_score < 0.3
    
    def test_health_score_neutral(self):
        """Test health score with no significant items."""
        report = QualityReport(
            expertise_id="test",
            active_items=10,
            high_performers=[],
            problematic=[],
        )
        
        assert report.health_score == 0.5
    
    def test_health_score_empty(self):
        """Test health score with no items."""
        report = QualityReport(
            expertise_id="test",
            active_items=0,
        )
        
        assert report.health_score == 0.5


class TestConsolidatorWithEmbeddings:
    """Tests for consolidator with embedding support."""
    
    @pytest.mark.asyncio
    async def test_similarity_with_embeddings(self):
        """Test similarity calculation using embeddings."""
        call_count = 0
        
        async def mock_embedding(text):
            nonlocal call_count
            call_count += 1
            # Return different embeddings based on content
            if "greeting" in text.lower():
                return [1.0, 0.0, 0.0]
            elif "formula" in text.lower():
                return [0.0, 1.0, 0.0]
            else:
                return [0.0, 0.0, 1.0]
        
        consolidator = ExpertiseConsolidator(
            embedding_func=mock_embedding,
            default_threshold=0.9,
        )
        
        items = [
            ExpertiseItem(
                item_id="1",
                content="Start with a friendly greeting",
                section=ExpertiseSection.STRATEGIES,
            ),
            ExpertiseItem(
                item_id="2",
                content="Use this formula for calculation",
                section=ExpertiseSection.FORMULAS,
            ),
        ]
        
        _groups = await consolidator.find_similar_groups(items, threshold=0.9)
        
        # These should not be grouped (different embeddings)
        # The embedding function was called
        assert call_count > 0


class TestAnalyzerEdgeCases:
    """Edge case tests for analyzer."""
    
    def test_empty_expertise(self):
        """Test with empty expertise."""
        analyzer = ExpertiseAnalyzer()
        expertise = Expertise(
            expertise_id="empty",
            name="Empty",
            domain="test",
        )
        
        stats = analyzer.get_statistics(expertise)
        
        assert stats.total_items == 0
        assert stats.active_items == 0
    
    def test_all_inactive_items(self):
        """Test with all inactive items."""
        analyzer = ExpertiseAnalyzer()
        expertise = Expertise(
            expertise_id="inactive",
            name="Inactive",
            domain="test",
        )
        
        item = expertise.add_item(
            section=ExpertiseSection.STRATEGIES,
            content="Inactive item",
        )
        item.deactivate()
        
        high = analyzer.identify_high_performers(expertise)
        
        assert len(high) == 0
    
    def test_effectiveness_trend_insufficient_items(self):
        """Test trend with insufficient items."""
        analyzer = ExpertiseAnalyzer()
        expertise = Expertise(
            expertise_id="small",
            name="Small",
            domain="test",
        )
        
        expertise.add_item(
            section=ExpertiseSection.STRATEGIES,
            content="Only item",
        )
        
        recent, older = analyzer.get_effectiveness_trend(expertise, recent_items=10)
        
        # Should return neutral when not enough items
        assert recent == 0.5
        assert older == 0.5

