"""
Expertise Analyzer Implementation.

Provides statistics and quality insights about expertise,
helping identify areas for improvement.

Inspired by ACE framework's BulletpointAnalyzer statistics.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from ctxforge.core.expertise import (
    Expertise,
    ExpertiseItem,
    ExpertiseSection,
    ExpertiseStats,
)


@dataclass
class QualityReport:
    """Comprehensive quality report for expertise."""
    
    expertise_id: str
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    
    # Overall metrics
    total_items: int = 0
    active_items: int = 0
    average_effectiveness: float = 0.0
    
    # Quality distribution
    high_performers: List[str] = field(default_factory=list)
    problematic: List[str] = field(default_factory=list)
    unused: List[str] = field(default_factory=list)
    stale: List[str] = field(default_factory=list)
    
    # Section analysis
    items_by_section: Dict[str, int] = field(default_factory=dict)
    effectiveness_by_section: Dict[str, float] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def health_score(self) -> float:
        """
        Calculate overall health score (0.0-1.0).
        
        Based on proportion of high performers vs problematic items.
        """
        if self.active_items == 0:
            return 0.5
        
        good = len(self.high_performers)
        bad = len(self.problematic)
        
        # Score based on ratio of good to bad items
        if good + bad == 0:
            return 0.5  # Neutral if no significant items
        
        return good / (good + bad)


class ExpertiseAnalyzer:
    """
    Provides statistics and quality insights about expertise.
    
    Analyzes expertise items to identify:
    - High-performing items (should be preserved)
    - Problematic items (candidates for removal or update)
    - Unused items (may need promotion or removal)
    - Section distribution and balance
    - Overall quality metrics
    
    Example:
        >>> analyzer = ExpertiseAnalyzer()
        >>> stats = analyzer.get_statistics(expertise)
        >>> print(f"Average effectiveness: {stats.average_effectiveness:.2%}")
        >>> 
        >>> high_performers = analyzer.identify_high_performers(expertise)
        >>> print(f"Found {len(high_performers)} high-performing items")
    """
    
    def __init__(
        self,
        high_performer_threshold: float = 0.8,
        problematic_threshold: float = 0.4,
        min_usage_for_evaluation: int = 3,
        stale_days: int = 30,
    ):
        """
        Initialize the analyzer.
        
        Args:
            high_performer_threshold: Effectiveness above this = high performer
            problematic_threshold: Effectiveness below this = problematic
            min_usage_for_evaluation: Minimum usage count to evaluate effectiveness
            stale_days: Days without updates to be considered stale
        """
        self._high_threshold = high_performer_threshold
        self._problem_threshold = problematic_threshold
        self._min_usage = min_usage_for_evaluation
        self._stale_days = stale_days
    
    def get_statistics(self, expertise: Expertise) -> ExpertiseStats:
        """
        Get comprehensive statistics about expertise.
        
        Args:
            expertise: The expertise to analyze
            
        Returns:
            ExpertiseStats with calculated values
        """
        return ExpertiseStats.from_expertise(expertise)
    
    def identify_high_performers(
        self,
        expertise: Expertise,
        threshold: Optional[float] = None,
        min_usage: Optional[int] = None,
    ) -> List[ExpertiseItem]:
        """
        Identify high-performing items.
        
        High performers have high effectiveness scores and
        sufficient usage to be statistically meaningful.
        
        Args:
            expertise: The expertise to analyze
            threshold: Effectiveness threshold (uses default if not provided)
            min_usage: Minimum usage count (uses default if not provided)
            
        Returns:
            List of high-performing items
        """
        threshold = threshold or self._high_threshold
        min_usage = min_usage if min_usage is not None else self._min_usage
        
        return [
            item for item in expertise.active_items
            if item.total_usage >= min_usage
            and item.effectiveness_score >= threshold
        ]
    
    def identify_problematic(
        self,
        expertise: Expertise,
        threshold: Optional[float] = None,
        min_usage: Optional[int] = None,
    ) -> List[ExpertiseItem]:
        """
        Identify problematic items.
        
        Problematic items have low effectiveness scores,
        indicating they may be harmful or misleading.
        
        Args:
            expertise: The expertise to analyze
            threshold: Effectiveness threshold (uses default if not provided)
            min_usage: Minimum usage count (uses default if not provided)
            
        Returns:
            List of problematic items
        """
        threshold = threshold or self._problem_threshold
        min_usage = min_usage if min_usage is not None else self._min_usage
        
        return [
            item for item in expertise.active_items
            if item.total_usage >= min_usage
            and item.effectiveness_score < threshold
        ]
    
    def identify_unused(
        self,
        expertise: Expertise,
    ) -> List[ExpertiseItem]:
        """
        Identify unused items.
        
        Unused items have never been selected during retrieval,
        which may indicate they're not relevant or too specific.
        
        Args:
            expertise: The expertise to analyze
            
        Returns:
            List of unused items
        """
        return [
            item for item in expertise.active_items
            if item.is_unused
        ]
    
    def identify_stale(
        self,
        expertise: Expertise,
        days: Optional[int] = None,
    ) -> List[ExpertiseItem]:
        """
        Identify stale items.
        
        Stale items haven't been updated in a long time,
        which may indicate they're outdated.
        
        Args:
            expertise: The expertise to analyze
            days: Number of days to consider stale
            
        Returns:
            List of stale items
        """
        days = days or self._stale_days
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        return [
            item for item in expertise.active_items
            if self._make_aware(item.updated_at) < cutoff
        ]
    
    def _make_aware(self, dt: datetime) -> datetime:
        """Ensure datetime is timezone-aware (UTC)."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    
    def get_section_analysis(
        self,
        expertise: Expertise,
    ) -> Dict[ExpertiseSection, Dict[str, Any]]:
        """
        Analyze expertise by section.
        
        Args:
            expertise: The expertise to analyze
            
        Returns:
            Dictionary with per-section analysis
        """
        analysis = {}
        
        for section in ExpertiseSection:
            items = expertise.get_items_by_section(section)
            
            if not items:
                continue
            
            # Calculate section metrics
            total_usage = sum(i.total_usage for i in items)
            total_helpful = sum(i.helpful_count for i in items)
            total_harmful = sum(i.harmful_count for i in items)
            
            if total_usage > 0:
                avg_effectiveness = total_helpful / total_usage
            else:
                avg_effectiveness = 0.5
            
            high_performers = [i for i in items if i.is_high_performing]
            problematic = [i for i in items if i.is_problematic]
            
            analysis[section] = {
                "count": len(items),
                "total_usage": total_usage,
                "total_helpful": total_helpful,
                "total_harmful": total_harmful,
                "average_effectiveness": avg_effectiveness,
                "high_performers": len(high_performers),
                "problematic": len(problematic),
            }
        
        return analysis
    
    def generate_quality_report(
        self,
        expertise: Expertise,
    ) -> QualityReport:
        """
        Generate a comprehensive quality report.
        
        Args:
            expertise: The expertise to analyze
            
        Returns:
            QualityReport with detailed analysis
        """
        stats = self.get_statistics(expertise)
        section_analysis = self.get_section_analysis(expertise)
        
        high_performers = self.identify_high_performers(expertise)
        problematic = self.identify_problematic(expertise)
        unused = self.identify_unused(expertise)
        stale = self.identify_stale(expertise)
        
        # Build section distribution
        items_by_section = {}
        effectiveness_by_section = {}
        for section, data in section_analysis.items():
            items_by_section[section.value] = data["count"]
            effectiveness_by_section[section.value] = data["average_effectiveness"]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            expertise, stats, high_performers, problematic, unused, stale
        )
        
        return QualityReport(
            expertise_id=expertise.expertise_id,
            total_items=stats.total_items,
            active_items=stats.active_items,
            average_effectiveness=stats.average_effectiveness,
            high_performers=[i.item_id for i in high_performers],
            problematic=[i.item_id for i in problematic],
            unused=[i.item_id for i in unused],
            stale=[i.item_id for i in stale],
            items_by_section=items_by_section,
            effectiveness_by_section=effectiveness_by_section,
            recommendations=recommendations,
        )
    
    def _generate_recommendations(
        self,
        expertise: Expertise,
        stats: ExpertiseStats,
        high_performers: List[ExpertiseItem],
        problematic: List[ExpertiseItem],
        unused: List[ExpertiseItem],
        stale: List[ExpertiseItem],
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Check for problematic items
        if problematic:
            if len(problematic) >= 3:
                recommendations.append(
                    f"Consider removing or revising {len(problematic)} problematic items "
                    f"with low effectiveness scores."
                )
            else:
                for item in problematic:
                    recommendations.append(
                        f"Review item '{item.item_id}' - effectiveness is only "
                        f"{item.effectiveness_score:.1%} ({item.harmful_count} harmful uses)."
                    )
        
        # Check for unused items
        if len(unused) > stats.active_items * 0.3:
            recommendations.append(
                f"{len(unused)} items ({len(unused)/stats.active_items:.0%}) have never been used. "
                "Consider revising content for better relevance or removing them."
            )
        
        # Check for stale items
        if len(stale) > stats.active_items * 0.5:
            recommendations.append(
                f"{len(stale)} items haven't been updated in {self._stale_days}+ days. "
                "Consider reviewing for outdated information."
            )
        
        # Check section balance
        section_counts = expertise.items_by_section if hasattr(expertise, 'items_by_section') else {}
        if not section_counts:
            section_counts = {}
            for item in expertise.active_items:
                key = item.section.value
                section_counts[key] = section_counts.get(key, 0) + 1
        
        # Check if any section is empty
        for section in [ExpertiseSection.STRATEGIES, ExpertiseSection.COMMON_MISTAKES]:
            if section.value not in section_counts or section_counts.get(section.value, 0) == 0:
                recommendations.append(
                    f"Consider adding items to the '{section.to_display_name()}' section."
                )
        
        # Check overall effectiveness
        if stats.average_effectiveness < 0.5:
            recommendations.append(
                f"Overall effectiveness ({stats.average_effectiveness:.1%}) is below 50%. "
                "Consider comprehensive review of expertise content."
            )
        
        # Token budget check
        if stats.estimated_tokens > expertise.token_budget * 0.9:
            recommendations.append(
                f"Token usage ({stats.estimated_tokens}) is near budget ({expertise.token_budget}). "
                "Consider consolidating or removing low-value items."
            )
        
        return recommendations
    
    def get_effectiveness_trend(
        self,
        expertise: Expertise,
        recent_items: int = 10,
    ) -> Tuple[float, float]:
        """
        Compare effectiveness of recent vs older items.
        
        Args:
            expertise: The expertise to analyze
            recent_items: Number of items to consider as "recent"
            
        Returns:
            Tuple of (recent_effectiveness, older_effectiveness)
        """
        active = expertise.active_items
        if len(active) < recent_items * 2:
            # Not enough items to compare
            return (0.5, 0.5)
        
        # Sort by creation date
        sorted_items = sorted(active, key=lambda i: i.created_at, reverse=True)
        
        recent = sorted_items[:recent_items]
        older = sorted_items[recent_items:]
        
        # Calculate effectiveness for each group
        def avg_effectiveness(items: List[ExpertiseItem]) -> float:
            used = [i for i in items if i.total_usage > 0]
            if not used:
                return 0.5
            return sum(i.effectiveness_score for i in used) / len(used)
        
        return (avg_effectiveness(recent), avg_effectiveness(older))
    
    def suggest_items_to_remove(
        self,
        expertise: Expertise,
        target_count: Optional[int] = None,
    ) -> List[ExpertiseItem]:
        """
        Suggest items that could be removed.
        
        Prioritizes:
        1. Highly problematic items
        2. Unused items
        3. Stale items with low effectiveness
        
        Args:
            expertise: The expertise to analyze
            target_count: Number of items to suggest (None = all candidates)
            
        Returns:
            List of items suggested for removal
        """
        candidates = []
        
        # First: highly problematic (effectiveness < 0.3)
        for item in expertise.active_items:
            if item.total_usage >= self._min_usage and item.effectiveness_score < 0.3:
                candidates.append((item, 3))  # Priority 3 (highest)
        
        # Second: problematic items
        problematic = self.identify_problematic(expertise)
        for item in problematic:
            if item not in [c[0] for c in candidates]:
                candidates.append((item, 2))
        
        # Third: unused items
        unused = self.identify_unused(expertise)
        for item in unused:
            if item not in [c[0] for c in candidates]:
                candidates.append((item, 1))
        
        # Sort by priority (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        items = [c[0] for c in candidates]
        
        if target_count is not None:
            return items[:target_count]
        
        return items
    
    def format_summary(self, expertise: Expertise) -> str:
        """
        Generate a human-readable summary.
        
        Args:
            expertise: The expertise to analyze
            
        Returns:
            Formatted summary string
        """
        stats = self.get_statistics(expertise)
        report = self.generate_quality_report(expertise)
        
        lines = [
            f"=== Expertise Analysis: {expertise.name} ===",
            "",
            f"Items: {stats.active_items} active / {stats.total_items} total",
            f"Effectiveness: {stats.average_effectiveness:.1%}",
            f"Health Score: {report.health_score:.1%}",
            f"Token Usage: {stats.estimated_tokens} / {expertise.token_budget}",
            "",
            "Quality Breakdown:",
            f"  - High Performers: {len(report.high_performers)}",
            f"  - Problematic: {len(report.problematic)}",
            f"  - Unused: {len(report.unused)}",
            f"  - Stale: {len(report.stale)}",
        ]
        
        if report.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in report.recommendations:
                lines.append(f"  • {rec}")
        
        return "\n".join(lines)

