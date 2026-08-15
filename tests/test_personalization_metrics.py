"""Tests for the personalization effectiveness metrics service."""

import pytest

from ctxforge.core.memory import MemoryFactory
from ctxforge.engine.services.personalization_metrics_service import (
    PersonalizationMetricsService,
    SessionMetrics,
    TurnMetrics,
)
from ctxforge.extraction.integration_config import PersonalizationMetricsConfig


def test_turn_metrics_defaults():
    """TurnMetrics has sensible defaults."""
    t = TurnMetrics(turn_number=1)
    assert t.memories_retrieved == 0
    assert t.memories_used == 0
    assert t.memory_hit is False
    assert t.feedback_occurred is False
    assert t.extraction_count == 0
    assert t.preference_changes == 0


def test_session_metrics_empty():
    """SessionMetrics handles empty turn list."""
    s = SessionMetrics(session_id="s1", user_id="u1")
    assert s.memory_hit_rate == 0.0
    assert s.avg_memory_relevance == 0.0
    assert s.feedback_frequency == 0.0
    assert s.cumulative_personalization_score == []
    assert s.memory_utilization == 0.0


def test_session_metrics_calculations():
    """SessionMetrics computes aggregated metrics correctly."""
    s = SessionMetrics(
        session_id="s1",
        user_id="u1",
        turn_metrics=[
            TurnMetrics(
                turn_number=1,
                memories_retrieved=5,
                memories_used=3,
                memory_hit=True,
                memory_avg_score=0.8,
            ),
            TurnMetrics(
                turn_number=2,
                memories_retrieved=3,
                memories_used=2,
                memory_hit=False,
                memory_avg_score=0.3,
                feedback_occurred=True,
            ),
            TurnMetrics(
                turn_number=3,
                memories_retrieved=4,
                memories_used=4,
                memory_hit=True,
                memory_avg_score=0.9,
            ),
        ],
    )

    assert s.memory_hit_rate == pytest.approx(2.0 / 3.0)
    assert s.avg_memory_relevance == pytest.approx((0.8 + 0.3 + 0.9) / 3.0)
    assert s.feedback_frequency == pytest.approx(1.0 / 3.0)
    assert s.memory_utilization == pytest.approx(9.0 / 12.0)

    # Cumulative personalization score
    cps = s.cumulative_personalization_score
    assert len(cps) == 3
    assert cps[0] == pytest.approx(1.0)      # 1/1
    assert cps[1] == pytest.approx(0.5)      # 1/2
    assert cps[2] == pytest.approx(2.0 / 3)  # 2/3


def test_record_retrieval():
    """record_retrieval creates turn metrics."""
    service = PersonalizationMetricsService(
        config=PersonalizationMetricsConfig(memory_hit_threshold=0.5),
    )

    m1 = MemoryFactory.semantic_memory(user_id="u1", content="pref1")
    m1.confidence_score = 0.9
    m2 = MemoryFactory.semantic_memory(user_id="u1", content="pref2")
    m2.confidence_score = 0.3

    service.record_retrieval(
        session_id="s1",
        user_id="u1",
        memories=[m1, m2],
    )

    session = service.get_session_metrics("s1")
    assert session is not None
    assert len(session.turn_metrics) == 1
    assert session.turn_metrics[0].memories_retrieved == 2
    assert session.turn_metrics[0].memory_hit is True
    assert session.turn_metrics[0].memory_avg_score == pytest.approx(0.6)


def test_record_retrieval_no_hit():
    """record_retrieval with low-confidence memories yields no hit."""
    service = PersonalizationMetricsService(
        config=PersonalizationMetricsConfig(memory_hit_threshold=0.8),
    )

    m1 = MemoryFactory.semantic_memory(user_id="u1", content="pref1")
    m1.confidence_score = 0.3

    service.record_retrieval(
        session_id="s1",
        user_id="u1",
        memories=[m1],
    )

    session = service.get_session_metrics("s1")
    assert session.turn_metrics[0].memory_hit is False


def test_record_feedback():
    """record_feedback updates latest turn."""
    service = PersonalizationMetricsService()

    m1 = MemoryFactory.semantic_memory(user_id="u1", content="test")
    m1.confidence_score = 0.9
    service.record_retrieval(session_id="s1", user_id="u1", memories=[m1])
    service.record_feedback(session_id="s1", feedback_occurred=True)

    session = service.get_session_metrics("s1")
    assert session.turn_metrics[-1].feedback_occurred is True


def test_record_feedback_no_session():
    """record_feedback is a no-op when session doesn't exist."""
    service = PersonalizationMetricsService()
    service.record_feedback(session_id="nonexistent", feedback_occurred=True)
    assert service.get_session_metrics("nonexistent") is None


def test_record_extraction():
    """record_extraction updates extraction count on latest turn."""
    service = PersonalizationMetricsService()

    m1 = MemoryFactory.semantic_memory(user_id="u1", content="test")
    m1.confidence_score = 0.9
    service.record_retrieval(session_id="s1", user_id="u1", memories=[m1])
    service.record_extraction(
        session_id="s1", extraction_count=3, preference_changes=1,
    )

    session = service.get_session_metrics("s1")
    assert session.turn_metrics[-1].extraction_count == 3
    assert session.turn_metrics[-1].preference_changes == 1


def test_record_extraction_creates_turn_if_needed():
    """record_extraction creates a turn if none exists."""
    service = PersonalizationMetricsService()
    service.record_extraction(session_id="s1", extraction_count=2)

    session = service.get_session_metrics("s1")
    assert session is not None
    assert len(session.turn_metrics) == 1
    assert session.turn_metrics[0].extraction_count == 2


def test_get_user_metrics():
    """get_user_metrics returns all sessions for a user."""
    service = PersonalizationMetricsService()

    m1 = MemoryFactory.semantic_memory(user_id="u1", content="test")
    m1.confidence_score = 0.9

    service.record_retrieval(session_id="s1", user_id="u1", memories=[m1])
    service.record_retrieval(session_id="s2", user_id="u1", memories=[m1])
    service.record_retrieval(session_id="s3", user_id="u2", memories=[m1])

    u1_metrics = service.get_user_metrics("u1")
    assert len(u1_metrics) == 2

    u2_metrics = service.get_user_metrics("u2")
    assert len(u2_metrics) == 1


def test_to_dict():
    """to_dict exports metrics as a dictionary."""
    service = PersonalizationMetricsService()

    m1 = MemoryFactory.semantic_memory(user_id="u1", content="test")
    m1.confidence_score = 0.9
    service.record_retrieval(session_id="s1", user_id="u1", memories=[m1])

    d = service.to_dict("s1")
    assert d["session_id"] == "s1"
    assert d["user_id"] == "u1"
    assert d["turns"] == 1
    assert "memory_hit_rate" in d
    assert "avg_memory_relevance" in d
    assert "feedback_frequency" in d
    assert "memory_utilization" in d
    assert "cumulative_personalization_score" in d


def test_to_dict_missing_session():
    """to_dict returns empty dict for unknown session."""
    service = PersonalizationMetricsService()
    assert service.to_dict("nonexistent") == {}
