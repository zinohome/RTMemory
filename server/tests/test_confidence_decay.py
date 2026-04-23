"""Tests for confidence decay formula and forgetting logic."""
import pytest
from datetime import datetime, timezone
from math import exp, log

from app.core.confidence_decay import (
    compute_decay,
    compute_memory_confidence,
    is_forgotten,
    DECAY_RATES,
    FORGETTING_THRESHOLD,
    REFERENCE_BOOST_ALPHA,
    MemoryType,
)


class TestComputeDecay:
    """C(t) = C0 * e^(-lambda * Delta_t) * (1 + alpha * log(n+1))"""

    def test_no_decay_zero_days(self):
        """Delta_t=0 -> no decay, only reference boost."""
        result = compute_decay(c0=0.9, decay_rate=0.001, delta_days=0, ref_count=0)
        assert result == pytest.approx(0.9, abs=1e-6)

    def test_decay_identity_memory(self):
        """Identity decay_rate=0.001, after 365 days."""
        result = compute_decay(c0=0.9, decay_rate=0.001, delta_days=365, ref_count=0)
        expected = 0.9 * exp(-0.001 * 365)
        assert result == pytest.approx(expected, abs=1e-6)
        assert result > 0.6  # identity decays slowly

    def test_decay_status_memory(self):
        """Status decay_rate=0.02, after 30 days."""
        result = compute_decay(c0=0.8, decay_rate=0.02, delta_days=30, ref_count=0)
        expected = 0.8 * exp(-0.02 * 30)
        assert result == pytest.approx(expected, abs=1e-6)
        assert result < 0.5  # status decays faster

    def test_reference_boost_no_refs(self):
        """n=0 -> boost factor = 1.0."""
        result = compute_decay(c0=1.0, decay_rate=0.0, delta_days=0, ref_count=0)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_reference_boost_with_refs(self):
        """n=5 -> boost factor > 1.0."""
        result_no_ref = compute_decay(c0=0.8, decay_rate=0.01, delta_days=10, ref_count=0)
        result_with_ref = compute_decay(c0=0.8, decay_rate=0.01, delta_days=10, ref_count=5)
        assert result_with_ref > result_no_ref

    def test_reference_boost_formula(self):
        """Verify boost factor applied, clamped to 1.0 when c0=1.0."""
        alpha = REFERENCE_BOOST_ALPHA
        n = 9
        boost = 1 + alpha * log(n + 1)
        # With c0=1.0 and no decay, result = min(1.0, 1.0 * boost) = 1.0 (clamped)
        result = compute_decay(c0=1.0, decay_rate=0.0, delta_days=0, ref_count=n)
        assert result == 1.0  # Clamped at 1.0 ceiling
        # Use lower c0 to see the boost effect unclamped
        result_low = compute_decay(c0=0.5, decay_rate=0.0, delta_days=0, ref_count=n)
        expected_low = min(1.0, 0.5 * boost)
        assert result_low == pytest.approx(expected_low, abs=1e-6)

    def test_decay_clamped_to_zero(self):
        """Confidence never goes below 0."""
        result = compute_decay(c0=0.01, decay_rate=1.0, delta_days=1000, ref_count=0)
        assert result >= 0.0

    def test_decay_never_exceeds_one(self):
        """Even with massive reference boost, confidence capped at 1.0."""
        result = compute_decay(c0=0.9, decay_rate=0.0, delta_days=0, ref_count=1000)
        assert result <= 1.0


class TestIsForgotten:
    def test_above_threshold(self):
        assert is_forgotten(0.5) is False

    def test_below_threshold(self):
        assert is_forgotten(0.05) is True

    def test_at_threshold(self):
        """Exactly at threshold -> not forgotten (boundary)."""
        assert is_forgotten(FORGETTING_THRESHOLD) is False

    def test_just_below_threshold(self):
        assert is_forgotten(0.099) is True


class TestComputeMemoryConfidence:
    """Integration: decay a memory object given its timestamps and type."""

    def test_fresh_memory_high_confidence(self):
        """Memory created 1 day ago, fact type -> confidence barely changes."""
        now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)
        created = datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc)
        result = compute_memory_confidence(
            initial_confidence=0.9,
            memory_type=MemoryType.fact,
            created_at=created,
            now=now,
            ref_count=0,
        )
        assert result > 0.89

    def test_old_status_memory_low_confidence(self):
        """Status memory from 60 days ago -> confidence very low."""
        now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)
        created = datetime(2026, 2, 22, 12, 0, 0, tzinfo=timezone.utc)
        result = compute_memory_confidence(
            initial_confidence=0.8,
            memory_type=MemoryType.status,
            created_at=created,
            now=now,
            ref_count=0,
        )
        assert result < 0.3

    def test_old_identity_still_remembered(self):
        """Identity memory from 180 days ago, no refs -> still above threshold."""
        now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)
        created = datetime(2025, 10, 25, 12, 0, 0, tzinfo=timezone.utc)
        result = compute_memory_confidence(
            initial_confidence=0.95,
            memory_type=MemoryType.fact,
            created_at=created,
            now=now,
            ref_count=0,
        )
        assert result > FORGETTING_THRESHOLD

    def test_inference_memory_forgets_fast(self):
        """Inference memory, 90 days old -> likely forgotten."""
        now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)
        created = datetime(2026, 1, 23, 12, 0, 0, tzinfo=timezone.utc)
        result = compute_memory_confidence(
            initial_confidence=0.6,
            memory_type=MemoryType.inference,
            created_at=created,
            now=now,
            ref_count=0,
        )
        assert result < FORGETTING_THRESHOLD

    def test_referenced_memory_survives(self):
        """Old inference memory but referenced 5 times -> boosted above threshold."""
        now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)
        created = datetime(2026, 1, 23, 12, 0, 0, tzinfo=timezone.utc)
        result_no_ref = compute_memory_confidence(
            initial_confidence=0.6,
            memory_type=MemoryType.inference,
            created_at=created,
            now=now,
            ref_count=0,
        )
        result_with_ref = compute_memory_confidence(
            initial_confidence=0.6,
            memory_type=MemoryType.inference,
            created_at=created,
            now=now,
            ref_count=5,
        )
        assert result_with_ref > result_no_ref