"""Confidence decay — exponential decay with reference boost.

Formula: C(t) = C0 * e^(-lambda * Delta_t) * (1 + alpha * log(n+1))

Where:
  C0     = initial confidence
  lambda = decay rate (per day)
  Delta_t = days since last reinforcement
  alpha  = reference boost coefficient (0.1)
  n      = number of times referenced / re-mentioned
"""
from __future__ import annotations

from datetime import datetime, timezone
from math import exp, log

from app.core.profile_models import (
    DECAY_RATES,
    FORGETTING_THRESHOLD,
    REFERENCE_BOOST_ALPHA,
    MemoryType,
)


def compute_decay(
    c0: float,
    decay_rate: float,
    delta_days: float,
    ref_count: int = 0,
) -> float:
    """Compute decayed confidence.

    C(t) = C0 * e^(-lambda * Delta_t) * (1 + alpha * log(n+1))

    Result clamped to [0.0, 1.0].
    """
    decay_factor = exp(-decay_rate * delta_days)
    boost_factor = 1.0 + REFERENCE_BOOST_ALPHA * log(ref_count + 1)
    raw = c0 * decay_factor * boost_factor
    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, raw))


def is_forgotten(confidence: float) -> bool:
    """Check if confidence has fallen below forgetting threshold."""
    return confidence < FORGETTING_THRESHOLD


def compute_memory_confidence(
    initial_confidence: float,
    memory_type: MemoryType,
    created_at: datetime,
    now: datetime,
    ref_count: int = 0,
) -> float:
    """Compute current confidence for a memory based on its type and age.

    Uses the decay_rate for the memory type to determine how fast
    confidence falls off. Computed at read time (not via background job).
    """
    delta_days = (now - created_at).total_seconds() / 86400.0
    decay_rate = DECAY_RATES[memory_type]
    return compute_decay(
        c0=initial_confidence,
        decay_rate=decay_rate,
        delta_days=delta_days,
        ref_count=ref_count,
    )