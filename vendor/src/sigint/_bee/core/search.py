"""Bit-pattern search with Hamming-distance tolerance.

Mirrors the original Find dialog: a binary pattern plus a "Max Err Bit"
tolerance (0 = exact match), searched over the full bit array.
"""
from __future__ import annotations

import numpy as np


def parse_pattern(text: str) -> np.ndarray:
    """Extract the 0/1 bits from a pattern string.

    Non-bit characters (spaces, a leading bit-count label, etc.) are ignored,
    so pasting something like ``"42 10000101..."`` just keeps the bit string.
    """
    bits = [c for c in text if c in "01"]
    if not bits:
        raise ValueError("pattern must contain at least one '0'/'1' bit")
    return np.array(bits, dtype=np.uint8)


def find_all(
    haystack: np.ndarray,
    pattern: np.ndarray,
    max_err_bits: int = 0,
    start: int = 0,
) -> list[tuple[int, int]]:
    """All positions where ``pattern`` matches within ``max_err_bits``.

    Returns a list of ``(bit_position, error_count)`` sorted by position.
    """
    n = haystack.size
    m = pattern.size
    if m == 0 or m > n or start >= n - m + 1:
        return []
    windows = np.lib.stride_tricks.sliding_window_view(haystack[start:], m)
    errors = np.count_nonzero(windows != pattern, axis=1)
    matches = np.flatnonzero(errors <= max_err_bits)
    return [(int(i) + start, int(errors[i])) for i in matches]


def find_next(
    haystack: np.ndarray,
    pattern: np.ndarray,
    max_err_bits: int = 0,
    after: int = -1,
) -> tuple[int, int] | None:
    """First match strictly after bit index ``after`` (for a "Find Next" action)."""
    start = after + 1
    if start < 0:
        start = 0
    matches = find_all(haystack, pattern, max_err_bits, start=start)
    return matches[0] if matches else None
