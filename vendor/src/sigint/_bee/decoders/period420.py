"""420-bit period synchronizer recovered from original DLL 071.

The legacy detector at RVA 0x3760 recognizes a six-byte sync sequence
``05 0C DF FA F3 20`` whose bytes begin ten bits apart. Five of six bytes
must agree. It tests normal/inverted polarity and then counts recurrences at
336- and 420-bit spacing; the bundled sample overwhelmingly uses 420 bits.
This implementation reports sync trains only, not downstream file recovery.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sigint._bee.decoders.base import DecodeResult, Decoder
from sigint._bee.decoders.registry import register

SYNC_BYTES = np.array([0x05, 0x0C, 0xDF, 0xFA, 0xF3, 0x20], dtype=np.int16)


@dataclass(frozen=True)
class Period420Train:
    offset: int
    frames: int
    inverted: bool
    orientation: str


def _reverse_each_byte(bits: np.ndarray) -> np.ndarray:
    complete = len(bits) // 8 * 8
    reversed_bits = bits.copy()
    if complete:
        reversed_bits[:complete] = bits[:complete].reshape(-1, 8)[:, ::-1].reshape(-1)
    return reversed_bits


def _sync_offsets(bits: np.ndarray, expected: np.ndarray) -> list[int]:
    if len(bits) < 60:
        return []
    values = np.correlate(bits.astype(np.int16), 1 << np.arange(8), "valid")
    count = len(bits) - 59
    score = np.zeros(count, dtype=np.uint8)
    for index, value in enumerate(expected):
        score += values[index * 10 : index * 10 + count] == value
    return np.flatnonzero(score >= 5).astype(int).tolist()


def _trains(offsets: list[int], inverted: bool, orientation: str) -> list[Period420Train]:
    remaining = set(offsets)
    trains = []
    for start in offsets:
        if start not in remaining or start - 420 in remaining:
            continue
        frames = 1
        remaining.discard(start)
        while start + frames * 420 in remaining:
            remaining.discard(start + frames * 420)
            frames += 1
        if frames >= 2:
            trains.append(Period420Train(start, frames, inverted, orientation))
    return trains


def find_period420_trains(bits: np.ndarray) -> list[Period420Train]:
    results = []
    for oriented, orientation in ((bits, "selected"), (_reverse_each_byte(bits), "byte-reversed")):
        results.extend(_trains(_sync_offsets(oriented, SYNC_BYTES), False, orientation))
        results.extend(_trains(_sync_offsets(oriented, 0xFF - SYNC_BYTES), True, orientation))
    return sorted(results, key=lambda item: (item.offset, item.orientation, item.inverted))


class Period420Decoder(Decoder):
    name = "420 (Period)"
    description = "Recovered error-tolerant 420-bit sync-train detector"
    status = "partial"
    confidence = "high"
    layer = "synchronization"
    provenance = "static analysis of original Lib/071.dll RVA 0x3760 plus bundled capture"

    def decode(self, bits: np.ndarray, block=None) -> DecodeResult:
        trains = find_period420_trains(bits)
        text = "\n".join(
            f"@{item.offset}b period=420 frames={item.frames} "
            f"polarity={'inverted' if item.inverted else 'normal'} "
            f"orientation={item.orientation}" for item in trains
        )
        return DecodeResult(
            text=text, symbol_count=sum(item.frames for item in trains), error_count=0,
            notes=(f"{len(trains)} repeated 420-bit sync train(s); five-of-six sync-byte "
                   "tolerance. Downstream file recovery is not yet implemented."),
        )


register(Period420Decoder())
