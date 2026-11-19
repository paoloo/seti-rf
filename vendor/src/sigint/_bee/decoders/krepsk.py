"""KRE-PSK synchronizer recovered from the original DLL 077.

The legacy routine at RVA 0x1ff0 assembles bytes LSB-first and scans for four
equal 0xD3 or 0x2C values at offsets 0, 128, 256 and 384 bits. This module
implements only that evidence-backed synchronization layer. Character and
message recovery performed by the following legacy routine remains unknown.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sigint._bee.decoders.base import DecodeResult, Decoder
from sigint._bee.decoders.registry import register


@dataclass(frozen=True)
class KreSync:
    offset: int
    marker: int
    bit_assembly: str


def _byte_at(bits: np.ndarray, offset: int, lsb_first: bool) -> int:
    chunk = bits[offset : offset + 8]
    weights = (1 << np.arange(8)) if lsb_first else (1 << np.arange(7, -1, -1))
    return int(chunk @ weights)


def find_kre_sync(bits: np.ndarray) -> list[KreSync]:
    """Return all four-marker candidates, including inverted polarity."""
    matches: list[KreSync] = []
    for lsb_first, assembly in ((True, "lsb"), (False, "msb")):
        for offset in range(max(0, len(bits) - 391)):
            value = _byte_at(bits, offset, lsb_first)
            if value not in (0xD3, 0x2C):
                continue
            if all(_byte_at(bits, offset + delta, lsb_first) == value
                   for delta in (128, 256, 384)):
                matches.append(KreSync(offset, value, assembly))
    return sorted(matches, key=lambda item: (item.offset, item.bit_assembly, item.marker))


class KrePskDecoder(Decoder):
    name = "KRE-PSK"
    description = "Recovered 128-bit-spaced KRE-PSK sync-train detector"
    status = "partial"
    confidence = "high"
    layer = "synchronization"
    provenance = "static analysis of original Lib/077.dll RVA 0x1ff0 plus oracle capture"

    def decode(self, bits: np.ndarray, block=None) -> DecodeResult:
        matches = find_kre_sync(bits)
        lines = [
            f"@{item.offset}b sync=0x{item.marker:02X} assembly={item.bit_assembly}-first"
            for item in matches
        ]
        return DecodeResult(
            text="\n".join(lines), symbol_count=len(matches), error_count=0,
            notes=(f"{len(matches)} four-marker KRE sync candidate(s). "
                   "Each marker is 128 bits apart; payload character recovery is not yet implemented."),
        )


register(KrePskDecoder())
