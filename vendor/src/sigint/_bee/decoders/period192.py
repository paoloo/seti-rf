"""192-bit period synchronizer recovered from original DLL 070.

The original detection routine at RVA 0x22d0 scans for four equal 0x14 or
0xEB bytes at bit offsets 8, 200, 392 and 584, i.e. one marker per 192-bit
frame. It counts both polarities and inverts the input when 0xEB dominates.
Only this framing invariant is implemented here; the downstream record/file
recovery remains outside the proven scope.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sigint._bee.decoders.base import DecodeResult, Decoder
from sigint._bee.decoders.registry import register


@dataclass(frozen=True)
class Period192Sync:
    offset: int
    marker: int
    bit_assembly: str


def _byte_at(bits: np.ndarray, offset: int, lsb_first: bool) -> int:
    chunk = bits[offset : offset + 8]
    weights = (1 << np.arange(8)) if lsb_first else (1 << np.arange(7, -1, -1))
    return int(chunk @ weights)


def find_period192_sync(bits: np.ndarray) -> list[Period192Sync]:
    matches: list[Period192Sync] = []
    for lsb_first, assembly in ((True, "lsb"), (False, "msb")):
        for offset in range(max(0, len(bits) - 591)):
            marker = _byte_at(bits, offset + 8, lsb_first)
            if marker not in (0x14, 0xEB):
                continue
            if all(_byte_at(bits, offset + delta, lsb_first) == marker
                   for delta in (200, 392, 584)):
                matches.append(Period192Sync(offset, marker, assembly))
    return sorted(matches, key=lambda item: (item.offset, item.bit_assembly, item.marker))


class Period192Decoder(Decoder):
    name = "192 (Period)"
    description = "Recovered four-frame 192-bit period synchronizer"
    status = "partial"
    confidence = "high"
    layer = "synchronization"
    provenance = "static analysis of original Lib/070.dll RVA 0x22d0 plus bundled captures"

    def decode(self, bits: np.ndarray, block=None) -> DecodeResult:
        matches = find_period192_sync(bits)
        text = "\n".join(
            f"@{item.offset}b period=192 sync=0x{item.marker:02X} "
            f"assembly={item.bit_assembly}-first" for item in matches
        )
        positive = sum(item.marker == 0x14 for item in matches)
        negative = sum(item.marker == 0xEB for item in matches)
        return DecodeResult(
            text=text, symbol_count=len(matches), error_count=0,
            notes=(f"{len(matches)} four-frame candidate(s): positive={positive}, "
                   f"inverted={negative}. Downstream record recovery is not yet implemented."),
        )


register(Period192Decoder())
