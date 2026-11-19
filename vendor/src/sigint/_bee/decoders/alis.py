"""ALIS: 111-bit transmission cycle with a repeated 8-bit frame marker.

The original application's help file (`Bee.Demo/BeeR.chm`,
`hlp_vwold_alis.htm`) documents ALIS as FSK2, 170 Hz shift, 225 bit/s, with a
"transmission cycle of 111 bits which includes a 54-bit information block and
a 57-bit reverse-reception interval", and lists it among the protocols whose
period is determinable automatically. The original DLL's internal export name
independently encodes the same two numbers: `Sig111c54.dll` (`001.dll`).

Both are confirmed against the bundled `Signals/ALIS.bin`:

* Read with the documented LSB-first bit packing (see
  `documented_period`), lags 111, 222 and 333 take the top three places in the
  whole 8-400 bit autocorrelation ranking (phi = 0.263 / 0.212 / 0.175,
  median 0.010; hypergeometric p = 1.0e-7).
* Sliced at 111 bits, the frame is not statistically flat: bit offsets 58-65
  carry the fixed 8-bit pattern `11111100` in 309 of 360 frames (85.8%), and
  offset 58 is by a wide margin the best 8-bit window in the frame (the next
  best, offset 57, only reaches 48.9% and is that same marker seen one bit
  early). Of the 544 raw occurrences of the pattern anywhere in the capture,
  309 sit at offset 58 mod 111 and the most common spacing between
  occurrences is exactly 111 bits.

The marker falls just past the documented 54-bit information block, i.e.
inside the reverse-reception interval, which is where an ARQ-style protocol's
local transmitter would idle -- consistent with, but not proof of, that
reading of the frame layout.

This module recovers frame synchronization only. The 54-bit information
block's alphabet, error correction and payload are not recovered, and no
claim is made about what the remaining reverse-interval bits carry.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sigint._bee.decoders.base import DecodeResult, Decoder
from sigint._bee.decoders.registry import register
from sigint._bee.decoders.documented_period import best_orientation_evidence, byte_reverse_bits

#: Frame period in bits, from the help file and the `Sig111c54` DLL codename.
ALIS_PERIOD = 111
#: Bit offset of the marker inside the 111-bit frame, measured on the capture.
ALIS_MARKER_OFFSET = 58
#: The marker itself, in transmitted (LSB-first-unpacked) bit order.
ALIS_MARKER = np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=np.uint8)


@dataclass(frozen=True)
class AlisTrain:
    """A run of consecutive frames whose marker is present at 111-bit spacing."""

    offset: int
    frames: int
    inverted: bool
    orientation: str


def _marker_offsets(bits: np.ndarray, pattern: np.ndarray) -> list[int]:
    if len(bits) < len(pattern):
        return []
    windows = np.lib.stride_tricks.sliding_window_view(bits, len(pattern))
    return np.flatnonzero((windows == pattern).all(axis=1)).astype(int).tolist()


def _trains(offsets: list[int], inverted: bool, orientation: str) -> list[AlisTrain]:
    remaining = set(offsets)
    trains = []
    for start in offsets:
        if start not in remaining or start - ALIS_PERIOD in remaining:
            continue
        frames = 1
        remaining.discard(start)
        while start + frames * ALIS_PERIOD in remaining:
            remaining.discard(start + frames * ALIS_PERIOD)
            frames += 1
        if frames >= 2:
            trains.append(AlisTrain(start, frames, inverted, orientation))
    return trains


def find_alis_trains(bits: np.ndarray) -> list[AlisTrain]:
    """Locate runs of the 8-bit ALIS marker recurring at the documented
    111-bit cycle, searching both byte orientations and both polarities."""
    stream = np.asarray(bits, dtype=np.uint8).reshape(-1) & 1
    results: list[AlisTrain] = []
    for oriented, orientation in ((stream, "as-given"),
                                  (byte_reverse_bits(stream), "byte-reversed")):
        results.extend(_trains(_marker_offsets(oriented, ALIS_MARKER), False, orientation))
        results.extend(_trains(_marker_offsets(oriented, 1 - ALIS_MARKER), True, orientation))
    return sorted(results, key=lambda item: (item.orientation, item.inverted, item.offset))


def best_alis_train(bits: np.ndarray) -> AlisTrain | None:
    trains = find_alis_trains(bits)
    return max(trains, key=lambda item: item.frames) if trains else None


class AlisDecoder(Decoder):
    name = "ALIS"
    description = "111-bit cycle with a repeated 8-bit frame marker"
    status = "partial"
    confidence = "medium"
    layer = "synchronization"
    provenance = ("original Bee help file (Bee.Demo/BeeR.chm, hlp_vwold_alis.htm) and "
                  "the Sig111c54.dll codename of 001.dll, both confirmed against the "
                  "bundled ALIS.bin capture")

    def decode(self, bits: np.ndarray, block=None) -> DecodeResult:
        evidence = best_orientation_evidence(bits, ALIS_PERIOD)
        trains = find_alis_trains(bits)
        lines = [f"@{item.offset}b period={ALIS_PERIOD} frames={item.frames} "
                 f"polarity={'inverted' if item.inverted else 'normal'} "
                 f"orientation={item.orientation}" for item in trains]
        if evidence is not None:
            lines.insert(0, f"period={evidence.period}b "
                            f"({'confirmed' if evidence.confirmed else 'NOT confirmed'}) "
                            f"correlation={evidence.correlation:.3f} "
                            f"multiples_top_ranked={evidence.top_ranked}/{evidence.multiples} "
                            f"p={evidence.p_value:.2e} orientation={evidence.orientation}")
        return DecodeResult(
            text="\n".join(lines),
            symbol_count=sum(item.frames for item in trains),
            error_count=0 if trains else 1,
            notes=("Frame synchronization only: the documented 111-bit cycle plus runs "
                   "of the 8-bit `11111100` marker at frame offset 58. The 54-bit "
                   "information block's alphabet, error correction and payload are not "
                   "recovered -- see bee/decoders/alis.py."),
        )


register(AlisDecoder())
