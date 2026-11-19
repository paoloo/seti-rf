"""LPC-10 (FS-1015) / MELPe (STANAG-4591) frame splitter (Tier C: partial spec).

Both are 2400 bps LPC-family vocoders using 54-bit frames (22.5ms of speech
per frame): confirmed publicly for FS-1015 as 41 bits of LPC/reflection
coefficients + 7 bits pitch/voicing + 5 bits energy + 1 sync bit, and for
MELPe 2400bps as the same 54-bits-per-frame size (RFC 8130 / STANAG-4591
public summaries). This module splits the bitstream into 54-bit frames and
reports the sync bit plus the raw parameter-field bits as hex.

It does *not* decode those fields into actual LPC coefficients/pitch/gain
values or synthesize audio: the per-coefficient bit sub-allocation (how the
41 LPC-family bits split across the 10 reflection coefficients/LSFs) and
MELP's distinct parameter set (mixed excitation, aperiodic flags, Fourier
magnitudes) could not be confirmed to the precision needed for a correct
implementation from public secondary sources in this session, and a
transcription error there would silently produce plausible-looking garbage.
Framing (finding the 54-bit boundaries and sync bit) is still real,
verified-format value for a bit-level analysis tool; full parameter
decoding is future work if the exact spec becomes available.

Frame alignment (this module's addition): rather than assume frames start at
bit 0 of whatever range the caller passed in, :func:`find_frame_alignment`
searches all 54 possible start offsets, in both the given bit order and its
per-byte-bit-reversed form (the same "try both orientations" approach used by
``italhdlc.py``/``datronlink.py``, since a decoder can't know which physical
bit order the caller's capture actually used), and scores each by how
consistently the last bit of every 54-bit frame agrees with the majority
value -- a real periodic sync bit should be highly consistent; a wrong
alignment should look close to a coin flip. Measured against the three
bundled real captures: ``MELPe2400.bin`` shows 98.8% consistency (5,328
frames) under the per-byte-bit-reversed orientation at offset 1, ``Melp.bin``
shows 90.3% (217 frames) at offset 50 in the same orientation, and
``lpc.bin`` only reaches 76.3% (662 frames) at offset 27 -- clearly weaker,
so it's reported but not asserted as a confirmed alignment. This is
still framing evidence only; it does not validate the sync-bit *position
being the last bit* against the DLL, only that a strong 54-bit period exists
in real captures once phase and orientation are searched for.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sigint._bee.decoders.base import DecodeResult, Decoder
from sigint._bee.decoders.registry import register

FRAME_BITS = 54
_CONFIDENT_CONSISTENCY = 0.85


def _reverse_each_byte(bits: np.ndarray) -> np.ndarray:
    complete = len(bits) // 8 * 8
    result = bits.copy()
    if complete:
        result[:complete] = bits[:complete].reshape(-1, 8)[:, ::-1].reshape(-1)
    return result


@dataclass(frozen=True)
class FrameAlignment:
    offset: int
    orientation: str
    consistency: float
    frame_count: int
    sync_value: int

    @property
    def confident(self) -> bool:
        return self.consistency >= _CONFIDENT_CONSISTENCY and self.frame_count >= 20


def find_frame_alignment(bits: np.ndarray) -> FrameAlignment:
    """Search phase offset and byte-bit-orientation for the best 54-bit period.

    Scores each (offset, orientation) by how often the frame's last bit
    agrees with its own majority value across all frames at that phase --
    the signature of a real periodic sync bit versus an arbitrary phase.
    """
    best: FrameAlignment | None = None
    for oriented, orientation in ((bits, "selected"), (_reverse_each_byte(bits), "byte-reversed")):
        for offset in range(FRAME_BITS):
            sync_bits = oriented[offset + FRAME_BITS - 1 :: FRAME_BITS]
            frame_count = len(sync_bits)
            if frame_count < 2:
                continue
            ones_ratio = float(sync_bits.mean())
            majority = 1 if ones_ratio >= 0.5 else 0
            consistency = ones_ratio if majority == 1 else 1 - ones_ratio
            candidate = FrameAlignment(offset, orientation, consistency, frame_count, majority)
            if best is None or candidate.consistency > best.consistency:
                best = candidate
    assert best is not None  # FRAME_BITS > 0 guarantees at least one candidate
    return best


class VocoderFrameDecoder(Decoder):
    name = "LPC10"
    description = "Aligns and splits 54-bit LPC-family vocoder frames"

    def decode(self, bits: np.ndarray, block=None) -> DecodeResult:
        alignment = find_frame_alignment(bits)
        oriented = bits if alignment.orientation == "selected" else _reverse_each_byte(bits)
        offset = alignment.offset
        n_frames = (len(oriented) - offset) // FRAME_BITS
        lines = []
        for i in range(n_frames):
            frame = oriented[offset + i * FRAME_BITS : offset + (i + 1) * FRAME_BITS]
            sync_bit = int(frame[-1])
            params = frame[:-1]
            value = 0
            for b in params:
                value = (value << 1) | int(b)
            lines.append(f"sync={sync_bit} params=0x{value:014X}")
        confidence_word = "confident" if alignment.confident else "weak/inconclusive"
        return DecodeResult(
            text="\n".join(lines),
            symbol_count=n_frames,
            error_count=0,
            notes=(f"frame-level split only (see module docstring for scope); alignment "
                   f"offset={offset} orientation={alignment.orientation} "
                   f"sync-bit consistency={alignment.consistency:.3f} over "
                   f"{alignment.frame_count} frames ({confidence_word})"),
        )


for _name in ("LPC10", "MELP"):
    _decoder = VocoderFrameDecoder()
    _decoder.name = _name
    register(_decoder)
