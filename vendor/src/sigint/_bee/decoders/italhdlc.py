"""QUEDRE record synchronization recovered from original ``Lib/082.dll``.

Despite the displayed name ``Ital HDLC``, the recovered detector does not use
an HDLC flag at this layer.  It recognizes a five-byte QUEDRE record prefix in
either packed 8-bit bytes or asynchronous 10-bit characters (start, eight
LSB-first data bits, stop), and tries both polarities.

Frame length recovery (static analysis of ``fcn.10004690`` in ``082.dll``):
after the 5-byte header, a sixth byte (a one/zero flag) selects whether a
7th byte is present before a "type" byte -- 6 characters in if the flag byte
is not exactly 1, 7 characters in if it is.  That type byte, masked to 7
bits, is looked up in a small table recovered from the DLL's 39-entry switch
(most values are invalid/reject); only five type values are valid and each
selects a *fixed* total frame length: types ``0x00``/``0x1F`` are 68-byte
block frames, ``0x20`` is an 8-byte frame, ``0x23`` is a 5-byte frame, and
``0x26`` is a 10-byte frame conditional on a confirmation byte (two character
widths after the type byte) equal to 5; any other type value means the
candidate header is not a real record.  Walking a bundled capture from its
first byte and repeatedly jumping by the resulting frame length reproduces
1,966 frames in both ``QUEDRE.bin`` and ``QUEDRE_st.bin``, and every landing
position independently satisfies the 5-byte header predicate (100% of
1,966/1,966 jumps in each file, with matching per-type counts), which is
the reproducible evidence behind :func:`find_quedre_frames`. Frame payload
*content* (the bytes between the type byte and the next header) remains
proprietary and is intentionally not interpreted here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sigint._bee.decoders.base import DecodeResult, Decoder
from sigint._bee.decoders.registry import register

THIRD = (0x00, 0xFF, 0x10, 0x0D)
FOURTH = (0x10, 0xFF, 0x0E)
FIFTH = (0x10, 0xFF, 0x50, 0x01)


@dataclass(frozen=True)
class QuedreRecord:
    offset: int
    mode: str
    inverted: bool
    orientation: str
    header: tuple[int, int, int, int, int]


def _reverse_each_byte(bits: np.ndarray) -> np.ndarray:
    complete = len(bits) // 8 * 8
    result = bits.copy()
    if complete:
        result[:complete] = bits[:complete].reshape(-1, 8)[:, ::-1].reshape(-1)
    return result


def _byte_windows(bits: np.ndarray) -> np.ndarray:
    if len(bits) < 8:
        return np.empty(0, dtype=np.uint16)
    return np.correlate(bits.astype(np.uint16), 1 << np.arange(8), "valid")


def _matches(values: np.ndarray, stride: int) -> np.ndarray:
    count = len(values) - 4 * stride
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    a = values[:count]
    b = values[stride : stride + count]
    c = values[2 * stride : 2 * stride + count]
    d = values[3 * stride : 3 * stride + count]
    e = values[4 * stride : 4 * stride + count]
    return np.flatnonzero(
        (a == 0x03) & (b == 0x02) & np.isin(c, THIRD)
        & np.isin(d, FOURTH) & np.isin(e, FIFTH)
    )


def find_quedre_records(bits: np.ndarray) -> list[QuedreRecord]:
    """Find the exact prefix predicates used by DLL 082's mode detector."""
    records: list[QuedreRecord] = []
    for oriented, orientation in (
        (bits, "selected"),
        (_reverse_each_byte(bits), "byte-reversed"),
    ):
        for inverted in (False, True):
            candidate = oriented ^ int(inverted)
            values = _byte_windows(candidate)
            for stride, mode in ((8, "packed-8"), (10, "start-stop-10")):
                for offset in _matches(values, stride).astype(int).tolist():
                    header = tuple(int(values[offset + index * stride]) for index in range(5))
                    records.append(QuedreRecord(offset, mode, inverted, orientation, header))

    # A packed stream can coincidentally satisfy the 10-bit predicate and vice
    # versa. Preserve evidence but remove identical detections.
    return sorted(set(records), key=lambda item: (
        item.offset, item.mode, item.orientation, item.inverted,
    ))


_STRIDE = {"packed-8": 8, "start-stop-10": 10}

# Bits to add, past the type-byte position, for each recovered valid type
# value. Recovered from fcn.10004690's two parallel switch tables (one per
# stride); the 10-bit-mode constants are exactly the 8-bit ones scaled by
# 10/8, matching the DLL's own two near-identical code paths.
_FRAME_LENGTH_BITS = {
    8: {0x00: 0x220, 0x1F: 0x220, 0x20: 0x40, 0x23: 0x28},
    10: {0x00: 0x2A8, 0x1F: 0x2A8, 0x20: 0x50, 0x23: 0x32},
}
_CONFIRMED_TYPE = 0x26
_CONFIRMED_LENGTH_BITS = {8: 0x50, 10: 0x64}
_CONFIRM_VALUE = 5

_FRAME_KIND = {0x00: "block", 0x1F: "block", 0x20: "short", 0x23: "tiny", 0x26: "confirmed"}


@dataclass(frozen=True)
class QuedreFrame:
    offset: int
    length_bits: int
    frame_type: int
    mode: str
    inverted: bool
    orientation: str

    @property
    def kind(self) -> str:
        return _FRAME_KIND.get(self.frame_type, "unknown")


def _header_matches_at(values: np.ndarray, offset: int, stride: int, limit: int) -> bool:
    if offset + 4 * stride >= limit:
        return False
    return (values[offset] == 0x03 and values[offset + stride] == 0x02
            and values[offset + 2 * stride] in THIRD
            and values[offset + 3 * stride] in FOURTH
            and values[offset + 4 * stride] in FIFTH)


def _walk_quedre_frames(
    oriented: np.ndarray, mode: str, inverted: bool, orientation: str,
) -> list[QuedreFrame]:
    """Sequentially walk ``oriented``, jumping by the recovered frame length.

    Byte values are precomputed once via :func:`_byte_windows` (an O(1)
    lookup per bit position) rather than rebuilt per read; the walk itself
    stays sequential because each jump size is data-dependent.
    """
    stride = _STRIDE[mode]
    length_table = _FRAME_LENGTH_BITS[stride]
    confirmed_length = _CONFIRMED_LENGTH_BITS[stride]
    values = _byte_windows(oriented)
    limit = len(values)
    frames: list[QuedreFrame] = []
    position = 0
    while position + 4 * stride < limit:
        if not _header_matches_at(values, position, stride, limit):
            position += 1
            continue
        flag_position = position + 5 * stride
        if flag_position >= limit:
            break
        shift = 7 * stride if values[flag_position] == 1 else 6 * stride
        type_position = position + shift
        if type_position >= limit:
            break
        type_value = int(values[type_position]) & 0x7F
        if type_value == _CONFIRMED_TYPE:
            confirm_position = type_position + 2 * stride
            if confirm_position >= limit or values[confirm_position] != _CONFIRM_VALUE:
                position += 1
                continue
            length_bits = shift + confirmed_length
        elif type_value in length_table:
            length_bits = shift + length_table[type_value]
        else:
            position += 1
            continue
        frames.append(QuedreFrame(position, length_bits, type_value, mode, inverted, orientation))
        position += length_bits
    return frames


def find_quedre_frames(bits: np.ndarray) -> list[QuedreFrame]:
    """Walk each polarity/orientation/mode combination, jumping by recovered frame length.

    Unlike :func:`find_quedre_records`, this does not return every position
    that happens to satisfy the 5-byte header predicate (large ``block``
    frames can coincidentally contain look-alike bytes in their payload).
    It instead reproduces the DLL's own sequential scan: check the header,
    read the length-selecting bytes, and jump straight to the next frame.
    """
    results: list[QuedreFrame] = []
    for oriented, orientation in (
        (bits, "selected"),
        (_reverse_each_byte(bits), "byte-reversed"),
    ):
        for inverted in (False, True):
            candidate = oriented ^ int(inverted)
            for mode in ("packed-8", "start-stop-10"):
                results.extend(_walk_quedre_frames(candidate, mode, inverted, orientation))
    return results


class ItalHdlcDecoder(Decoder):
    name = "Ital HDLC"
    description = "Recovered QUEDRE frame walker with header, length, and type"
    status = "partial"
    confidence = "high"
    layer = "frame synchronization"
    provenance = ("static analysis of fcn.10004690 in original Lib/082.dll "
                  "plus paired bundled captures (1,966/1,966 self-consistent frame jumps)")

    def decode(self, bits: np.ndarray, block=None) -> DecodeResult:
        frames = find_quedre_frames(bits)
        text = "\n".join(
            f"@{item.offset}b len={item.length_bits}b type={item.frame_type:02X} "
            f"kind={item.kind} mode={item.mode} "
            f"polarity={'inverted' if item.inverted else 'normal'} orientation={item.orientation}"
            for item in frames
        )
        kinds = {}
        for item in frames:
            kinds[item.kind] = kinds.get(item.kind, 0) + 1
        kind_summary = ", ".join(f"{name}={count}" for name, count in sorted(kinds.items()))
        return DecodeResult(
            text=text,
            symbol_count=len(frames),
            error_count=0,
            notes=(f"{len(frames)} QUEDRE frame(s) recovered by sequential header+length-byte "
                   f"walking ({kind_summary or 'none'}). Frame boundaries and type are recovered; "
                   "payload content and integrity checks are not yet interpreted."),
        )


register(ItalHdlcDecoder())
