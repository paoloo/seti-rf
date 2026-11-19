"""Brazil1016 frame synchronization recovered from original ``Lib/083.dll``.

The legacy DatronLINK plugin is internally named ``Brazil1016.dll``.  Its
detector treats a frame as 1,016 data bits; when the two-byte end marker
``FF FE`` follows a frame, consecutive frame starts recur every 1,032 bits
instead.  Bytes are packed MSB-first out of the bit array (unlike the
LSB-first convention used by ``082.dll`` for Ital HDLC).

Frame walking (instruction-level static analysis of ``fcn.100042f0``,
``fcn.10004bd0`` and ``fcn.10004630`` in ``083.dll``; the GLM-5.2 triage of
``fcn.10004630`` logged under ``tools/static_triage/log/`` was corrected and
extended by direct reading).  The original runs two passes over the bit
array -- the second after inverting it in place at ``0x10004490`` -- and each
pass is exactly:

``fcn.10004bd0`` slides one bit at a time until a *candidate* frame start is
found, requiring header byte 0 (bits 0..7) and header byte 2 (bits 16..23) to
be 1 or 2 and header byte 5 (bits 40..47) to be 0.

``fcn.10004630`` then validates the candidate against the *next* frame's
header.  If bytes 1,016 and 1,024 are exactly ``FF`` and ``FE`` it accepts
immediately, advances 1,032 bits and returns.  Otherwise it scores six
relationships -- next header byte 0 in {1,2} (offset 1,016), next header
byte 2 in {1,2} (offset 1,032), next header byte 5 zero (offset 1,056), the
low nibble of header byte 4 unchanged (offsets 32 and 1,048), and this
frame's header bytes 1 and 3 (offsets 8 and 24) each matching the next
frame's header byte 1 or byte 3 (offsets 1,024 and 1,040) -- rejecting
outright when neither of the first two holds, and accepting on a total of at
least 4.  A scored accept advances only 1,016 bits and keeps looping without
re-running the candidate scan; a failure returns to the candidate scan one
bit further on.

Against the real bundled ``DatronLink.bin`` this separates the four
byte-orientation/polarity combinations decisively: ``byte-reversed``/normal
yields 170 frames covering 61% of the capture, against 6 frames (2%) for
``selected``/normal and none at all for either inverted polarity.  Those 170
frames all sit at the same bit phase (1 modulo 8), 140 of the 169 gaps
between them are exactly 1,016 bits, and 168 of them carry a literal six-byte
header ``01 [62|66] 01 [62|66] ss 00`` -- the swapped byte-1/byte-3 pair being
exactly why the original scores "matches byte 1 *or* byte 3".  Header byte 7
counts down independently of anything the DLL inspects, and it is 0 for all
ten frames the DLL closes with the exact ``FF FE`` marker and non-zero for all
160 it does not, which is the reproducible cross-check behind
:func:`find_brazil1016_frames`.

The original itself confirms this.  Its report pane
(``tests/fixtures/oracle/batch-datron-20260814/datronlink-report-y214/06-report-top.png``)
prints ``- positive - 63% (889 bytes)`` and ``- negative - Protocol is not
detected``.  Percent and bytes come from the same accumulator, so 889 bytes
fixes the accepted payload at 7,112 bits -- exactly seven 1,016-bit frames --
and 63% fixes the bit count it was handed at 11,113 to 11,288.  This walk
reproduces seven frames, 7,112 bits, 889 bytes and 63% for every bit count in
that range, and finds nothing under either inverted polarity.

Frame *payload* content (the bytes past the eight-byte header) and the
reconstruction of the three bundled historical ``.TXS`` outputs remain
proprietary and are intentionally not interpreted here.  The ``.TXS`` files
are message-transfer logs (``F 727874.MSG\\545`` style filename/size lines)
whose numbers appear nowhere in the capture, so they cannot validate payload
bytes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sigint._bee.decoders.base import DecodeResult, Decoder
from sigint._bee.decoders.registry import register

PAYLOAD_BITS = 1016
FRAME_BITS = 1032
MARKER = (0xFF, 0xFE)

# Header byte offsets, in bits, relative to a frame start.
_HEADER_BITS = (0, 8, 16, 24, 32, 40, 48, 56)
# fcn.10004bd0 needs this much room past a candidate, fcn.10004630 this much
# past a frame start (0x40 and 0x438 in the original).
_CANDIDATE_MARGIN = 64
_VALIDATE_MARGIN = 1080
_START_CODES = (1, 2)
_SCORE_THRESHOLD = 4


@dataclass(frozen=True)
class Brazil1016Train:
    offset: int
    frames: int
    inverted: bool
    orientation: str


@dataclass(frozen=True)
class Brazil1016Frame:
    offset: int
    length_bits: int
    terminal: bool
    inverted: bool
    orientation: str
    header: tuple[int, ...]


def _reverse_each_byte(bits: np.ndarray) -> np.ndarray:
    complete = len(bits) // 8 * 8
    result = bits.copy()
    if complete:
        result[:complete] = bits[:complete].reshape(-1, 8)[:, ::-1].reshape(-1)
    return result


def _byte_windows(bits: np.ndarray) -> np.ndarray:
    """MSB-first byte value starting at every bit position."""
    if len(bits) < 8:
        return np.empty(0, dtype=np.uint16)
    return np.correlate(bits.astype(np.uint16), 1 << np.arange(7, -1, -1), "valid")


def _marker_offsets(bits: np.ndarray) -> list[int]:
    values = _byte_windows(bits)
    if len(values) < 9:
        return []
    return np.flatnonzero(
        (values[:-8] == MARKER[0]) & (values[8:] == MARKER[1])
    ).astype(int).tolist()


def _trains(markers: list[int], inverted: bool, orientation: str) -> list[Brazil1016Train]:
    marker_set = set(markers)
    result = []
    for marker in markers:
        if marker - FRAME_BITS in marker_set:
            continue
        frames = 1
        while marker + frames * FRAME_BITS in marker_set:
            frames += 1
        if frames >= 2 and marker >= PAYLOAD_BITS:
            result.append(Brazil1016Train(
                offset=marker - PAYLOAD_BITS,
                frames=frames,
                inverted=inverted,
                orientation=orientation,
            ))
    return result


def find_brazil1016_trains(bits: np.ndarray) -> list[Brazil1016Train]:
    """Report runs of exact ``FF FE`` markers spaced one frame apart.

    This is the older, marker-only view.  :func:`find_brazil1016_frames`
    reproduces what the original DLL actually does.
    """
    trains = []
    for oriented, orientation in (
        (bits, "selected"),
        (_reverse_each_byte(bits), "byte-reversed"),
    ):
        trains.extend(_trains(_marker_offsets(oriented), False, orientation))
        trains.extend(_trains(_marker_offsets(oriented ^ 1), True, orientation))
    return sorted(set(trains), key=lambda item: (item.offset, item.orientation, item.inverted))


def _candidate_mask(values: np.ndarray) -> np.ndarray:
    """Positions accepted by ``fcn.10004bd0``'s one-bit-at-a-time scan."""
    count = len(values) - 40
    if count <= 0:
        return np.zeros(0, dtype=bool)
    first = values[:count]
    third = values[16 : 16 + count]
    sixth = values[40 : 40 + count]
    return (np.isin(first, _START_CODES) & np.isin(third, _START_CODES) & (sixth == 0))


def _walk_brazil1016_frames(
    oriented: np.ndarray, inverted: bool, orientation: str,
) -> list[Brazil1016Frame]:
    """Reproduce one original pass: candidate scan, then scored validation."""
    total = len(oriented)
    values = _byte_windows(oriented)
    candidates = np.flatnonzero(_candidate_mask(values))
    limit = len(values)

    def byte_at(index: int) -> int:
        return int(values[index]) if 0 <= index < limit else 0

    frames: list[Brazil1016Frame] = []
    position = 0
    while True:
        index = int(np.searchsorted(candidates, position))
        if index >= len(candidates):
            break
        position = int(candidates[index])
        if position + _CANDIDATE_MARGIN >= total:
            break
        accepted = 0
        while position + _VALIDATE_MARGIN < total:
            after = byte_at(position + PAYLOAD_BITS)
            following = byte_at(position + PAYLOAD_BITS + 8)
            header = tuple(byte_at(position + offset) for offset in _HEADER_BITS)
            if (after, following) == MARKER:
                frames.append(Brazil1016Frame(
                    position, FRAME_BITS, True, inverted, orientation, header,
                ))
                position += FRAME_BITS
                accepted += 1
                break  # the original returns immediately on the exact marker
            score = 1 if after in _START_CODES else 0
            if byte_at(position + FRAME_BITS) in _START_CODES:
                score += 1
            if score == 0:
                break
            if byte_at(position + 1056) == 0:
                score += 1
            if (byte_at(position + 1048) ^ byte_at(position + 32)) & 0x0F == 0:
                score += 1
            paired = (following, byte_at(position + 1040))
            if header[1] in paired:
                score += 1
            if header[3] in paired:
                score += 1
            if score < _SCORE_THRESHOLD:
                break
            frames.append(Brazil1016Frame(
                position, PAYLOAD_BITS, False, inverted, orientation, header,
            ))
            position += PAYLOAD_BITS
            accepted += 1
        if not accepted:
            position += 1
    return frames


def find_brazil1016_frames(bits: np.ndarray) -> list[Brazil1016Frame]:
    """Walk every orientation/polarity the way ``083.dll`` walks the stream."""
    results: list[Brazil1016Frame] = []
    for oriented, orientation in (
        (bits, "selected"),
        (_reverse_each_byte(bits), "byte-reversed"),
    ):
        for inverted in (False, True):
            candidate = oriented ^ int(inverted)
            results.extend(_walk_brazil1016_frames(candidate, inverted, orientation))
    return results


def brazil1016_coverage(frames: list[Brazil1016Frame]) -> dict[tuple[str, bool], int]:
    """Payload bits accepted per orientation/polarity, the original's own metric."""
    coverage: dict[tuple[str, bool], int] = {}
    for frame in frames:
        key = (frame.orientation, frame.inverted)
        coverage[key] = coverage.get(key, 0) + PAYLOAD_BITS
    return coverage


class DatronLinkDecoder(Decoder):
    name = "DatronLINK"
    description = "Recovered Brazil1016 candidate-scan and scored frame walker"
    status = "partial"
    confidence = "high"
    layer = "frame synchronization"
    provenance = ("static analysis of fcn.100042f0/10004bd0/10004630 in original Lib/083.dll "
                  "plus the bundled DatronLink.bin (170 frames, 61% coverage, "
                  "10/10 agreement between the FF FE marker and the header countdown)")

    def decode(self, bits: np.ndarray, block=None) -> DecodeResult:
        frames = find_brazil1016_frames(bits)
        coverage = brazil1016_coverage(frames)
        text = "\n".join(
            f"@{item.offset}b len={item.length_bits}b "
            f"{'terminal' if item.terminal else 'continuing'} "
            f"header={' '.join(f'{value:02X}' for value in item.header)} "
            f"polarity={'inverted' if item.inverted else 'normal'} "
            f"orientation={item.orientation}"
            for item in frames
        )
        summary = ", ".join(
            f"{orientation}/{'inverted' if inverted else 'normal'}="
            f"{accepted * 100 // len(bits) if len(bits) else 0}%"
            for (orientation, inverted), accepted in sorted(coverage.items())
        )
        terminal = sum(1 for item in frames if item.terminal)
        return DecodeResult(
            text=text,
            symbol_count=len(frames),
            error_count=0,
            notes=(f"{len(frames)} Brazil1016 frame(s) recovered by the original's candidate "
                   f"scan and 4-of-6 header score, {terminal} closed by the exact FF FE marker"
                   + (f" (coverage {summary})" if summary else "")
                   + ". Frame boundaries and the eight-byte header are recovered; payload "
                     "content and the historical .TXS outputs are not yet interpreted."),
        )


register(DatronLinkDecoder())
