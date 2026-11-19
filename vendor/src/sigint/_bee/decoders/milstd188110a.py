"""Cadence recovery for the two legacy MIL-STD-188-110A captures.

The BEE labels describe 1,060- and 2,760-bit periods.  The corresponding
plugins (073/B121 and 081/MexKaz) operate after waveform demodulation; 073 in
particular is a package reassembler, not an 188-110A modem.  This module keeps
that boundary explicit and provides deterministic frame slicing plus a lag
correlation measurement useful for validating acquisition and polarity.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from sigint._bee.decoders.base import DecodeArtifact, DecodeResult, Decoder
from sigint._bee.decoders.registry import register


@dataclass(frozen=True)
class Mil110Cadence:
    period: int
    frames: int
    trailing_bits: int
    lag_correlation: float


@dataclass(frozen=True)
class B121Frame:
    """One framed B121 record after asynchronous character recovery."""

    byte_offset: int
    physical_size: int
    sequence: int


@dataclass(frozen=True)
class B121Recovery:
    frames: tuple[B121Frame, ...]
    decoded: bytes
    inverted: bool
    orientation: str

    @property
    def decoded_bytes(self) -> int:
        return len(self.decoded)


@dataclass(frozen=True)
class B121PackageFragment:
    """File/package metadata carried by a B121 ``RP`` record."""

    record_offset: int
    transmission: int
    package_group: int
    package_number: int
    package_sequence: int
    frame_number: int
    fragment_size: int
    fragment_type: str
    filename: str
    file_size: int
    data: bytes


@dataclass(frozen=True)
class B121ReassembledFile:
    filename: str
    declared_size: int
    first_frame: int
    last_frame: int
    data: bytes


@dataclass(frozen=True)
class MexKazFrame:
    byte_offset: int
    sequence: int
    payload_length: int
    check_a: int
    check_b: int
    payload: bytes

    @property
    def payload_sum_valid(self) -> bool:
        """Field +12 is the unsigned sum of all payload octets."""
        return self.check_a == (sum(self.payload) & 0xFFFFFFFF)

    @property
    def header_sum_valid(self) -> bool:
        """Field +16 is the unsigned sum of the preceding header octets."""
        header = (MEXKAZ_MARKER
                  + self.sequence.to_bytes(4, "little")
                  + self.payload_length.to_bytes(4, "little")
                  + self.check_a.to_bytes(4, "little"))
        return self.check_b == (sum(header) & 0xFFFFFFFF)

    @property
    def integrity_valid(self) -> bool:
        return self.payload_sum_valid and self.header_sum_valid


@dataclass(frozen=True)
class MexKazRecovery:
    frames: tuple[MexKazFrame, ...]
    decoded: bytes
    inverted: bool
    orientation: str


@dataclass(frozen=True)
class TibmAttachment:
    name: str
    declared_size: int
    data: bytes
    complete: bool


@dataclass(frozen=True)
class TibmEnvelope:
    message_number: int
    user_properties: bytes
    importance: bytes
    body: bytes
    html_body: bytes
    attachments: tuple[TibmAttachment, ...]


MEXKAZ_MARKER = b"\x40\xCC\xCC\xCC"


def _reverse_each_byte(bits: np.ndarray) -> np.ndarray:
    complete = len(bits) // 8 * 8
    result = bits.copy()
    if complete:
        result[:complete] = bits[:complete].reshape(-1, 8)[:, ::-1].reshape(-1)
    return result


def decode_b121_async(bits: np.ndarray) -> bytes:
    """Recover the 8N1-like byte stream used by original ``073.dll``.

    This follows ``B121.dll``'s start/stop scan, including its paired-character
    recovery and ten-bad-bit fallback. Data bits are transmitted least
    significant bit first.
    """
    stream = np.asarray(bits, dtype=np.uint8).reshape(-1) & 1
    output = bytearray()
    offset = 0
    bad_run = 0
    limit = len(stream)

    def value(start: int) -> int:
        return sum(int(stream[start + 1 + bit]) << bit for bit in range(8))

    while offset < limit - 9:
        if stream[offset]:
            while offset < limit - 9 and stream[offset]:
                offset += 1
        if offset + 9 >= limit:
            break
        if stream[offset + 9]:
            output.append(value(offset))
            offset += 9
            continue
        if offset + 20 < limit and not stream[offset + 10] and stream[offset + 19]:
            output.append(value(offset))
            offset += 10
            output.append(value(offset))
            offset += 9
            continue
        if bad_run >= 10 and offset >= 9:
            output.append(sum(int(stream[offset - 9 + bit]) << bit for bit in range(8)))
            bad_run = 0
        bad_run += 1
        offset += 1
    return bytes(output)


def find_b121_records(data: bytes) -> tuple[B121Frame, ...]:
    """Find B121's escaped 102-byte logical records.

    The original predicate accepts ``CE``, optionally collapses a doubled
    boundary ``CE CE``, skips inserted ``DB`` escape octets while counting six
    header bytes, and requires the following three logical bytes to be zero.
    A record is 102 logical bytes, extended by every physical ``DB`` escape.
    """
    records: list[B121Frame] = []
    offset = 0
    while offset < len(data) - 9:
        if data[offset] != 0xCE:
            offset += 1
            continue
        start = offset + 1 if data[offset + 1] == 0xCE else offset

        header_size = 6
        physical = 0
        while physical < header_size and start + physical < len(data):
            if data[start + physical] == 0xDB:
                header_size += 1
            physical += 1
        header_end = start + header_size
        if header_end + 3 > len(data) or data[header_end:header_end + 3] != b"\0\0\0":
            offset += 1
            continue

        record_size = 102
        physical = 0
        while physical < record_size and start + physical < len(data):
            if data[start + physical] == 0xDB:
                record_size += 1
            physical += 1
        if start + record_size > len(data):
            break
        records.append(B121Frame(start, record_size, data[start + 1]))
        offset = start + record_size
    return tuple(records)


def unescape_b121_record(data: bytes, frame: B121Frame) -> bytes:
    """Apply B121's two byte escape map: ``DB DC`` -> ``CE``, ``DB DD`` -> ``DB``."""
    physical = data[frame.byte_offset:frame.byte_offset + frame.physical_size]
    logical = bytearray()
    offset = 0
    while offset < len(physical) and len(logical) < 102:
        byte = physical[offset]
        if byte == 0xDB and offset + 1 < len(physical):
            escaped = physical[offset + 1]
            if escaped == 0xDC:
                logical.append(0xCE)
                offset += 2
                continue
            if escaped == 0xDD:
                logical.append(0xDB)
                offset += 2
                continue
        logical.append(byte)
        offset += 1
    return bytes(logical)


_FRAGMENT_TYPES = {10: "FIRST", 11: "SINGLE", 24: "MIDDLE", 25: "FINAL"}


def find_b121_package_fragments(recovery: B121Recovery) -> tuple[B121PackageFragment, ...]:
    """Order 32 records per package and extract its ``RP`` file fragment.

    Each record contributes 80 bytes at logical offsets 20..99. A complete
    package therefore contains 2,560 bytes. Its 40-byte ``RP`` header is
    followed by the file/archive bytes described by the fragment size.
    """
    decoded_records = []
    for frame in recovery.frames:
        record = unescape_b121_record(recovery.decoded, frame)
        if len(record) == 102:
            decoded_records.append((record, frame.byte_offset))

    package_numbers = [record[5] for record, _ in decoded_records]
    wrap_starts = set()
    for index in range(2, len(package_numbers) - 1):
        current = package_numbers[index]
        following = package_numbers[index + 1]
        if (current < 10 and package_numbers[index - 1] - current > 40
                and following < 10
                and package_numbers[index - 2] - following > 40):
            wrap_starts.add(index)

    temporal_groups: dict[tuple[int, int, int], list[tuple[bytes, int]]] = {}
    epoch = 0
    for index, (record, byte_offset) in enumerate(decoded_records):
        if index in wrap_starts:
            epoch += 1
        package_number = record[5]
        temporal_groups.setdefault(
            (epoch, package_number, record[2]), [],
        ).append((record, byte_offset))

    packages: dict[tuple[int, int], dict[int, tuple[bytes, int]]] = {}
    for candidates in temporal_groups.values():
        record, byte_offset = _vote_b121_copies(candidates)
        packages.setdefault((record[6], record[5]), {}).setdefault(
            record[2], (record, byte_offset),
        )

    fragments = []
    for records in packages.values():
        if not all(sequence in records for sequence in range(1, 33)):
            continue
        package = b"".join(records[sequence][0][20:100] for sequence in range(1, 33))
        if package[:2] != b"RP":
            continue
        fragment_type = _FRAGMENT_TYPES.get(package[33])
        if fragment_type is None:
            continue
        fragment_size = int.from_bytes(package[2:4], "big")
        if not 40 <= fragment_size <= len(package):
            continue
        first, first_offset = records[1]
        filename = package[4:16].decode("ascii", errors="replace").rstrip(" \0")
        fragments.append(B121PackageFragment(
            record_offset=first_offset,
            transmission=first[1],
            package_group=first[6],
            package_number=first[5],
            package_sequence=first[2],
            frame_number=package[24],
            fragment_size=fragment_size,
            fragment_type=fragment_type,
            filename=filename,
            file_size=int.from_bytes(package[16:20], "little"),
            data=package[40:fragment_size],
        ))
    return tuple(sorted(fragments, key=lambda item: (
        item.package_group, item.frame_number, item.package_number,
    )))


B121_DELIMITER = 0xCE


def _b121_record_is_framed(record: bytes) -> bool:
    """Report whether a logical record carries both frame delimiters.

    A B121 record is ``CE`` plus 100 content bytes plus a closing ``CE``; a
    content ``CE`` is escaped as ``DB DC``, so byte 101 is the only delimiter
    inside the frame.  ``073.dll``'s adjacent-copy splice reads exactly that
    byte: ``fcn.10004360`` compares ``byte [buffer + 0x65]`` with ``0xCE`` at
    0x10004452 for the first copy and again at 0x1000449b for the second,
    deleting whichever copy fails through the ``fcn.1001345a`` memmove splice.

    A bare ``0xCE`` at byte 100 means the closing delimiter arrived one
    character early, so the frame is short and everything after the damaged run
    is shifted by one byte.  The bundled 1060 capture confirms this
    structurally: 8,692 of the 9,025 records satisfying both conditions are
    followed by a record starting on the expected ``CE``, against only 14 of
    the 61 records whose byte 100 is ``CE`` and 10 of the 86 whose byte 101 is
    not.
    """
    return (len(record) > 101
            and record[101] == B121_DELIMITER
            and record[100] != B121_DELIMITER)


def _vote_b121_copies(copies: list[tuple[bytes, int]]) -> tuple[bytes, int]:
    """Apply signed per-bit voting, retaining the earliest copy on ties.

    Copies whose frame delimiters are damaged are discarded first, matching the
    original's splice branch, unless that would discard every copy.
    """
    if len(copies) == 1:
        return copies[0]
    framed = [item for item in copies if _b121_record_is_framed(item[0])]
    if framed:
        copies = framed
    if len(copies) == 1:
        return copies[0]
    width = min(len(record) for record, _ in copies)
    voted = bytearray(width)
    for byte_index in range(width):
        fallback = copies[0][0][byte_index]
        for bit in range(8):
            mask = 1 << bit
            score = sum(1 if record[byte_index] & mask else -1
                        for record, _ in copies)
            if score > 0 or (score == 0 and fallback & mask):
                voted[byte_index] |= mask
    return bytes(voted), copies[0][1]


def reassemble_b121_files(
    fragments: tuple[B121PackageFragment, ...],
) -> tuple[B121ReassembledFile, ...]:
    """Join complete FIRST/MIDDLE/FINAL or SINGLE fragment sequences."""
    output = []
    by_name: dict[str, list[B121PackageFragment]] = {}
    for fragment in fragments:
        by_name.setdefault(fragment.filename, []).append(fragment)
    for filename, candidates in by_name.items():
        candidates.sort(key=lambda item: (item.package_group, item.frame_number))
        active: list[B121PackageFragment] = []
        for fragment in candidates:
            if fragment.fragment_type == "SINGLE":
                output.append(B121ReassembledFile(
                    filename, fragment.file_size, fragment.frame_number,
                    fragment.frame_number, fragment.data[:fragment.file_size],
                ))
                active = []
            elif fragment.fragment_type == "FIRST":
                active = [fragment]
            elif fragment.fragment_type == "MIDDLE" and active:
                active.append(fragment)
            elif fragment.fragment_type == "FINAL" and active:
                active.append(fragment)
                data = b"".join(item.data for item in active)
                declared = fragment.file_size
                if all(item.file_size == declared for item in active) and len(data) >= declared:
                    output.append(B121ReassembledFile(
                        filename, declared, active[0].frame_number,
                        fragment.frame_number, data[:declared],
                    ))
                active = []
    return tuple(sorted(output, key=lambda item: (item.filename, item.first_frame)))


def find_mexkaz_frames(data: bytes) -> tuple[MexKazFrame, ...]:
    """Parse MexKaz's 20-byte header and length-delimited payload records."""
    frames = []
    offset = 0
    while True:
        offset = data.find(MEXKAZ_MARKER, offset)
        if offset < 0:
            break
        if offset + 20 <= len(data):
            payload_length = int.from_bytes(data[offset + 8:offset + 12], "little")
            end = offset + 20 + payload_length
            if payload_length <= 4096 and end <= len(data):
                frames.append(MexKazFrame(
                    byte_offset=offset,
                    sequence=int.from_bytes(data[offset + 4:offset + 8], "little"),
                    payload_length=payload_length,
                    check_a=int.from_bytes(data[offset + 12:offset + 16], "little"),
                    check_b=int.from_bytes(data[offset + 16:offset + 20], "little"),
                    payload=data[offset + 20:end],
                ))
        offset += 1
    return tuple(frames)


def recover_mexkaz(bits: np.ndarray) -> MexKazRecovery:
    """Select the bit orientation/polarity with the most MexKaz frames."""
    candidates = []
    for oriented, orientation in ((bits, "selected"),
                                  (_reverse_each_byte(bits), "byte-reversed")):
        for inverted in (False, True):
            decoded = decode_b121_async(oriented ^ int(inverted))
            candidates.append(MexKazRecovery(
                find_mexkaz_frames(decoded), decoded, inverted, orientation,
            ))
    return max(candidates, key=lambda item: (len(item.frames), -item.inverted,
                                             item.orientation == "selected"))


def assemble_mexkaz_payload(recovery: MexKazRecovery) -> bytes:
    """Assemble the first contiguous MexKaz message, matching original output."""
    messages = split_mexkaz_messages(recovery)
    return b"" if not messages else b"".join(frame.payload for frame in messages[0])


def split_mexkaz_messages(
        recovery: MexKazRecovery) -> tuple[tuple[MexKazFrame, ...], ...]:
    """Split records whenever the sequence number resets or stops increasing."""
    messages: list[tuple[MexKazFrame, ...]] = []
    current: list[MexKazFrame] = []
    for frame in recovery.frames:
        if current and frame.sequence <= current[-1].sequence:
            messages.append(tuple(current))
            current = []
        current.append(frame)
    if current:
        messages.append(tuple(current))
    return tuple(messages)


def parse_tibm_envelope(payload: bytes) -> TibmEnvelope | None:
    """Parse the length-delimited fields following a ``;*TIBM*;`` envelope."""
    if len(payload) < 12 or not payload.startswith(b";*TIBM*;"):
        return None
    properties_start = payload.find(b"<UserProperties>", 12)
    properties_marker_end = payload.find(b"<End UserProperties>", properties_start)
    if properties_start < 0 or properties_marker_end < 0:
        return None
    position = properties_marker_end + len(b"<End UserProperties>")
    if payload[position:position + 2] == b"\r\n":
        position += 2

    def line() -> bytes | None:
        nonlocal position
        end = payload.find(b"\r\n", position)
        if end < 0:
            return None
        value = payload[position:end]
        position = end + 2
        return value

    importance_line = line()
    body_line = line()
    if importance_line is None or not importance_line.startswith(b"Importance>"):
        return None
    if body_line is None or not body_line.startswith(b"Body len>"):
        return None
    try:
        body_size = int(body_line.partition(b">")[2])
    except ValueError:
        return None
    body = payload[position:position + body_size]
    if len(body) != body_size:
        return None
    position += body_size
    html_line = line()
    if html_line is None or not html_line.startswith(b"HTMLBody len>"):
        return None
    try:
        html_size = int(html_line.partition(b">")[2])
    except ValueError:
        return None
    html_body = payload[position:position + html_size]
    if len(html_body) != html_size:
        return None
    position += html_size

    attachments = []
    while position < len(payload):
        attachment_line = line()
        if attachment_line is None:
            break
        name, separator, size_text = attachment_line.rpartition(b">")
        if not separator or not name:
            break
        try:
            declared_size = int(size_text)
        except ValueError:
            break
        available = min(declared_size, len(payload) - position)
        data = payload[position:position + available]
        position += available
        attachments.append(TibmAttachment(
            name=name.decode("cp1251", errors="replace"),
            declared_size=declared_size,
            data=data,
            complete=available == declared_size,
        ))
        if available != declared_size:
            break
    properties_end = properties_marker_end + len(b"<End UserProperties>")
    return TibmEnvelope(
        message_number=int.from_bytes(payload[8:12], "little"),
        user_properties=payload[properties_start:properties_end],
        importance=importance_line.partition(b">")[2],
        body=body,
        html_body=html_body,
        attachments=tuple(attachments),
    )


def reassemble_tibm_envelopes(recovery: MexKazRecovery) -> tuple[TibmEnvelope, ...]:
    """Join type-2+ TIBM continuation chunks onto preceding type-1 envelopes."""
    payloads = [b"".join(frame.payload for frame in message)
                for message in split_mexkaz_messages(recovery)]
    output = []
    for index, payload in enumerate(payloads):
        envelope = parse_tibm_envelope(payload)
        if envelope is None or envelope.message_number != 1:
            continue
        continuation = bytearray()
        expected_type = 2
        for following in payloads[index + 1:]:
            if len(following) < 12 or not following.startswith(b";*TIBM*;"):
                continue
            fragment_type = int.from_bytes(following[8:12], "little")
            if fragment_type == 1 or fragment_type != expected_type:
                break
            continuation.extend(following[12:])
            expected_type += 1
        attachments = []
        cursor = 0
        for attachment in envelope.attachments:
            missing = max(0, attachment.declared_size - len(attachment.data))
            extra = bytes(continuation[cursor:cursor + missing])
            cursor += len(extra)
            data = attachment.data + extra
            attachments.append(TibmAttachment(
                attachment.name, attachment.declared_size, data,
                len(data) == attachment.declared_size,
            ))
        output.append(TibmEnvelope(
            envelope.message_number, envelope.user_properties, envelope.importance,
            envelope.body, envelope.html_body, tuple(attachments),
        ))
    return tuple(output)


def recover_b121(bits: np.ndarray) -> B121Recovery:
    """Select the orientation/polarity yielding the most valid B121 records."""
    candidates: list[B121Recovery] = []
    for oriented, orientation in ((bits, "selected"),
                                  (_reverse_each_byte(bits), "byte-reversed")):
        for inverted in (False, True):
            decoded = decode_b121_async(oriented ^ int(inverted))
            candidates.append(B121Recovery(
                find_b121_records(decoded), decoded, inverted, orientation,
            ))
    return max(candidates, key=lambda item: (len(item.frames), -item.inverted,
                                             item.orientation == "selected"))


def measure_cadence(bits: np.ndarray, period: int) -> Mil110Cadence:
    """Measure the named cadence without pretending to decode modem FEC.

    Correlation is the Pearson/phi coefficient between the stream and a copy
    delayed by one period.  It is polarity invariant and, unlike raw equality,
    does not mistake a long biased idle run for convincing synchronization.
    """
    stream = np.asarray(bits, dtype=np.uint8).reshape(-1) & 1
    frames, trailing = divmod(len(stream), period)
    if len(stream) <= period:
        correlation = 0.0
    else:
        left = stream[:-period].astype(np.float64)
        right = stream[period:].astype(np.float64)
        left -= left.mean()
        right -= right.mean()
        scale = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
        correlation = float(np.dot(left, right) / scale) if scale else 0.0
    return Mil110Cadence(period, frames, trailing, correlation)


class _MilStd188110ADecoder(Decoder):
    status = "partial"
    confidence = "medium"
    layer = "synchronization"
    provenance = "BEE labels, static analysis of original plugins, and four bundled captures"
    period = 0

    def decode(self, bits: np.ndarray, block=None) -> DecodeResult:
        cadence = measure_cadence(bits, self.period)
        text = (f"offset=0b period={cadence.period} frames={cadence.frames} "
                f"trailing={cadence.trailing_bits}b "
                f"lag_correlation={cadence.lag_correlation:.6f}")
        return DecodeResult(
            text=text,
            symbol_count=cadence.frames,
            error_count=0,
            notes=("Cadence/frame slicing only. The input is already-demodulated bits; "
                   "waveform acquisition, interleaving, FEC, and B121/MexKaz payload "
                   "reassembly are not yet implemented."),
        )


class MilStd188110A1060Decoder(_MilStd188110ADecoder):
    name = "MIL-STD-188-110A (1060-period)"
    description = "Recovered 1,060-bit cadence and frame slicer"
    period = 1060

    def decode(self, bits: np.ndarray, block=None) -> DecodeResult:
        cadence = measure_cadence(bits, self.period)
        b121 = recover_b121(bits)
        packages = find_b121_package_fragments(b121)
        files = reassemble_b121_files(packages)
        artifacts = tuple(
            DecodeArtifact(
                name=f"{index:03d}_{''.join(character if character.isalnum() or character in '._-' else '_' for character in item.filename)}.arj",
                data=item.data,
                media_type="application/x-arj-compressed",
                confidence="structural-no-integrity-check",
                notes=(f"B121 {item.first_frame}..{item.last_frame}; original metadata "
                       f"name={item.filename!r}, declared bytes={item.declared_size}"),
            )
            for index, item in enumerate(files)
        )
        first = b121.frames[0].byte_offset if b121.frames else None
        text = (f"offset=0b period={cadence.period} frames={cadence.frames} "
                f"trailing={cadence.trailing_bits}b "
                f"lag_correlation={cadence.lag_correlation:.6f}\n"
                f"b121_records={len(b121.frames)} decoded_bytes={b121.decoded_bytes} "
                f"first_record={first if first is not None else '-'} "
                f"polarity={'inverted' if b121.inverted else 'normal'} "
                f"orientation={b121.orientation}\n"
                f"b121_package_fragments={len(packages)} "
                f"reassembled_files={len(files)} "
                f"first_file={files[0].filename if files else (packages[0].filename if packages else '-')}")
        return DecodeResult(
            text=text,
            symbol_count=len(b121.frames) or cadence.frames,
            error_count=0,
            notes=("Recovered B121 asynchronous characters, escaped 102-byte records, "
                   "RP metadata, 32-record package ordering, and complete file/archive "
                   "fragment assembly. Integrity checks, incomplete-package recovery, "
                   "ARJ extraction, and modem interleaving/FEC remain incomplete."),
            artifacts=artifacts,
        )


class MilStd188110A2760Decoder(_MilStd188110ADecoder):
    name = "MIL-STD-188-110A (2760-period)"
    description = "Recovered 2,760-bit cadence and MexKaz record/payload ordering"
    period = 2760

    def decode(self, bits: np.ndarray, block=None) -> DecodeResult:
        cadence = measure_cadence(bits, self.period)
        mexkaz = recover_mexkaz(bits)
        payload = assemble_mexkaz_payload(mexkaz)
        messages = split_mexkaz_messages(mexkaz)
        sequences = sorted({frame.sequence for frame in mexkaz.frames})
        valid = sum(frame.integrity_valid for frame in mexkaz.frames)
        has_tibm = payload.startswith(b";*TIBM*;")
        envelope = parse_tibm_envelope(payload)
        reassembled_envelopes = reassemble_tibm_envelopes(mexkaz)
        payload_integrity = bool(messages and all(
            frame.integrity_valid for frame in messages[0]
        ))
        artifacts = []
        if payload:
            artifacts.append(DecodeArtifact(
                name="mexkaz-ordered-payload.bin",
                data=payload,
                confidence=("integrity-verified" if payload_integrity
                            else "structural-integrity-failed"),
                notes="First contiguous MexKaz message in capture order.",
            ))
        if envelope is not None:
            artifacts.append(DecodeArtifact(
                name="mexkaz-user-properties.txt",
                data=envelope.user_properties,
                media_type="text/plain; charset=windows-1251",
                confidence=("integrity-verified" if payload_integrity
                            else "structural-integrity-failed"),
                notes="TIBM user-properties section extracted from ordered sequences.",
            ))
            unique_attachments: list[TibmAttachment] = []
            seen_attachments: set[bytes] = set()
            for recovered_envelope in reassembled_envelopes:
                for attachment in recovered_envelope.attachments:
                    digest = hashlib.sha256(attachment.data).digest()
                    if digest not in seen_attachments:
                        seen_attachments.add(digest)
                        unique_attachments.append(attachment)
            for index, attachment in enumerate(unique_attachments, 1):
                safe_name = "".join(
                    character if character.isalnum() or character in "._-" else "_"
                    for character in attachment.name
                ) or "attachment.bin"
                suffix = "" if attachment.complete else ".partial"
                artifacts.append(DecodeArtifact(
                    name=f"mexkaz-attachment-{index:03d}_{safe_name}{suffix}",
                    data=attachment.data,
                    media_type=("image/gif" if attachment.data.startswith((b"GIF87a", b"GIF89a"))
                                else "image/jpeg" if attachment.data.startswith(b"\xFF\xD8\xFF")
                                else "application/octet-stream"),
                    confidence=("integrity-verified" if attachment.complete
                                else "truncated-integrity-verified-records"),
                    notes=(f"TIBM attachment {attachment.name!r}; declared bytes="
                           f"{attachment.declared_size}, recovered bytes={len(attachment.data)}."),
                ))
        text = (f"offset=0b period={cadence.period} frames={cadence.frames} "
                f"trailing={cadence.trailing_bits}b "
                f"lag_correlation={cadence.lag_correlation:.6f}\n"
                f"mexkaz_frames={len(mexkaz.frames)} unique_sequences={len(sequences)} "
                f"messages={len(messages)} integrity_valid={valid}/{len(mexkaz.frames)} "
                f"sequence_range={sequences[0] if sequences else '-'}.."
                f"{sequences[-1] if sequences else '-'} payload_bytes={len(payload)} "
                f"tibm_envelope={'yes' if has_tibm else 'no'} "
                f"attachments={len(unique_attachments) if envelope else 0} "
                f"polarity={'inverted' if mexkaz.inverted else 'normal'} "
                f"orientation={mexkaz.orientation}")
        return DecodeResult(
            text=text,
            symbol_count=len(mexkaz.frames) or cadence.frames,
            error_count=0,
            notes=("Recovered MexKaz asynchronous characters, 40 CC CC CC record "
                   "headers, payload/header byte-sum integrity fields, message resets, "
                   "sequence ordering, and TIBM payload bytes. Cross-copy selection, "
                   "attachments, and modem interleaving/FEC remain incomplete."),
            artifacts=tuple(artifacts),
        )


register(MilStd188110A1060Decoder())
register(MilStd188110A2760Decoder())
