"""FED-STD-1052 (Harris Modems) recovered from original ``Lib/072.dll``.

The exported DLL name is ``520F512.dll``. Its three configured modes use sync
byte pairs ``6F/0D``, ``93/11``, and ``4B/09``. Static analysis of the original
decoder (see ``GUIDE.md``) recovered the following layers, each of which is
cross-validated against the four bundled captures:

* **Framing.** ``fcn.100042e0`` walks the loaded bit buffer, assembles a byte
  from eight consecutive bits, and accepts a candidate when that byte equals
  either sync byte of the selected mode and at least two of the three bytes at
  +520/+1040/+1560 bits equal the mode's second sync byte (``cmp dword
  [var_40h_3], 2`` at ``0x1000485b``). On acquisition it unpacks exactly 520
  bits into a 65-byte buffer (``0x100048c0``-``0x10004983``).
* **Frame CRC.** ``fcn.10002830`` computes a standard reflected CRC-32
  (polynomial ``0xEDB88320``, initial ``0xFFFFFFFF``; the table is rebuilt on
  the stack by ``fcn.10001d70``, first entry ``0x77073096``, last
  ``0x2D02EF8D``) over frame bytes 0..60, finishes with ``~crc ^ 0xC1108241``
  (``0x10002846``-``0x10002848``), and compares it against the little-endian
  dword at frame offset 61 (``cmp eax, dword [edi + 0x3d]``).
* **Single-bit repair.** ``fcn.10002860`` retries the CRC after flipping each
  of the 520 bits of the frame in turn (``0x100028a9``-``0x10002941``) and
  accepts the corrected frame if the CRC then matches.
* **Frame fields.** ``0x10004999``-``0x100049d5`` copies frame byte ``0x22``
  (package count), the 24-bit little-endian value at ``0x2b``..``0x2d``
  shifted right by three (total size, i.e. a bit count converted to bytes),
  and the 24-bit little-endian value at ``0x2e``..``0x30`` into the 32-byte
  per-frame record. The log format strings at ``0x1002e650``/``0x1002e698``/
  ``0x1002e730`` name these fields.
* **Payload.** Data frames carry 56 payload bytes at frame offset 5, addressed
  by the 24-bit little-endian byte offset at frame offset 2. 56 is the
  divisor the original logger uses to turn "transfer" into "num"
  (``0x100072d4``: multiply by ``0x24924925`` then shift, i.e. divide by 56)
  and appears literally as byte ``0x23`` of every observed header frame.

The DLL performs **no decompression of its own**. It writes the reassembled
blob as ``%.3d_%.2d_out.gz`` / ``.zip`` / ``.jpg`` (``0x1002e8e8``,
``0x1002e934``, ``0x1002e950``) and, for the GZip case, launches the bundled
helper ``Bin\\UnPack\\t520_GZ.exe`` ("GZ Unpacker v1.1R") through
``CreateProcessA`` (import at ``0x1002b264``; the path is built in
``fcn.10003260`` and the command line in ``fcn.10008e70``). ``072.dll`` imports
no zlib/inflate/deflate entry point at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sigint._bee.decoders.base import DecodeArtifact, DecodeResult, Decoder
from sigint._bee.decoders.registry import register

PERIOD_BITS = 520
FRAME_BITS = 520
FRAME_BYTES = 65
CRC_BYTES = 61
PAYLOAD_BYTES = 56
PAYLOAD_OFFSET = 5
CRC_XOROUT = 0xC1108241
CRC_POLY = 0xEDB88320

#: ``(header, data)`` sync byte per configured mode, from ``0x1000448f``,
#: ``0x1000449d``, and ``0x100044ab`` of ``fcn.100042e0``.
SYNC_MODES = ((0x6F, 0x0D), (0x93, 0x11), (0x4B, 0x09))
SYNC_BYTES = (0x6F, 0x0D, 0x93, 0x11, 0x4B, 0x09)


def _crc_table() -> list[int]:
    table = []
    for index in range(256):
        value = index
        for _ in range(8):
            value = (value >> 1) ^ (CRC_POLY if value & 1 else 0)
        table.append(value)
    return table


_TABLE = _crc_table()
_NP_TABLE = np.array(_TABLE, dtype=np.uint32)


def frame_crc(data: bytes) -> int:
    """Original ``fcn.10002830`` CRC over the first 61 bytes of a frame."""
    value = 0xFFFFFFFF
    for byte in data[:CRC_BYTES]:
        value = _TABLE[(value ^ byte) & 0xFF] ^ (value >> 8)
    return ((~value) & 0xFFFFFFFF) ^ CRC_XOROUT


def stored_crc(frame: bytes) -> int:
    return int.from_bytes(frame[CRC_BYTES:FRAME_BYTES], "little")


def check_frame(frame: bytes) -> bool:
    """True when a 65-byte frame satisfies the original CRC check."""
    return len(frame) == FRAME_BYTES and frame_crc(frame) == stored_crc(frame)


def _syndromes() -> dict[int, tuple[int, int]]:
    """Map ``computed ^ stored`` to the single bit flip that explains it.

    The CRC is linear, so flipping one bit changes the syndrome by a constant.
    This reproduces ``fcn.10002860``'s 520-trial search in constant time.
    """
    table: dict[int, tuple[int, int]] = {}
    base = bytearray(FRAME_BYTES)
    base_syndrome = frame_crc(bytes(base)) ^ stored_crc(bytes(base))
    for index in range(FRAME_BYTES):
        for bit in range(8):
            probe = bytearray(base)
            probe[index] ^= 1 << bit
            syndrome = frame_crc(bytes(probe)) ^ stored_crc(bytes(probe))
            table.setdefault(syndrome ^ base_syndrome, (index, bit))
    return table


_SYNDROMES = _syndromes()


def correct_frame(frame: bytes) -> tuple[bytes | None, int]:
    """Reproduce ``fcn.10002860``: accept, repair one bit, or reject.

    Returns the (possibly repaired) frame and the index of the flipped bit,
    ``-1`` when the frame was already exact and ``None``/``-1`` when the frame
    cannot be explained by a single bit error.
    """
    if len(frame) != FRAME_BYTES:
        return None, -1
    syndrome = frame_crc(frame) ^ stored_crc(frame)
    if syndrome == 0:
        return frame, -1
    hit = _SYNDROMES.get(syndrome)
    if hit is None:
        return None, -1
    index, bit = hit
    repaired = bytearray(frame)
    repaired[index] ^= 1 << bit
    return bytes(repaired), index * 8 + bit


@dataclass(frozen=True)
class Fed1052Frame:
    """One CRC-validated 520-bit frame."""

    offset: int
    mode: int
    kind: str
    data: bytes
    corrected_bit: int = -1
    orientation: str = "byte-reversed"

    @property
    def sync(self) -> int:
        return self.data[0]

    @property
    def counter(self) -> int:
        """Frame byte 1: a modulo-256 countdown of remaining data frames."""
        return self.data[1]

    @property
    def payload_offset(self) -> int:
        """Data frames: 24-bit little-endian byte offset at frame bytes 2..4."""
        return int.from_bytes(self.data[2:5], "little")

    @property
    def payload(self) -> bytes:
        return self.data[PAYLOAD_OFFSET:PAYLOAD_OFFSET + PAYLOAD_BYTES]

    @property
    def destination(self) -> str:
        return self.data[2:18].split(b"\x00")[0].decode("latin-1")

    @property
    def source(self) -> str:
        return self.data[18:34].split(b"\x00")[0].decode("latin-1")

    @property
    def package_count(self) -> int:
        """Header frame byte 0x22, copied to record +0x0c by ``0x10004999``."""
        return self.data[0x22]

    @property
    def payload_size(self) -> int:
        """Header frame byte 0x23; 56 in every observed capture."""
        return self.data[0x23]

    @property
    def total_size(self) -> int:
        """Header bytes 0x2b..0x2d as a bit count, shifted right by three."""
        return int.from_bytes(self.data[0x2B:0x2E], "little") >> 3

    @property
    def transfer(self) -> int:
        """Header bytes 0x2e..0x30, copied to record +0x10 by ``0x100049b0``."""
        return int.from_bytes(self.data[0x2E:0x31], "little")


@dataclass(frozen=True)
class Fed1052Package:
    """A header frame and the run of data frames it announces.

    Each data frame's countdown byte and declared payload offset move in
    lock-step: ``counter`` falls by one and ``payload_offset`` rises by 56 per
    frame, so a package is the maximal run that keeps both invariants. The
    header's byte ``0x22`` states how many data frames the package has and its
    ``transfer`` field states the byte offset the run starts at.
    """

    header: Fed1052Frame | None
    frames: tuple[Fed1052Frame, ...]

    @property
    def start(self) -> int:
        return self.frames[0].payload_offset

    @property
    def data(self) -> bytes:
        return b"".join(item.payload for item in self.frames)

    @property
    def complete(self) -> bool:
        return (self.header is not None
                and self.header.package_count == len(self.frames)
                and self.frames[-1].counter == 1)


def find_fed1052_packages(frames: list[Fed1052Frame]) -> list[Fed1052Package]:
    """Group CRC-valid frames into packages."""
    packages: list[Fed1052Package] = []
    header: Fed1052Frame | None = None
    run: list[Fed1052Frame] = []
    run_header: Fed1052Frame | None = None

    def flush() -> None:
        if run:
            packages.append(Fed1052Package(run_header, tuple(run)))

    for item in frames:
        if item.kind == "header":
            flush()
            run = []
            run_header = None
            header = item
            continue
        if run:
            previous = run[-1]
            if (item.counter == (previous.counter - 1) % 256
                    and item.payload_offset == previous.payload_offset + PAYLOAD_BYTES):
                run.append(item)
                continue
            flush()
        run = [item]
        run_header = header
        header = None
    flush()
    return packages


@dataclass(frozen=True)
class Fed1052Message:
    """Payload reassembled from the declared per-frame byte offsets."""

    mode: int
    data: bytes
    frames: int
    gaps: tuple[tuple[int, int], ...] = field(default_factory=tuple)

    @property
    def missing(self) -> int:
        return sum(length for _, length in self.gaps)


def _lsb_byte_values(bits: np.ndarray) -> np.ndarray:
    """Byte at every bit offset, least-significant bit first.

    ``fcn.100042e0`` builds each byte this way out of the application's
    one-byte-per-bit buffer (``0x100048c0``-``0x10004983``: bit ``k`` of the
    window contributes ``1 << k``).
    """
    count = len(bits) - 7
    if count <= 0:
        return np.zeros(0, dtype=np.uint8)
    values = np.zeros(count, dtype=np.uint8)
    for index in range(8):
        values |= bits[index:count + index].astype(np.uint8) << index
    return values


def _scan_values(values: np.ndarray, orientation: str, correct: bool,
                 modes) -> list[Fed1052Frame]:
    limit = len(values) - FRAME_BITS
    if limit <= 0:
        return []
    columns = np.arange(FRAME_BYTES) * 8
    found: list[Fed1052Frame] = []
    head = values[:limit]
    for index in modes:
        header_sync, data_sync = SYNC_MODES[index]
        candidates = np.flatnonzero((head == header_sync) | (head == data_sync))
        if not len(candidates):
            continue
        crc = np.full(len(candidates), 0xFFFFFFFF, dtype=np.uint32)
        for column in range(CRC_BYTES):
            crc = _NP_TABLE[(crc ^ values[candidates + column * 8]) & 0xFF] ^ (crc >> 8)
        crc = (~crc) ^ np.uint32(CRC_XOROUT)
        stored = np.zeros(len(candidates), dtype=np.uint32)
        for byte in range(4):
            stored |= (values[candidates + (CRC_BYTES + byte) * 8]
                       .astype(np.uint32) << (8 * byte))
        syndrome = crc ^ stored
        keep = syndrome == 0
        if correct:
            for value in np.unique(syndrome[~keep]):
                if int(value) in _SYNDROMES:
                    keep |= syndrome == value
        for offset in candidates[keep]:
            raw = bytes(values[offset + columns])
            frame, bit = correct_frame(raw) if correct else (raw, -1)
            if frame is None:
                continue
            kind = "header" if frame[0] == header_sync else "data"
            found.append(Fed1052Frame(int(offset), index, kind, frame, bit,
                                      orientation))
    found.sort(key=lambda item: (item.offset, item.mode))
    return found


def find_fed1052_frames(bits: np.ndarray, *, correct: bool = True,
                        mode: int | None = None) -> list[Fed1052Frame]:
    """Every CRC-validated 65-byte frame in ``bits``.

    A candidate is taken wherever the byte equals one of the configured sync
    bytes, matching the original acquisition test, and is kept only when the
    original CRC accepts it, optionally after the single-bit repair that
    ``fcn.10002860`` performs.

    The original DLL reads bytes least-significant bit first out of a buffer
    the host application filled least-significant bit first, so a capture
    loaded with this project's default ``msb`` bit order has to be
    byte-reversed first. Both orientations are searched and the one the CRC
    actually validates is returned.
    """
    modes = range(len(SYNC_MODES)) if mode is None else (mode,)
    best: list[Fed1052Frame] = []
    for oriented, orientation in (
        (_reverse_each_byte(bits), "byte-reversed"),
        (bits, "selected"),
    ):
        found = _scan_values(_lsb_byte_values(oriented), orientation, correct, modes)
        if len(found) > len(best):
            best = found
    return best


def assemble_fed1052_message(frames: list[Fed1052Frame]) -> Fed1052Message | None:
    """Place each data frame's 56 payload bytes at its declared byte offset."""
    data_frames = [item for item in frames if item.kind == "data"]
    if not data_frames:
        return None
    placed: dict[int, bytes] = {}
    for item in data_frames:
        offset = item.payload_offset
        if offset % PAYLOAD_BYTES:
            continue
        placed.setdefault(offset, item.payload)
    if not placed:
        return None
    total = max(placed) + PAYLOAD_BYTES
    buffer = bytearray(total)
    for offset, payload in placed.items():
        buffer[offset:offset + PAYLOAD_BYTES] = payload
    gaps: list[tuple[int, int]] = []
    run = 0
    for offset in range(0, total, PAYLOAD_BYTES):
        if offset in placed:
            if run:
                gaps.append((offset - run * PAYLOAD_BYTES, run * PAYLOAD_BYTES))
                run = 0
        else:
            run += 1
    if run:
        gaps.append((total - run * PAYLOAD_BYTES, run * PAYLOAD_BYTES))
    return Fed1052Message(data_frames[0].mode, bytes(buffer), len(placed),
                          tuple(gaps))


#: Minimum bytes the OLE2 scan requires past the signature (``cmp edi, 0x493``
#: at ``0x100061bb``).
OLE2_MINIMUM = 0x493


def classify_fed1052_payload(data: bytes) -> list[tuple[str, int]]:
    """Payload content signatures the original DLL scans a blob for.

    ``fcn.10005df0`` runs three scanners over the reassembled blob and names
    the output file after whichever signature matches:

    * ``fcn.10003c10`` searches for ``4A 46 49 46`` (``JFIF``) at
      ``0x10003c77``-``0x10003c96`` and writes ``%.3d_%.2d_out.jpg``;
    * ``fcn.10003260`` searches for ``1F 8B 08`` at ``0x100033b2``-
      ``0x100033c6`` and writes ``%.3d_%.2d_out.gz``, then hands the file to
      ``Bin\\UnPack\\t520_GZ.exe``; and for ``50 4B 03 04`` at ``0x1000341a``-
      ``0x10003432`` or ``50 4B 07 08`` at ``0x10003492``-``0x100034aa``,
      writing ``%.3d_%.2d_out.zip``;
    * ``fcn.10005df0`` itself scans for an OLE2 compound-file header at
      ``0x10006181``-``0x100061ae`` — it checks bytes ``D0 CF`` at +0/+1 and
      ``A1 B1`` at +4/+5, skipping the middle two — requires at least
      ``0x493`` bytes to remain, and writes ``%.3d_outTNEF.4``.

    Returns ``(kind, offset)`` pairs in the order the DLL tests them.
    """
    found: list[tuple[str, int]] = []
    index = data.find(b"JFIF")
    if index >= 0:
        found.append(("jpg", index))
    index = data.find(b"\x1f\x8b\x08")
    if index >= 0:
        found.append(("gz", index))
    for marker in (b"PK\x03\x04", b"PK\x07\x08"):
        index = data.find(marker)
        if index >= 0:
            found.append(("zip", index))
    for index in range(max(0, len(data) - 5)):
        if (data[index] == 0xD0 and data[index + 1] == 0xCF
                and data[index + 4] == 0xA1 and data[index + 5] == 0xB1
                and len(data) - index >= OLE2_MINIMUM):
            found.append(("tnef", index))
            break
    return found


@dataclass(frozen=True)
class Fed1052Train:
    offset: int
    frames: int
    sync_byte: int
    inverted: bool
    orientation: str


def _reverse_each_byte(bits: np.ndarray) -> np.ndarray:
    complete = len(bits) // 8 * 8
    result = bits.copy()
    if complete:
        result[:complete] = bits[:complete].reshape(-1, 8)[:, ::-1].reshape(-1)
    return result


def _byte_values(bits: np.ndarray) -> np.ndarray:
    if len(bits) < 8:
        return np.empty(0, dtype=np.uint16)
    return np.correlate(bits.astype(np.uint16), 1 << np.arange(7, -1, -1), "valid")


def _byte_offsets(values: np.ndarray, value: int) -> list[int]:
    return np.flatnonzero(values == value).astype(int).tolist()


def _trains(offsets: list[int], sync_byte: int, inverted: bool,
            orientation: str) -> list[Fed1052Train]:
    offset_set = set(offsets)
    result = []
    for start in offsets:
        if start - PERIOD_BITS in offset_set:
            continue
        frames = 1
        while start + frames * PERIOD_BITS in offset_set:
            frames += 1
        if frames >= 4:
            result.append(Fed1052Train(start, frames, sync_byte, inverted, orientation))
    return result


def find_fed1052_trains(bits: np.ndarray) -> list[Fed1052Train]:
    result = []
    for oriented, orientation in (
        (bits, "selected"),
        (_reverse_each_byte(bits), "byte-reversed"),
    ):
        for inverted in (False, True):
            candidate = oriented ^ int(inverted)
            values = _byte_values(candidate)
            for sync_byte in SYNC_BYTES:
                result.extend(_trains(
                    _byte_offsets(values, sync_byte), sync_byte, inverted, orientation,
                ))
    return sorted(set(result), key=lambda item: (
        item.offset, item.sync_byte, item.orientation, item.inverted,
    ))


class FedStd1052Decoder(Decoder):
    name = "FED-STD-1052 (Harris Modems)"
    description = "Recovered 520F512 framing, CRC-32, and payload reassembly"
    status = "partial"
    confidence = "high"
    layer = "frame and package"
    provenance = "static analysis of original Lib/072.dll plus four bundled captures"

    def decode(self, bits: np.ndarray, block=None) -> DecodeResult:
        frames = find_fed1052_frames(bits)
        lines: list[str] = []
        artifacts: list[DecodeArtifact] = []
        if frames:
            headers = [item for item in frames if item.kind == "header"]
            data_frames = [item for item in frames if item.kind == "data"]
            repaired = sum(1 for item in frames if item.corrected_bit >= 0)
            lines.append(
                f"{len(frames)} CRC-valid 520-bit frames "
                f"({len(headers)} header, {len(data_frames)} data, "
                f"{repaired} repaired by single-bit correction)")
            for item in headers[:8]:
                lines.append(
                    f"@{item.offset}b header mode={item.mode} "
                    f"sync={item.sync:02X} to={item.destination!r} "
                    f"from={item.source!r} packages={item.package_count} "
                    f"payload={item.payload_size} total={item.total_size}B "
                    f"transfer={item.transfer}B")
            packages = find_fed1052_packages(frames)
            complete = [item for item in packages if item.complete]
            lines.append(
                f"{len(packages)} package run(s), {len(complete)} complete "
                "(declared frame count and countdown both satisfied)")
            message = assemble_fed1052_message(frames)
            if message is not None:
                lines.append(
                    f"reassembled {len(message.data)} payload bytes from "
                    f"{message.frames} data frames, {message.missing} missing")
                kinds = classify_fed1052_payload(message.data)
                lines.append("payload signatures: " + (
                    ", ".join(f"{kind}@{offset}" for kind, offset in kinds)
                    if kinds else "none (the original writes a plain DAT.0 file)"))
                artifacts.append(DecodeArtifact(
                    name="fed1052_payload.bin",
                    data=message.data,
                    notes=("payload bytes placed at the 24-bit offsets declared "
                           "by each data frame"),
                ))
        else:
            for item in find_fed1052_trains(bits):
                lines.append(
                    f"@{item.offset}b period=520 sync={item.sync_byte:02X} "
                    f"frames={item.frames} "
                    f"polarity={'inverted' if item.inverted else 'normal'} "
                    f"orientation={item.orientation}")
        return DecodeResult(
            text="\n".join(lines),
            symbol_count=len(frames),
            error_count=0,
            notes=("520F512 framing, CRC-32 frame check, single-bit correction, "
                   "and offset-addressed payload reassembly are implemented. "
                   "The original DLL hands the reassembled blob to the external "
                   "Bin/UnPack/t520_GZ.exe helper for GZip payloads; TNEF/MAPI "
                   "message parsing is not implemented."),
            artifacts=tuple(artifacts),
        )


register(FedStd1052Decoder())
