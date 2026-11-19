"""Reader/writer for BEE's ``.blk`` session file format.

Confirmed from ``Data/111.blk`` to be plain text (CRLF line endings), not a
binary format:

    Block Data File

    Division of session:
        type    1
        size    11
        syn     11111000000
    ------------------------------
    PERIOD 0
    Block 0
        type    1
        size    8
        syn     00001111
        delta   0
        reserv  1
        precis  0
        order   1
        conform 0
        mask    0

    Block 1
        ...
    ------------------------------
    eof

Each ``Block`` is one field of the frame being analyzed (matches the F1/D1/
F2/D2 field annotations seen in the original UI): a bit ``size``, an optional
``syn`` sync pattern, bit ``order``, and optional ``conform`` (value
correction table: N ``data``/``real`` pairs) and ``mask`` (N-item value list)
sub-tables. A file can have multiple ``PERIOD`` sections (different period
hypotheses tried while framing a signal).

A second variant exists (``Data/obr.blk``, ``Data/test3.blk``): a
``Division`` of ``type 4`` followed by raw binary bytes embedded directly
after the ``Block 0`` header, with no further text structure recoverable.
This looks like a self-contained capture+session file. Its exact internal
record layout isn't recoverable from samples alone, so it is preserved
byte-for-byte as :attr:`Project.raw_tail` rather than mis-parsed.
"""
from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

_HEADER = "Block Data File"
_SEPARATOR = "-" * 30
_KV_RE = re.compile(r"^\t([A-Za-z]+)\s*(.*)$")
_SUBKV_RE = re.compile(r"^\t\t([A-Za-z]+)\s*(.*)$")
_INT_FIELDS = ("type", "size", "delta", "reserv", "precis", "order")

_EOF = object()
_BINARY = object()
_RAW_MARKER = "BEEPY RAW"


@dataclass
class Division:
    type: int = 0
    size: int = 0
    syn: str = ""


@dataclass
class Block:
    index: int
    type: int = 0
    size: int = 0
    syn: str = ""
    delta: int = 0
    reserv: int = 1
    precis: int = 0
    order: int = 1
    conform: list[tuple[int, int]] = field(default_factory=list)
    mask_values: list[int] = field(default_factory=list)


@dataclass
class Period:
    index: int
    blocks: list[Block] = field(default_factory=list)


@dataclass
class Project:
    division: Division = field(default_factory=Division)
    periods: list[Period] = field(default_factory=list)
    raw_tail: bytes = b""

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        return _load(Path(path).read_bytes())

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(_dump(self))


class _Reader:
    """Line reader over raw bytes that can detect and rewind past binary data."""

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    @property
    def pos(self) -> int:
        return self._pos

    def seek(self, pos: int) -> None:
        self._pos = pos

    def next_line(self):
        if self._pos >= len(self._data):
            return _EOF
        idx = self._data.find(b"\r\n", self._pos)
        if idx == -1:
            raw, new_pos = self._data[self._pos :], len(self._data)
        else:
            raw, new_pos = self._data[self._pos : idx], idx + 2
        try:
            text = raw.decode("ascii")
        except UnicodeDecodeError:
            return _BINARY
        self._pos = new_pos
        return text

    def tail(self) -> bytes:
        return self._data[self._pos :]


def _assign_int(obj, key: str, value: str) -> None:
    if key == "syn":
        obj.syn = value
        return
    if key in _INT_FIELDS:
        try:
            setattr(obj, key, int(value))
        except ValueError:
            pass


def _is_section_boundary(line: str) -> bool:
    return (
        line.startswith("Block ")
        or line.startswith("PERIOD")
        or line.startswith(_RAW_MARKER)
        or line.startswith(_SEPARATOR[:3])
        or line.strip() == "eof"
    )


def _parse_block(reader: _Reader, header_line: str) -> tuple[Block, bool]:
    """Parse one Block section. Returns (block, hit_binary_tail)."""
    parts = header_line.split()
    index = int(parts[1]) if len(parts) > 1 else 0
    block = Block(index=index)
    while True:
        save_pos = reader.pos
        line = reader.next_line()
        if line is _EOF:
            return block, False
        if line is _BINARY:
            reader.seek(save_pos)
            return block, True
        if not line.strip():
            continue
        if _is_section_boundary(line):
            reader.seek(save_pos)
            return block, False
        m = _KV_RE.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip()
        if key == "conform":
            count = int(value) if value.strip().isdigit() else 0
            for _ in range(count):
                data_line = reader.next_line()
                real_line = reader.next_line()
                if data_line is _BINARY or real_line is _BINARY or data_line is _EOF:
                    return block, True
                dm = _SUBKV_RE.match(data_line) if isinstance(data_line, str) else None
                rm = _SUBKV_RE.match(real_line) if isinstance(real_line, str) else None
                d_val = int(dm.group(2)) if dm and dm.group(2).strip().lstrip("-").isdigit() else 0
                r_val = int(rm.group(2)) if rm and rm.group(2).strip().lstrip("-").isdigit() else 0
                block.conform.append((d_val, r_val))
        elif key == "mask":
            count = int(value) if value.strip().isdigit() else 0
            if count:
                vals_line = reader.next_line()
                if isinstance(vals_line, str):
                    block.mask_values = [
                        int(x) for x in vals_line.strip().split() if x.lstrip("-").isdigit()
                    ]
        else:
            _assign_int(block, key, value)


def _load(data: bytes) -> Project:
    reader = _Reader(data)
    header = reader.next_line()
    if not isinstance(header, str) or header.strip() != _HEADER:
        raise ValueError("not a BEE .blk file (missing 'Block Data File' header)")

    line = reader.next_line()
    while isinstance(line, str) and not line.strip():
        line = reader.next_line()
    if not isinstance(line, str) or not line.startswith("Division of session"):
        raise ValueError("not a BEE .blk file (missing 'Division of session' section)")

    division = Division()
    while True:
        line = reader.next_line()
        if line is _EOF:
            return Project(division=division, periods=[])
        if line is _BINARY or (isinstance(line, str) and line.startswith(_SEPARATOR[:3])):
            break
        if isinstance(line, str):
            m = _KV_RE.match(line)
            if m:
                _assign_int(division, m.group(1), m.group(2).strip())

    periods: list[Period] = []
    while True:
        line = reader.next_line()
        if line is _EOF or (isinstance(line, str) and line.strip() == "eof"):
            return Project(division=division, periods=periods)
        if line is _BINARY:
            return Project(division=division, periods=periods, raw_tail=reader.tail())
        if line.startswith(_RAW_MARKER):
            parts = line.split()
            if len(parts) != 4 or not parts[2].isdigit():
                raise ValueError("invalid BEEPY RAW marker")
            raw = reader.tail()
            expected_size, expected_hash = int(parts[2]), parts[3]
            if len(raw) != expected_size:
                raise ValueError("embedded capture size mismatch")
            if hashlib.sha256(raw).hexdigest() != expected_hash:
                raise ValueError("embedded capture hash mismatch")
            return Project(division=division, periods=periods, raw_tail=raw)
        if not line.startswith("PERIOD"):
            continue
        parts = line.split()
        period = Period(index=int(parts[1]) if len(parts) > 1 else len(periods))
        while True:
            save_pos = reader.pos
            bline = reader.next_line()
            if bline is _EOF:
                periods.append(period)
                return Project(division=division, periods=periods)
            if bline is _BINARY:
                periods.append(period)
                reader.seek(save_pos)
                return Project(division=division, periods=periods, raw_tail=reader.tail())
            if not bline.strip():
                continue
            if bline.startswith("Block"):
                block, hit_binary = _parse_block(reader, bline)
                period.blocks.append(block)
                if hit_binary:
                    periods.append(period)
                    return Project(division=division, periods=periods, raw_tail=reader.tail())
                continue
            # PERIOD / separator / eof: end this period, replay the line for the outer loop
            reader.seek(save_pos)
            break
        periods.append(period)


def _kv(key: str, value, indent: int = 1) -> str:
    pad = 8 if indent == 1 else 7
    return f"{'	' * indent}{str(key).ljust(pad)}{value}"


def _block_lines(block: Block) -> list[str]:
    lines = [
        f"Block {block.index}",
        _kv("type", block.type),
        _kv("size", block.size),
        _kv("syn", block.syn),
        _kv("delta", block.delta),
        _kv("reserv", block.reserv),
        _kv("precis", block.precis),
        _kv("order", block.order),
        _kv("conform", len(block.conform)),
    ]
    for data_val, real_val in block.conform:
        lines.append(_kv("data", data_val, indent=2))
        lines.append(_kv("real", real_val, indent=2))
    lines.append(_kv("mask", len(block.mask_values)))
    if block.mask_values:
        lines.append("\t\t" + " ".join(str(v) for v in block.mask_values))
    lines.append("\t")
    return lines


def _dump(project: Project) -> bytes:
    lines = [_HEADER, "", "Division of session:"]
    d = project.division
    lines += [_kv("type", d.type), _kv("size", d.size), _kv("syn", d.syn)]
    lines.append(_SEPARATOR)
    for i, period in enumerate(project.periods):
        lines.append(f"PERIOD {period.index}")
        for block in period.blocks:
            lines.extend(_block_lines(block))
        is_last = i == len(project.periods) - 1
        if not (is_last and project.raw_tail):
            lines.append(_SEPARATOR)
    if project.raw_tail:
        digest = hashlib.sha256(project.raw_tail).hexdigest()
        lines.append(f"{_RAW_MARKER} {len(project.raw_tail)} {digest}")
        text = ("\r\n".join(lines) + "\r\n").encode("ascii")
        return text + project.raw_tail
    lines.append("eof")
    return ("\r\n".join(lines) + "\r\n").encode("ascii")
