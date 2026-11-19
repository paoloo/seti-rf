"""Loader for BEE's ``.tab`` code tables.

These map an N-digit numeric code to a decoded character, one
``<code>\\t<value>\\r\\n`` pair per line. ``Bin/table/default.tab`` (confirmed
plain text, e.g. ``0000\\tU+6B7F``) is a 4-digit Chinese telegraph/GB2312 code
table; the value column is either a ``U+XXXX`` Unicode codepoint reference or
a literal string. The "Code" dropdown in the original UI picks one of these
tables to decode fixed-width groups of bits into text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CodeTable:
    name: str
    key_width: int  # digit count of the code key, e.g. 4 for "0000".."9999"
    entries: dict[str, str] = field(default_factory=dict)

    def lookup(self, code: int) -> str | None:
        """Look up by the integer value of an N-digit decimal code."""
        return self.entries.get(str(code).zfill(self.key_width))

    def lookup_key(self, key: str) -> str | None:
        return self.entries.get(key)

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def load(cls, path: str | Path) -> "CodeTable":
        path = Path(path)
        raw = path.read_bytes()
        text = _decode_text(raw)
        entries: dict[str, str] = {}
        key_width = 0
        for line in text.splitlines():
            if not line:
                continue
            key, _, value = line.partition("\t")
            if not value:
                continue
            entries[key] = _decode_value(value)
            key_width = max(key_width, len(key))
        return cls(name=path.stem, key_width=key_width, entries=entries)


def list_tables(table_dir: str | Path) -> list[Path]:
    """All ``.tab`` files available in a table directory (e.g. ``Bin/table``)."""
    return sorted(Path(table_dir).glob("*.tab"))


def _decode_text(raw: bytes) -> str:
    for encoding in ("utf-8", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def _decode_value(value: str) -> str:
    value = value.strip("\r")
    if value.startswith("U+"):
        try:
            return chr(int(value[2:], 16))
        except ValueError:
            return value
    return value
