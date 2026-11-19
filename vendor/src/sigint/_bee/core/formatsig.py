"""Parser for BEE's ``Bin/Format/format.bin`` file-signature database.

The database is a flat binary record stream:

    header: uint32 record_count
    record: uint32 ext_len, bytes ext,
            uint32 desc_len, bytes desc,
            uint32 pattern_count,
            pattern_count * (uint32 plen, bytes pattern, int32 trailing),
            uint32 record_trailing

This was reconstructed by hand (no source ships with the tool) by walking the
file byte-for-byte against known-good anchors (e.g. the ``1BK``/``2IMG``/
``3D2`` entries whose extensions and descriptions are unambiguous ASCII) and
confirming field boundaries land exactly on the next record's ``ext_len``.

That schema holds for simple entries (one or two flat magic-byte patterns,
the overwhelming majority of the database). A minority of entries describe
*compound* container formats (chunked formats like 3D Studio Mesh's 2-byte
chunk IDs, or multi-file signatures for ZIP-based formats like 3DMark
Results / 3D Movie Maker) whose pattern payloads embed their own nested,
format-specific structure that doesn't fit this flat schema — full parity
for those was not achieved. To stay robust in the presence of these,
:func:`load` validates each record (extension/description must look like
text, pattern/field lengths must be plausible) and, on a validation
failure, scans forward for the next well-formed record rather than trusting
a desynced read. This recovers the large majority of common, simple
signatures (executables, archives, images, audio, documents — everything
relevant to identifying an embedded payload carved out of a decoded
bitstream) while silently skipping the exotic compound entries it can't
parse. :attr:`SignatureDatabase.declared_count` vs. ``len(entries)`` makes
that gap visible rather than hiding it.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

_MAX_EXT_LEN = 32
_MAX_DESC_LEN = 500
_MAX_PATTERNS = 64
_MAX_PATTERN_LEN = 8192
_RESYNC_WINDOW = 4096


@dataclass
class Pattern:
    magic: bytes
    trailing: int  # meaning not fully determined; preserved for completeness


@dataclass
class FileSignature:
    extension: str
    description: str
    patterns: list[Pattern]


@dataclass
class SignatureDatabase:
    entries: list[FileSignature]
    declared_count: int

    def identify(self, data: bytes, anywhere: bool = False) -> list[FileSignature]:
        """Signatures whose magic bytes match the start of ``data``.

        The intended workflow is: isolate a candidate payload as a selection
        (e.g. the byte range right after a sync word/header field you've
        identified in the Graph view), then identify *that* range — so a
        header (offset-0) match is what "identify" means here, matching
        regardless of pattern length.

        Pass ``anywhere=True`` to also do a substring search over the whole
        buffer as a lower-confidence fallback (only for patterns >=4 bytes,
        since 1-2 byte patterns match unrelated binary data constantly by
        chance) — useful when you haven't isolated an exact start offset yet,
        but expect more false positives.
        """
        header_hits, anywhere_hits = [], []
        for entry in self.entries:
            for p in entry.patterns:
                if len(p.magic) < 2:
                    continue  # too short to mean anything; matches noise by chance
                if data.startswith(p.magic):
                    header_hits.append(entry)
                    break
                if anywhere and len(p.magic) >= 4 and p.magic in data:
                    anywhere_hits.append(entry)
                    break
        return header_hits + anywhere_hits

    @classmethod
    def load(cls, path: str | Path) -> "SignatureDatabase":
        data = Path(path).read_bytes()
        return _parse(data)


def _looks_like_text(b: bytes) -> bool:
    return bool(b) and all(9 <= c <= 13 or 32 <= c <= 126 or 160 <= c <= 255 for c in b)


def _try_parse_record(data: bytes, pos: int):
    end = len(data)
    if pos + 4 > end:
        return None
    (ext_len,) = struct.unpack_from("<I", data, pos)
    if not (0 < ext_len <= _MAX_EXT_LEN) or pos + 4 + ext_len > end:
        return None
    ext = data[pos + 4 : pos + 4 + ext_len]
    if not _looks_like_text(ext):
        return None
    p = pos + 4 + ext_len

    if p + 4 > end:
        return None
    (desc_len,) = struct.unpack_from("<I", data, p)
    if not (0 < desc_len <= _MAX_DESC_LEN) or p + 4 + desc_len > end:
        return None
    desc = data[p + 4 : p + 4 + desc_len]
    if not _looks_like_text(desc):
        return None
    p += 4 + desc_len

    if p + 4 > end:
        return None
    (pattern_count,) = struct.unpack_from("<I", data, p)
    if not (0 <= pattern_count <= _MAX_PATTERNS):
        return None
    p += 4

    patterns: list[Pattern] = []
    for _ in range(pattern_count):
        if p + 4 > end:
            return None
        (plen,) = struct.unpack_from("<I", data, p)
        if not (0 <= plen <= _MAX_PATTERN_LEN) or p + 4 + plen + 4 > end:
            return None
        magic = data[p + 4 : p + 4 + plen]
        p += 4 + plen
        (trailing,) = struct.unpack_from("<i", data, p)
        p += 4
        patterns.append(Pattern(magic=magic, trailing=trailing))

    if p + 4 > end:
        return None
    p += 4  # record_trailing, meaning undetermined

    signature = FileSignature(
        extension=ext.decode("latin-1"), description=desc.decode("latin-1"), patterns=patterns
    )
    return signature, p


def _parse(data: bytes) -> SignatureDatabase:
    if len(data) < 4:
        return SignatureDatabase(entries=[], declared_count=0)
    (declared_count,) = struct.unpack_from("<I", data, 0)

    entries: list[FileSignature] = []
    pos = 4
    while pos < len(data) - 8:
        result = _try_parse_record(data, pos)
        if result is None:
            resynced = False
            for cand in range(pos + 1, min(pos + _RESYNC_WINDOW, len(data) - 8)):
                result = _try_parse_record(data, cand)
                if result is not None:
                    pos = cand
                    resynced = True
                    break
            if not resynced:
                break
            continue
        signature, new_pos = result
        entries.append(signature)
        pos = new_pos

    return SignatureDatabase(entries=entries, declared_count=declared_count)
