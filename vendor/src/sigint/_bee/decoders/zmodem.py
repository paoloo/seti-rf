"""ZMODEM frame parser (Tier A: public protocol, Forsberg's ZMODEM spec).

A frame is ``ZPAD ZPAD ZDLE <enc> <type> <4 data bytes> <CRC>``, where
``ZPAD``=0x2A, ``ZDLE``=0x18, and ``<enc>`` selects binary-CRC16 (0x42
'B'), binary-CRC32 (0x43 'C'), or hex (0x68 'h') encoding. Inside the
frame, 0x18 escapes the following byte (control chars and 0x18/0x7D/0xFF
themselves are sent as ZDLE, byte^0x40) to keep the data 8-bit/control-code
clean over a plain serial link.
"""
from __future__ import annotations

import numpy as np

from sigint._bee.core.crc import crc16_ccitt, crc32
from sigint._bee.decoders.base import DecodeResult, Decoder
from sigint._bee.decoders.registry import register

ZPAD = 0x2A
ZDLE = 0x18
ZBIN, ZBIN32, ZHEX = 0x42, 0x43, 0x68

_FRAME_TYPES = {
    0: "ZRQINIT", 1: "ZRINIT", 2: "ZSINIT", 3: "ZACK", 4: "ZFILE", 5: "ZSKIP",
    6: "ZNAK", 7: "ZABORT", 8: "ZFIN", 9: "ZRPOS", 10: "ZDATA", 11: "ZEOF",
    12: "ZFERR", 13: "ZCRC", 14: "ZCHALLENGE", 15: "ZCOMPL", 16: "ZCAN",
    17: "ZFREECNT", 18: "ZCOMMAND", 19: "ZSTDERR",
}


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    n = bits.size // 8
    trimmed = bits[: n * 8].reshape(n, 8)
    weights = (1 << np.arange(8)).astype(np.int64)  # LSB first
    return bytes(int(v) for v in (trimmed.astype(np.int64) @ weights))


def _find_frames(data: bytes) -> list[tuple[int, str]]:
    """Returns (frame_type, description) for each ZDLE header found."""
    results = []
    i = 0
    while i < len(data) - 3:
        if data[i] == ZPAD and data[i + 1] in (ZPAD, ZDLE):
            zdle_pos = i + 1 if data[i + 1] == ZDLE else i + 2
            if zdle_pos >= len(data) or data[zdle_pos] != ZDLE or zdle_pos + 1 >= len(data):
                i += 1
                continue
            enc = data[zdle_pos + 1]
            if enc not in (ZBIN, ZBIN32, ZHEX):
                i += 1
                continue
            header_start = zdle_pos + 2
            raw = data[header_start:]
            crc_size = 4 if enc == ZBIN32 else 2
            need = 5 + crc_size
            de_escaped = []
            j = 0
            while j < len(raw) and len(de_escaped) < need:
                b = raw[j]
                if b == ZDLE and j + 1 < len(raw):
                    de_escaped.append(raw[j + 1] ^ 0x40)
                    j += 2
                else:
                    de_escaped.append(b)
                    j += 1
            if len(de_escaped) < 5:
                i += 1
                continue
            frame_type = de_escaped[0]
            payload = bytes(de_escaped[:5])
            crc_ok = None
            if len(de_escaped) >= 5 + crc_size:
                received_crc_bytes = bytes(de_escaped[5 : 5 + crc_size])
                if enc == ZBIN32:
                    expected = crc32(payload)
                    received = int.from_bytes(received_crc_bytes, "little")
                    crc_ok = expected == received
                else:
                    expected = crc16_ccitt(payload)
                    received = int.from_bytes(received_crc_bytes, "big")
                    crc_ok = expected == received
            name = _FRAME_TYPES.get(frame_type, f"type{frame_type}")
            status = "" if crc_ok is None else (" CRC-OK" if crc_ok else " CRC-BAD")
            results.append((frame_type, f"{name}{status}"))
            i = header_start
        else:
            i += 1
    return results


class ZModemDecoder(Decoder):
    name = "ZModem"
    description = "ZMODEM ZPAD/ZDLE frame headers, CRC-16/CRC-32 checked"

    def decode(self, bits: np.ndarray, block=None) -> DecodeResult:
        data = _bits_to_bytes(bits)
        frames = _find_frames(data)
        text = "\n".join(desc for _, desc in frames)
        errors = sum(1 for _, desc in frames if "CRC-BAD" in desc)
        return DecodeResult(
            text=text, symbol_count=len(frames), error_count=errors,
            notes=f"{len(frames)} ZMODEM frame header(s) found",
        )


register(ZModemDecoder())
