"""CRC checksum utilities (Tier A: public standards).

A CRC is a validity check, not a symbol decoder, so it doesn't implement the
``Decoder`` interface — it's a plain function used by framing/validation code
(e.g. checking a trailing checksum field against the preceding bytes).
"""
from __future__ import annotations


def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, MSB-first, no input/output reflection."""
    crc = init
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def crc16_x25(data: bytes) -> int:
    """CRC-16/X-25 (aka CRC-16/HDLC): poly 0x1021 reflected, init/xorout 0xFFFF.

    This is the FCS used by HDLC, AX.25, and X.25/LAPB framing -- a
    different CRC-16 variant than :func:`crc16_ccitt` (which is the
    non-reflected "CCITT-FALSE" variant used elsewhere).
    """
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
    return crc ^ 0xFFFF


def crc32(data: bytes) -> int:
    """Standard CRC-32 (poly 0xEDB88320, reflected), as used by zip/gzip/ethernet."""
    import zlib

    return zlib.crc32(data) & 0xFFFFFFFF
