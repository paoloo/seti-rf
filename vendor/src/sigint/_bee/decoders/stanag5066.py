"""STANAG 5066 Data Transfer Sublayer generic D_PDU framing.

All D_PDUs begin with the 16-bit Maury-Styles sequence, observed on the byte
interface as ``90 EB``.  The common header starts with a 4-bit type, 12-bit
engineering order wire, 8-bit end-of-transmission value, and a byte carrying
the address/header sizes.  This decoder intentionally stops at the generic
header: type-specific ARQ state and segmented C_PDU reassembly are not claimed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sigint._bee.decoders.base import DecodeResult, Decoder
from sigint._bee.decoders.registry import register

SYNC = b"\x90\xeb"
TYPE_NAMES = {
    0: "DATA-ONLY", 1: "ACK-ONLY", 2: "DATA-ACK", 3: "RESET/WIN-RESYNC",
    4: "EXP-DATA-ONLY", 5: "EXP-ACK-ONLY", 6: "MANAGEMENT",
    7: "NON-ARQ DATA", 8: "EXP NON-ARQ DATA", 15: "WARNING",
}


@dataclass(frozen=True)
class GenericDpdu:
    offset: int
    frame_type: int
    eow: int
    eot: int
    address_size: int
    type_header_size: int
    address: bytes
    type_header: bytes
    header_crc: bytes


def parse_generic_dpdus(data: bytes) -> list[GenericDpdu]:
    frames: list[GenericDpdu] = []
    start = 0
    while True:
        offset = data.find(SYNC, start)
        if offset < 0:
            break
        start = offset + 1
        if offset + 6 > len(data):
            continue
        common = data[offset + 2 : offset + 6]
        frame_type = common[0] >> 4
        eow = ((common[0] & 0x0F) << 8) | common[1]
        eot = common[2]
        address_size = common[3] >> 5
        type_header_size = common[3] & 0x1F
        header_end = offset + 6 + address_size + type_header_size
        if header_end + 2 > len(data):
            continue
        address = data[offset + 6 : offset + 6 + address_size]
        type_header = data[offset + 6 + address_size : header_end]
        frames.append(GenericDpdu(
            offset, frame_type, eow, eot, address_size, type_header_size,
            address, type_header, data[header_end : header_end + 2],
        ))
    return frames


class Stanag5066Decoder(Decoder):
    name = "Stanag-5066"
    description = "Generic STANAG 5066 D_PDU sync and common-header parser"
    status = "partial"
    confidence = "high"
    layer = "framing"
    provenance = "STANAG 5066 Annex C generic D_PDU format plus original capture"

    def decode(self, bits: np.ndarray, block=None) -> DecodeResult:
        usable = bits[: len(bits) // 8 * 8]
        data = np.packbits(usable, bitorder="big").tobytes()
        frames = parse_generic_dpdus(data)
        lines = []
        for frame in frames:
            kind = TYPE_NAMES.get(frame.frame_type, "RESERVED")
            lines.append(
                f"@{frame.offset * 8}b type={frame.frame_type}:{kind} "
                f"eow=0x{frame.eow:03X} eot={frame.eot} "
                f"addr[{frame.address_size}]={frame.address.hex().upper()} "
                f"header[{frame.type_header_size}]={frame.type_header.hex().upper()} "
                f"crc={frame.header_crc.hex().upper()}"
            )
        return DecodeResult(
            text="\n".join(lines), symbol_count=len(frames), error_count=0,
            notes=(f"{len(frames)} Maury-Styles sync occurrence(s); generic headers only. "
                   "Header CRC validation and type-specific payload reassembly are not yet implemented."),
        )


register(Stanag5066Decoder())
