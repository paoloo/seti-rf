"""HDLC bit-oriented framing (Tier B: public ISO 13239 / ITU-T Q.921 standard).

Frames are delimited by the flag byte 0x7E; between flags, the transmitter
"bit-stuffs" a 0 after every run of five consecutive 1-bits so the flag
pattern can't occur mid-frame, and the receiver must undo that. Each frame
ends with a 16-bit FCS (CRC-16/X-25) over the de-stuffed frame content.

Covers HDLC, RSX.25 and Q15X25 (both X.25/LAPB-style protocols that reuse
standard HDLC framing+FCS -- the Lib/074.dll and Lib/089.dll strings
("Rsx25.dll", "Q15x25.dll") only differ in modem-specific physical-layer
details upstream of this framing layer, which is out of scope for a
bit-level viewer).
"""
from __future__ import annotations

import numpy as np

from sigint._bee.core.crc import crc16_x25
from sigint._bee.decoders.base import DecodeResult, Decoder
from sigint._bee.decoders.registry import register

FLAG = [0, 1, 1, 1, 1, 1, 1, 0]  # 0x7E, LSB first


def _bits_to_bytes(bits: list[int]) -> bytes:
    n = len(bits) // 8
    out = bytearray(n)
    for i in range(n):
        byte = 0
        for b in range(8):
            byte |= bits[i * 8 + b] << b
        out[i] = byte
    return bytes(out)


def destuff_and_split_frames(bits: np.ndarray) -> list[list[int]]:
    """Find flag-delimited frames and remove stuffed zero bits from each."""
    bit_list = bits.tolist()
    n = len(bit_list)
    flag_positions = [i for i in range(n - 7) if bit_list[i : i + 8] == FLAG]

    frames: list[list[int]] = []
    for start, end in zip(flag_positions, flag_positions[1:]):
        raw = bit_list[start + 8 : end]
        if not raw:
            continue
        destuffed: list[int] = []
        ones_run = 0
        i = 0
        while i < len(raw):
            bit = raw[i]
            if ones_run == 5:
                ones_run = 0
                if bit == 0:
                    i += 1  # stuffed bit, drop it
                    continue
            destuffed.append(bit)
            ones_run = ones_run + 1 if bit == 1 else 0
            i += 1
        frames.append(destuffed)
    return frames


class HdlcDecoder(Decoder):
    name = "HDLC"
    description = "Flag-delimited, bit-stuffed frames with a CRC-16/X-25 FCS"

    def decode(self, bits: np.ndarray, block=None) -> DecodeResult:
        frames = destuff_and_split_frames(bits)
        lines = []
        error_count = 0
        for frame_bits in frames:
            data = _bits_to_bytes(frame_bits)
            if len(data) < 3:
                continue
            payload, fcs_bytes = data[:-2], data[-2:]
            fcs_received = fcs_bytes[0] | (fcs_bytes[1] << 8)
            fcs_ok = crc16_x25(payload) == fcs_received
            if not fcs_ok:
                error_count += 1
            hexdump = payload.hex(" ").upper()
            lines.append(f"[{'OK ' if fcs_ok else 'BAD'}] {hexdump}")
        return DecodeResult(
            text="\n".join(lines),
            symbol_count=len(frames),
            error_count=error_count,
            notes=f"{len(frames)} HDLC frame(s) found between 0x7E flags",
        )


for _name in ("HDLC", "RSX.25", "Q15X25"):
    _decoder = HdlcDecoder()
    _decoder.name = _name
    register(_decoder)
