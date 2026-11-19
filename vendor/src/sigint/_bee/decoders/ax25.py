"""AX.25 packet radio framing (Tier B: public AX.25 2.2 spec, on top of HDLC).

Address field: destination + source callsigns (7 bytes each: 6 ASCII chars
left-shifted by 1 bit, plus an SSID byte whose bit 0 is the address-field
extension bit -- 0 means more addresses follow, 1 means this was the last
one), optionally followed by up to two repeater addresses. Then a 1-byte
control field, a 1-byte PID (for I/UI frames), the payload, and the same
HDLC FCS.
"""
from __future__ import annotations

import numpy as np

from sigint._bee.core.crc import crc16_x25
from sigint._bee.decoders.base import DecodeResult, Decoder
from sigint._bee.decoders.hdlc import _bits_to_bytes, destuff_and_split_frames
from sigint._bee.decoders.registry import register


def _decode_address(raw7: bytes) -> tuple[str, bool]:
    call = "".join(chr(b >> 1) for b in raw7[:6]).rstrip()
    ssid = (raw7[6] >> 1) & 0x0F
    is_last = bool(raw7[6] & 0x01)
    label = f"{call}-{ssid}" if ssid else call
    return label, is_last


def _describe_frame(data: bytes) -> str:
    if len(data) < 14:
        return f"(frame too short for AX.25 addressing: {data.hex(' ').upper()})"
    addrs = []
    pos = 0
    while pos + 7 <= len(data):
        label, is_last = _decode_address(data[pos : pos + 7])
        addrs.append(label)
        pos += 7
        if is_last:
            break
    if len(addrs) < 2 or pos + 2 > len(data):
        return f"(malformed address field: {data.hex(' ').upper()})"
    dest, src, *repeaters = addrs
    control = data[pos]
    pid = data[pos + 1] if pos + 1 < len(data) else None
    info = data[pos + 2 :] if pid is not None else b""
    via = f" via {','.join(repeaters)}" if repeaters else ""
    text = info.decode("latin-1", errors="replace")
    return f"{src}>{dest}{via} ctrl=0x{control:02X} pid=0x{pid:02X}: {text!r}"


class Ax25Decoder(Decoder):
    name = "PACKET AX.25"
    description = "AX.25 UI/I frames over standard HDLC bit-stuffed framing"

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
            lines.append(f"[{'OK ' if fcs_ok else 'BAD'}] {_describe_frame(payload)}")
        return DecodeResult(
            text="\n".join(lines),
            symbol_count=len(frames),
            error_count=error_count,
            notes=f"{len(frames)} AX.25 frame(s) found",
        )


register(Ax25Decoder())
