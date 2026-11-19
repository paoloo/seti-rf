"""ACARS (Aircraft Communications Addressing and Reporting System): the
VHF datalink aircraft use for short operational/maintenance messages --
HFDL's much simpler, much more common cousin.

Character-oriented, 2400 bps, MSK (modeled here as continuous-phase
binary FSK, the same simplification :mod:`sigint.ham.ais` uses for GMSK).
Every character is 7-bit ASCII plus one parity bit set so the total
number of 1-bits is odd -- including the control characters, which is
why ACARS's own ETX/ETB (``0x83``/``0x97``) don't look like the usual
ASCII ``0x03``/``0x17``: bit 7 is that character's own parity bit, not
part of its value. A message is ``SOH`` + mode(1) + address(7) + ack(1) +
label(2) + block ID(1) + ``STX`` + text + ``ETX``/``ETB`` + a 2-byte BCS
(block check sequence: CRC-CCITT, poly 0x8408, init 0 -- a message
including its own trailing BCS bytes must produce a zero remainder).
These constants (the control-character values *with* their parity bits
pre-applied, the message field order, and the CRC construction) are
cross-checked against `acarsdec`'s real, working decoder
(github.com/TLeconte/acarsdec, `acars.c` + `syndrom.h`), not
reconstructed from memory.

Self-rejecting: no ``SOH`` found, an odd-parity violation on any
character, or a nonzero BCS remainder, all return ``None``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BIT_RATE_HZ = 2400.0

SYN = 0x16
SOH = 0x01
STX = 0x02
ETX = 0x83
ETB = 0x97
DLE = 0x7F

_CRC_TABLE = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = (_c >> 1) ^ 0x8408 if _c & 1 else _c >> 1
    _CRC_TABLE.append(_c)


def bcs(data: bytes) -> int:
    """The block check sequence: CRC-CCITT (poly 0x8408, reflected),
    init 0, no final XOR."""
    crc = 0
    for byte in data:
        crc = (crc >> 8) ^ _CRC_TABLE[(crc ^ byte) & 0xFF]
    return crc


def add_parity(char7: int) -> int:
    """7-bit value -> that value with bit 7 set iff needed to make the
    total number of 1-bits odd (ACARS's per-character parity rule --
    this is exactly what turns plain ASCII ETX (0x03) into ACARS's own
    0x83, and leaves SOH (0x01, already odd) unchanged)."""
    char7 &= 0x7F
    return char7 if bin(char7).count("1") % 2 else char7 | 0x80


def check_parity(byte: int) -> bool:
    return bin(byte & 0xFF).count("1") % 2 == 1


# --------------------------------------------------------------------------
# Message packing
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ACARSMessage:
    mode: str
    address: str
    label: str
    block_id: str
    text: str
    ack: str = " "


def pack_message(mode: str, address: str, label: str, block_id: str,
                 text: str, ack: str = " ") -> bytes:
    """Field values -> the full byte sequence from SOH through the BCS
    (parity bits applied to every character, including the control ones)."""
    address = address.ljust(7)[:7]
    label = label.ljust(2)[:2]
    body = bytes(add_parity(ord(c)) for c in (mode + address + ack + label + block_id))
    body += bytes([STX])
    body += bytes(add_parity(ord(c)) for c in text)
    body += bytes([ETX])
    check = bcs(body)
    return bytes([SOH]) + body + bytes([check & 0xFF, (check >> 8) & 0xFF])


def unpack_message(frame: bytes) -> ACARSMessage | None:
    """The bytes from (and including) SOH through the BCS -> fields, or
    ``None`` if parity or the BCS don't check out."""
    if len(frame) < 1 + 1 + 7 + 1 + 2 + 1 + 1 + 1 + 2 or frame[0] != SOH:
        return None
    body, check_bytes = frame[1:-2], frame[-2:]
    if bcs(body + check_bytes) != 0:
        return None
    for b in body:
        if not check_parity(b):
            return None
    mode = chr(body[0] & 0x7F)
    address = "".join(chr(b & 0x7F) for b in body[1:8]).strip()
    ack = chr(body[8] & 0x7F)
    label = "".join(chr(b & 0x7F) for b in body[9:11])
    block_id = chr(body[11] & 0x7F)
    if body[12] != STX:
        return None
    text_bytes = body[13:-1]
    text = "".join(chr(b & 0x7F) for b in text_bytes)
    return ACARSMessage(mode=mode, address=address, label=label, block_id=block_id,
                        text=text, ack=ack)


# --------------------------------------------------------------------------
# Synthesis and detection
# --------------------------------------------------------------------------

def generate(mode: str, address: str, label: str, block_id: str, text: str,
            sample_rate: float, ack: str = " ", base_hz: float = 8000.0,
            deviation_hz: float = 2000.0, preamble_bits: int = 128) -> np.ndarray:
    """Synthesize an FSK recording: a NRZ-mark preamble tone (all-ones,
    the real ACARS "pre-key" bit-sync pattern) then the framed message."""
    frame = pack_message(mode, address, label, block_id, text, ack=ack)
    bits = ([1] * preamble_bits
           + [(byte >> i) & 1 for byte in frame for i in range(8)])
    sps = max(1, int(round(sample_rate / BIT_RATE_HZ)))
    phase = 0.0
    chunks = []
    t = np.arange(sps) / sample_rate
    for bit in bits:
        f = base_hz + (deviation_hz if bit else -deviation_hz)
        chunks.append(np.cos(2 * np.pi * f * t + phase))
        phase = (phase + 2 * np.pi * f * sps / sample_rate) % (2 * np.pi)
    return np.concatenate(chunks)


def decode(samples: np.ndarray, sample_rate: float, base_hz: float = 8000.0,
          deviation_hz: float = 2000.0, phase_steps: int = 8) -> ACARSMessage | None:
    """Bit-synchronize, FSK-demod, find ``SOH``, and unpack the message
    that follows. Self-rejects if no parity- and BCS-valid frame is
    found."""
    x = np.asarray(samples, dtype=np.float64).ravel()
    sps = max(1, int(round(sample_rate / BIT_RATE_HZ)))
    t = np.arange(sps) / sample_rate
    mark = np.exp(-2j * np.pi * (base_hz + deviation_hz) * t)
    space = np.exp(-2j * np.pi * (base_hz - deviation_hz) * t)

    best_bits, best_quality = None, -1.0
    for off in np.linspace(0, sps - 1, phase_steps).astype(int):
        n_sym = (len(x) - off) // sps
        if n_sym < 64:
            continue
        bits = np.empty(n_sym, dtype=np.int64)
        mags = np.empty(n_sym)
        for k in range(n_sym):
            seg = x[off + k * sps: off + (k + 1) * sps]
            m, s = abs(np.dot(seg, mark)), abs(np.dot(seg, space))
            bits[k] = 1 if m > s else 0
            mags[k] = max(m, s) / sps
        quality = float(mags.mean())
        if quality > best_quality:
            best_bits, best_quality = bits, quality

    if best_bits is None:
        return None
    n_bytes = len(best_bits) // 8
    byte_vals = []
    for i in range(n_bytes):
        byte = 0
        for b in range(8):
            byte |= int(best_bits[i * 8 + b]) << b
        byte_vals.append(byte)
    data = bytes(byte_vals)

    soh_positions = [i for i, b in enumerate(data) if b == SOH]
    for start in soh_positions:
        for end in range(start + 14, len(data) - 1):
            if data[end] in (ETX, ETB):
                msg = unpack_message(data[start:end + 3])
                if msg is not None:
                    return msg
    return None
