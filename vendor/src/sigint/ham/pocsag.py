"""POCSAG: pager paging protocol (addresses + alphanumeric/numeric text).

A codeword is 32 bits: a flag bit (0 = address, 1 = message), 20 data
bits, 10 BCH(31,21) check bits, and one even-parity bit over the whole
31 bits above it. 16 codewords (8 frames of 2) form a batch, preceded by
a fixed 32-bit sync codeword; a pager only wakes for the frame matching
`address & 7`, so idle time between messages is filled with an idle
codeword rather than silence. The sync word (`0x7CD215D8`), the BCH
generator polynomial, the address/frame-number split, and the numeric
4-bit-to-digit table are all cross-checked against `multimon-ng`'s real
POCSAG decoder (`pocsag.c`) -- a widely-deployed open implementation, not
reconstructed from memory (POCSAG's alpha character bit-reversal in
particular -- `rev7` in that source -- is a well-known but easy-to-miss
detail: getting it backwards garbles every alpha message).

FLEX (the other major paging protocol, more common for US commercial
paging) is not implemented -- its multi-phase, multi-rate, block-
interleaved structure is substantially more involved than POCSAG's;
recognizing/decoding it would be a separate, comparably-sized project.

Self-rejecting: no sync codeword found, or a BCH check failure on every
candidate offset, returns ``None``.

Known limitation: numeric messages don't self-delimit their length at
the codeword level (bit-padding to fill the last 20-bit codeword decodes
as extra trailing digits) -- real deployments rely on the receiving
pager already knowing what to expect; `unpack_numeric_message` has no
such out-of-band hint, so a decoded numeric message may have a couple of
extra trailing characters. Alpha messages aren't affected (padding
decodes as NUL/space, stripped).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SYNC_CODEWORD = 0x7CD215D8
IDLE_CODEWORD = 0x7A89C197
BATCH_CODEWORDS = 16   # 8 frames x 2
PREAMBLE_BITS = 576    # alternating 1/0

_BCH_POLY = 0b11101101001   # x^10+x^9+x^8+x^6+x^5+x^3+1 (11 bits, degree 10)
_BCH_DATA_BITS = 21          # flag + 20 data bits
_NUMERIC_TABLE = "084 2.6]195-3U7["   # 4-bit nibble -> digit/symbol


def _bch_check_bits(data21: int) -> int:
    """10-bit BCH(31,21) remainder of a 21-bit (flag+data) value, via
    binary polynomial division (the systematic-cyclic-code encode step,
    same technique as any CRC in this codebase)."""
    reg = data21 << 10
    for i in range(_BCH_DATA_BITS - 1, -1, -1):
        if reg & (1 << (i + 10)):
            reg ^= _BCH_POLY << i
    return reg & 0x3FF


def _even_parity(value: int, bits: int) -> int:
    return bin(value & ((1 << bits) - 1)).count("1") & 1


def make_codeword(flag: int, data20: int) -> int:
    """flag(1) + data(20) -> the full 32-bit codeword (BCH + parity added)."""
    data21 = (flag << 20) | data20
    bch = _bch_check_bits(data21)
    cw31 = (data21 << 10) | bch
    parity = _even_parity(cw31, 31)
    return (cw31 << 1) | parity


def check_codeword(codeword: int) -> bool:
    """True if the codeword's BCH remainder and parity both check out."""
    if _even_parity(codeword >> 1, 31) != (codeword & 1):
        return False
    data21 = (codeword >> 11) & 0x1FFFFF
    bch = (codeword >> 1) & 0x3FF
    return _bch_check_bits(data21) == bch


def _rev7(b: int) -> int:
    """Reverse the bit order of a 7-bit value -- POCSAG transmits each
    alpha character's bits reversed within the packed message stream."""
    out = 0
    for i in range(7):
        if b & (1 << i):
            out |= 1 << (6 - i)
    return out


# --------------------------------------------------------------------------
# Message packing
# --------------------------------------------------------------------------

def pack_address(address: int, function: int = 3) -> int:
    """21-bit pager address + 2-bit function code -> one address codeword.
    Function 3 conventionally means "this pager also accepts alpha text"."""
    addr18 = (address >> 3) & 0x3FFFF
    data20 = (addr18 << 2) | (function & 3)
    return make_codeword(0, data20)


def pack_alpha_message(text: str) -> list[int]:
    """ASCII text -> message codewords (7 bits/char, bit-reversed, packed
    MSB-first across 20-bit codewords; the tail is zero-padded)."""
    bits = "".join(format(_rev7(ord(c) & 0x7F), "07b") for c in text)
    pad = (-len(bits)) % 20
    bits += "0" * pad
    return [make_codeword(1, int(bits[i:i + 20], 2)) for i in range(0, len(bits), 20)]


def unpack_alpha_message(codewords: list[int]) -> str:
    bits = "".join(format((cw >> 11) & 0xFFFFF, "020b") for cw in codewords)
    n = len(bits) // 7
    chars = [_rev7(int(bits[i * 7:i * 7 + 7], 2)) for i in range(n)]
    return "".join(chr(c) if 0x20 <= c < 0x7F else "" for c in chars).rstrip()


def pack_numeric_message(digits: str) -> list[int]:
    nibbles = [_NUMERIC_TABLE.index(c) for c in digits]
    if len(nibbles) % 2:
        nibbles.append(0)
    bits = "".join(format(n, "04b") for n in nibbles)
    pad = (-len(bits)) % 20
    bits += "0" * pad
    return [make_codeword(1, int(bits[i:i + 20], 2)) for i in range(0, len(bits), 20)]


def unpack_numeric_message(codewords: list[int]) -> str:
    bits = "".join(format((cw >> 11) & 0xFFFFF, "020b") for cw in codewords)
    n = len(bits) // 4
    return "".join(_NUMERIC_TABLE[int(bits[i * 4:i * 4 + 4], 2)] for i in range(n)).strip()


def build_batch(address: int, message: str, function: int = 3,
               numeric: bool = False) -> list[int]:
    """One address + its message -> a full 16-codeword batch (idle-filled
    everywhere else, message starting right after the address codeword,
    wrapping into later frames if it doesn't fit in the current one)."""
    msg_words = (pack_numeric_message(message) if numeric
                else pack_alpha_message(message))
    frame = address & 7
    batch = [IDLE_CODEWORD] * BATCH_CODEWORDS
    batch[frame * 2] = pack_address(address, function)
    slot = frame * 2 + 1
    for w in msg_words:
        if slot >= BATCH_CODEWORDS:
            break   # would need a second batch; out of scope for one message
        batch[slot] = w
        slot += 1
    return batch


# --------------------------------------------------------------------------
# Synthesis and detection
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PagerMessage:
    address: int
    function: int
    text: str
    numeric: bool


def generate(address: int, message: str, sample_rate: float, baud: float = 1200.0,
            function: int = 3, numeric: bool = False, base_hz: float = 5000.0,
            deviation_hz: float = 2000.0) -> np.ndarray:
    """Synthesize an FSK recording: preamble, one sync codeword, one
    batch carrying `message` addressed to `address`."""
    batch = build_batch(address, message, function=function, numeric=numeric)
    bits = ([1, 0] * (PREAMBLE_BITS // 2)
           + [int(b) for b in format(SYNC_CODEWORD, "032b")]
           + [int(b) for cw in batch for b in format(cw, "032b")])
    sps = max(1, int(round(sample_rate / baud)))
    phase = 0.0
    chunks = []
    t = np.arange(sps) / sample_rate
    for bit in bits:
        f = base_hz + (deviation_hz if bit else -deviation_hz)
        chunks.append(np.cos(2 * np.pi * f * t + phase))
        phase = (phase + 2 * np.pi * f * sps / sample_rate) % (2 * np.pi)
    return np.concatenate(chunks)


def decode(samples: np.ndarray, sample_rate: float, baud: float = 1200.0,
          base_hz: float = 5000.0, deviation_hz: float = 2000.0,
          phase_steps: int = 8, numeric: bool = False) -> PagerMessage | None:
    """Bit-synchronize, FSK-demod, find the sync codeword, and decode one
    batch's address + message. Self-rejects if no sync codeword (within
    2 bit errors) is found, or the address codeword's BCH fails."""
    x = np.asarray(samples, dtype=np.float64).ravel()
    sps = max(1, int(round(sample_rate / baud)))
    t = np.arange(sps) / sample_rate
    mark = np.exp(-2j * np.pi * (base_hz + deviation_hz) * t)
    space = np.exp(-2j * np.pi * (base_hz - deviation_hz) * t)

    best_bits, best_quality = None, -1.0
    for off in np.linspace(0, sps - 1, phase_steps).astype(int):
        n_sym = (len(x) - off) // sps
        if n_sym < 32:
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
    bitstr = "".join(str(int(b)) for b in best_bits)

    sync_bits = format(SYNC_CODEWORD, "032b")
    sync_pos = None
    for i in range(len(bitstr) - 32):
        if sum(a != b for a, b in zip(bitstr[i:i + 32], sync_bits)) <= 2:
            sync_pos = i
            break
    if sync_pos is None:
        return None

    pos = sync_pos + 32
    address, function = None, 0
    msg_words: list[int] = []
    for slot in range(BATCH_CODEWORDS):
        if pos + 32 > len(bitstr):
            break
        cw = int(bitstr[pos:pos + 32], 2)
        pos += 32
        if cw == IDLE_CODEWORD:
            continue
        if not check_codeword(cw):
            continue
        if (cw >> 31) == 0:
            if address is not None:
                break   # a second address codeword: stop, this pager's slot ended
            addr18 = (cw >> 13) & 0x3FFFF
            function = (cw >> 11) & 3
            frame = slot // 2   # which of the batch's 8 frames this slot is in
            address = (addr18 << 3) | frame
        else:
            msg_words.append(cw)

    if address is None:
        return None
    text = (unpack_numeric_message(msg_words) if numeric
           else unpack_alpha_message(msg_words))
    return PagerMessage(address=address, function=function, text=text, numeric=numeric)
