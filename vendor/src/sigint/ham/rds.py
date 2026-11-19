"""RDS / RBDS (Radio Data System): the FM broadcast subcarrier data
service almost every FM station transmits (station name, PTY, alternative
frequencies) and nothing else in this package covers.

A group is 4 blocks of 26 bits (16 data + 10 check bits each), block
types A/B/C/C'/D identified by which of 5 fixed 10-bit "offset words" is
XORed into that block's checkword before transmission -- so checking a
block's syndrome against each offset word both validates it (CRC-style)
and identifies which block it is, with no separate framing bits needed.
The (26,16) cyclic code's parity-check matrix and the five offset word
values are transcribed from `redsea`'s real block-sync implementation
(github.com/windytan/redsea, `src/block_sync.cc`), which cites its
values to EN 50067 / IEC 62106 directly -- not reconstructed from memory.

This module works at the recovered-bitstream level (the biphase/Manchester
baseband signal after 57 kHz-subcarrier extraction and 1187.5 bps symbol
sync) rather than modeling the subcarrier itself -- the same scoping
choice :mod:`sigint.rfi.gps`/`.adsb` make for their own baseband-IQ inputs.

Only Group Type 0A (basic tuning: PI code, PTY, TP/TA flags, and the
8-character PS station name, assembled across 4 successive 0A groups) is
decoded -- by far the most commonly displayed RDS feature. Other group
types (RadioText, Alternative Frequencies, Clock Time, etc.) are not
implemented.

Self-rejecting: fewer than one full, syndrome-valid A-B-C-D block
sequence found in the bitstream returns ``None``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BIT_RATE_HZ = 1187.5
BLOCK_BITS = 26
DATA_BITS = 16
CHECK_BITS = 10

# EN 50067:1998 / IEC 62106, transcribed from redsea's block_sync.cc.
_PARITY_CHECK_MATRIX = (
    0b1000000000, 0b0100000000, 0b0010000000, 0b0001000000, 0b0000100000,
    0b0000010000, 0b0000001000, 0b0000000100, 0b0000000010, 0b0000000001,
    0b1011011100, 0b0101101110, 0b0010110111, 0b1010000111, 0b1110011111,
    0b1100010011, 0b1101010101, 0b1101110110, 0b0110111011, 0b1000000001,
    0b1111011100, 0b0111101110, 0b0011110111, 0b1010100111, 0b1110001111,
    0b1100011011,
)
OFFSET_A = 0b0011111100
OFFSET_B = 0b0110011000
OFFSET_C = 0b0101101000
OFFSET_CPRIME = 0b1101010000
OFFSET_D = 0b0110110100
_OFFSETS = {"A": OFFSET_A, "B": OFFSET_B, "C": OFFSET_C, "C'": OFFSET_CPRIME, "D": OFFSET_D}


def calculate_syndrome(vector26: int) -> int:
    """EN 50067 B.1.1: XOR together the matrix rows selected by set bits
    of `vector26`, row k for bit k (bit 0 = LSB = the checkword's own
    last bit in this module's `(data16 << 10) | check10` convention)."""
    result = 0
    for k in range(BLOCK_BITS):
        if (vector26 >> k) & 1:
            result ^= _PARITY_CHECK_MATRIX[k]
    return result


def _reverse_bits(v: int, n: int) -> int:
    r = 0
    for i in range(n):
        if (v >> i) & 1:
            r |= 1 << (n - 1 - i)
    return r


def encode_block(data16: int, block_type: str) -> int:
    """16-bit data word -> its 10-bit checkword for the given block type
    ('A'/'B'/'C'/"C'"/'D').

    The parity-check matrix's first 10 rows (transcribed as-is from
    redsea) form a bit-*reversing* identity on the checkword's own bits
    under this module's `(data16 << 10) | check10` vector convention --
    i.e. `calculate_syndrome(check10)` alone gives back `check10`
    bit-reversed, not unchanged. Reversing here undoes that so a fresh
    encode's own syndrome recomputes to exactly the offset word, the
    condition :func:`check_block` (and a receiver) tests for.
    """
    return _reverse_bits(calculate_syndrome(data16 << CHECK_BITS) ^ _OFFSETS[block_type],
                         CHECK_BITS)


def check_block(block26: int, block_type: str) -> bool:
    return calculate_syndrome(block26) == _OFFSETS[block_type]


def _bits_of(v: int, n: int) -> list[int]:
    return [(v >> (n - 1 - i)) & 1 for i in range(n)]


# --------------------------------------------------------------------------
# Group Type 0A: PI, PTY, TP/TA, and the 8-character PS name
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RDSInfo:
    pi_code: int
    pty: int
    tp: bool
    ta: bool
    ps_name: str   # accumulated from however many of the 4 segments were seen


def build_group_0a(pi_code: int, pty: int, tp: bool, ta: bool, ms: bool,
                   segment: int, ps_chars: str) -> tuple[int, int, int, int]:
    """One Group Type 0A's four 16-bit data words (PS name segment
    `segment` in 0..3, contributing 2 characters of the 8-char name)."""
    if len(ps_chars) != 2:
        raise ValueError("ps_chars must be exactly 2 characters")
    block_a = pi_code & 0xFFFF
    b_bits = ([0, 0, 0, 0]           # group type 0
             + [0]                  # version A
             + [int(tp)]
             + _bits_of(pty, 5)
             + [int(ta), int(ms), 0]  # TA, M/S, DI bit (not modeled)
             + _bits_of(segment, 2))
    block_b = int("".join(map(str, b_bits)), 2)
    block_c = 0xCDCD   # alternative-frequency codes: not modeled here
    block_d = (ord(ps_chars[0]) << 8) | ord(ps_chars[1])
    return block_a, block_b, block_c, block_d


def _parse_group_0a(a: int, b: int, d: int) -> tuple[int, int, bool, bool, int, str]:
    pi_code = a
    pty = (b >> 5) & 0x1F
    tp = bool((b >> 10) & 1)
    ta = bool((b >> 4) & 1)
    segment = b & 3
    chars = chr((d >> 8) & 0xFF) + chr(d & 0xFF)
    return pi_code, pty, tp, ta, segment, chars


# --------------------------------------------------------------------------
# Synthesis and detection
# --------------------------------------------------------------------------

def generate(pi_code: int, ps_name: str, sample_rate: float, pty: int = 0,
            tp: bool = False, ta: bool = False, base_hz: float = 5000.0,
            deviation_hz: float = 2000.0) -> np.ndarray:
    """Synthesize a biphase-coded (Manchester-family) recording of the
    4 Group 0A transmissions needed to carry an 8-character PS name.

    Modeled as FSK (bit -> one of two tones for the bit's whole period)
    rather than true biphase-mark subcarrier modulation -- consistent
    with this module working at the recovered-bitstream level, not the
    57 kHz-subcarrier level (see the module docstring).
    """
    ps_name = ps_name.ljust(8)[:8]
    bits: list[int] = []
    for segment in range(4):
        a, b, c, d = build_group_0a(pi_code, pty, tp, ta, False, segment,
                                    ps_name[segment * 2:segment * 2 + 2])
        for data16, block_type in ((a, "A"), (b, "B"), (c, "C"), (d, "D")):
            check = encode_block(data16, block_type)
            bits.extend(_bits_of(data16, 16))
            bits.extend(_bits_of(check, 10))

    sps = max(1, int(round(sample_rate / BIT_RATE_HZ)))
    phase = 0.0
    chunks = []
    t = np.arange(sps) / sample_rate
    for bit in bits:
        f = base_hz + (deviation_hz if bit else -deviation_hz)
        chunks.append(np.cos(2 * np.pi * f * t + phase))
        phase = (phase + 2 * np.pi * f * sps / sample_rate) % (2 * np.pi)
    return np.concatenate(chunks)


def decode(samples: np.ndarray, sample_rate: float, base_hz: float = 5000.0,
          deviation_hz: float = 2000.0, phase_steps: int = 8) -> RDSInfo | None:
    """Bit-synchronize, FSK-demod, and scan for syndrome-valid A-B-C/C'-D
    block sequences, decoding every Group 0A found. Self-rejects if no
    valid group is found."""
    x = np.asarray(samples, dtype=np.float64).ravel()
    sps = max(1, int(round(sample_rate / BIT_RATE_HZ)))
    t = np.arange(sps) / sample_rate
    mark = np.exp(-2j * np.pi * (base_hz + deviation_hz) * t)
    space = np.exp(-2j * np.pi * (base_hz - deviation_hz) * t)

    best_bits, best_quality = None, -1.0
    for off in np.linspace(0, sps - 1, phase_steps).astype(int):
        n_sym = (len(x) - off) // sps
        if n_sym < 26:
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

    pi_code, pty, tp, ta = None, 0, False, False
    ps_slots: dict[int, str] = {}
    i = 0
    while i + 26 * 4 <= len(bitstr):
        blocks = []
        for j, btype in enumerate(("A", "B", "C", "D")):
            block26 = int(bitstr[i + j * 26: i + j * 26 + 26], 2)
            candidates = ("C", "C'") if btype == "C" else (btype,)
            if not any(check_block(block26, c) for c in candidates):
                blocks = None
                break
            blocks.append(block26 >> CHECK_BITS)
        if blocks is None:
            i += 1
            continue
        a, b, c, d = blocks
        group_type = (b >> 12) & 0xF
        version = (b >> 11) & 1
        if group_type == 0 and version == 0:
            pi_code, pty, tp, ta, segment, chars = _parse_group_0a(a, b, d)
            ps_slots[segment] = chars
        i += 26 * 4

    if pi_code is None:
        return None
    ps_name = "".join(ps_slots.get(k, "  ") for k in range(4))
    return RDSInfo(pi_code=pi_code, pty=pty, tp=tp, ta=ta, ps_name=ps_name.rstrip())
