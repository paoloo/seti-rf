"""WSPR (Weak Signal Propagation Reporter): the third WSJT-X mode, and
much simpler than FT8/FT4's LDPC -- an extremely narrowband (~6 Hz total)
4-GFSK beacon carrying just callsign + 4-character grid + power, meant to
be decoded 20-30 dB below the noise floor over a fixed 110.6 second
transmission.

The message (28-bit callsign + 15-bit grid + 7-bit power = 50 bits) is
protected by a rate-1/2, constraint-length-**32** convolutional code (not
LDPC) -- an unusually long constraint length that makes a textbook
from-scratch Viterbi decoder (2^31 possible shift-register states)
computationally impossible. This module encodes the real way (the exact
scheme production WSPR software uses) and decodes via a bounded-beam
(M-algorithm) sequential search over that trellis instead of full
Viterbi: at each step, keep only the `beam_width` highest-metric partial
paths rather than all of them. This is a legitimate, standard technique
for long-constraint-length codes, just with less processing gain than
WSJT-X's own tuned sequential decoder -- reflected in this module's
weaker noise tolerance versus :mod:`sigint.ham.ft8`'s LDPC, which is
exactly why LDPC largely displaced codes like this one in newer protocols.

The convolutional polynomials (`0xf2d05351`, `0xe4613c47`), the 162-bit
sync vector, and the bit-reversal interleave table are transcribed from a
real, working WSPR encoder (`micro-wspr`,
github.com/stephanelsmith/micro-wspr, `src/wspr/encoder.py` -- itself
crediting Mark VandeWettering K6HX's original `genwspr`), not
reconstructed from memory.

Self-rejecting: the best surviving path's convolutional-code flush tail
(the last 31 decoded bits, which a valid message always zeros out) isn't
all-zero, or the decoded callsign doesn't parse, returns ``None``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SYMBOL_COUNT = 162
BAUD_HZ = 12000.0 / 8192.0        # ~1.4648 baud
TONE_SPACING_HZ = 12000.0 / 8192.0  # == baud (4-GFSK, tones adjacent)
CONSTRAINT_LEN = 32
MESSAGE_BITS = 50

_POLY1 = 0xF2D05351
_POLY2 = 0xE4613C47

_SYNC_VECTOR = (
    1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1,
    1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
    1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1,
    0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0,
)

# bit-reversal-of-8-bit-index interleave, values < 162 kept, in order.
_RIDX = tuple(i for i in (int(format(x, "08b")[::-1], 2) for x in range(256)) if i < 162)

_CALLSIGN_DIGITS = "0123456789"
_CALLSIGN_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LDS = _CALLSIGN_DIGITS + _CALLSIGN_UPPER + " "
_LD = _CALLSIGN_DIGITS + _CALLSIGN_UPPER
_D = _CALLSIGN_DIGITS
_LS = _CALLSIGN_UPPER + " "


# --------------------------------------------------------------------------
# Message packing
# --------------------------------------------------------------------------

def _normalize_callsign(callsign: str) -> str:
    idx = next(i for i, ch in enumerate(callsign) if ch in _CALLSIGN_DIGITS)
    out = list("      ")
    out[2 - idx:2 - idx + len(callsign)] = list(callsign)
    return "".join(out)


def encode_callsign(callsign: str) -> int:
    call = _normalize_callsign(callsign.upper())
    acc = _LDS.find(call[0])
    acc = acc * len(_LD) + _LD.find(call[1])
    acc = acc * len(_D) + _D.find(call[2])
    acc = acc * len(_LS) + _LS.find(call[3])
    acc = acc * len(_LS) + _LS.find(call[4])
    acc = acc * len(_LS) + _LS.find(call[5])
    return acc


def decode_callsign(acc: int) -> str:
    chars = [""] * 6
    acc, r = divmod(acc, len(_LS)); chars[5] = _LS[r]
    acc, r = divmod(acc, len(_LS)); chars[4] = _LS[r]
    acc, r = divmod(acc, len(_LS)); chars[3] = _LS[r]
    acc, r = divmod(acc, len(_D)); chars[2] = _D[r]
    acc, r = divmod(acc, len(_LD)); chars[1] = _LD[r]
    chars[0] = _LDS[acc]
    return "".join(chars).strip()


def grid_to_latlon(grid: str) -> tuple[float, float]:
    grid = grid.upper()
    p = (ord(grid[0]) - ord("A")) * 10 + (ord(grid[2]) - ord("0"))
    p = p * 24 + 12
    lon = (p / 12) - 180.0
    p = (ord(grid[1]) - ord("A")) * 10 + (ord(grid[3]) - ord("0"))
    p = p * 24 + 12
    lat = (p / 24) - 90.0
    return lat, lon


def encode_latlon(lat: float, lon: float) -> int:
    lon_i = int((180 - lon) / 2.0)
    lat_i = int(lat + 90)
    return lon_i * 180 + lat_i


def decode_latlon(value: int) -> tuple[float, float]:
    lon_i, lat_i = divmod(value, 180)
    return lat_i - 90, 180 - lon_i * 2.0


@dataclass(frozen=True)
class WSPRMessage:
    callsign: str
    lat: float
    lon: float
    power_dbm: int


def pack_message(callsign: str, grid: str, power_dbm: int) -> np.ndarray:
    """callsign + 4-char Maidenhead grid + power (dBm) -> the 50-bit
    message (28 + 15 + 7)."""
    call_bits = format(encode_callsign(callsign), "028b")
    lat, lon = grid_to_latlon(grid)
    grid_bits = format(encode_latlon(lat, lon), "015b")
    power_bits = format(power_dbm + 64, "07b")
    bits = call_bits + grid_bits + power_bits
    return np.array([int(b) for b in bits], dtype=np.int64)


def unpack_message(bits50: np.ndarray) -> WSPRMessage:
    bitstr = "".join(str(int(b)) for b in bits50)
    call_acc = int(bitstr[0:28], 2)
    latlon = int(bitstr[28:43], 2)
    power = int(bitstr[43:50], 2) - 64
    callsign = decode_callsign(call_acc)
    lat, lon = decode_latlon(latlon)
    return WSPRMessage(callsign=callsign, lat=lat, lon=lon, power_dbm=power)


# --------------------------------------------------------------------------
# The K=32, r=1/2 convolutional code + sync interleave
# --------------------------------------------------------------------------

def _parity32(x: int) -> int:
    return bin(x).count("1") & 1


def convolve(message_bits: np.ndarray) -> np.ndarray:
    """The 50-bit message, flush-padded to 81 bits (50 + 31 zero tail
    bits), through the rate-1/2 K=32 encoder -> 162 channel bits."""
    bits = np.concatenate([message_bits, np.zeros(CONSTRAINT_LEN - 1, dtype=np.int64)])
    acc = 0
    out = []
    for b in bits:
        acc = ((acc << 1) & 0xFFFFFFFF) | int(b)
        out.append(_parity32(acc & _POLY1))
        out.append(_parity32(acc & _POLY2))
    return np.array(out, dtype=np.int64)


def encode_symbols(message_bits: np.ndarray) -> np.ndarray:
    """The 50-bit message -> 162 tone indices (0..3), sync-vector-
    combined and bit-reversal-interleaved."""
    channel_bits = convolve(message_bits)
    interleaved = np.zeros(SYMBOL_COUNT, dtype=np.int64)
    for i in range(SYMBOL_COUNT):
        interleaved[_RIDX[i]] = channel_bits[i]
    return np.array([2 * interleaved[i] + _SYNC_VECTOR[i] for i in range(SYMBOL_COUNT)])


def _deinterleave_llr(symbol_llr: np.ndarray) -> np.ndarray:
    """Undo the bit-reversal interleave on a per-symbol channel-bit LLR
    array (the inverse of `encode_symbols`'s `interleaved[_RIDX[i]]`)."""
    out = np.zeros(SYMBOL_COUNT)
    for i in range(SYMBOL_COUNT):
        out[i] = symbol_llr[_RIDX[i]]
    return out


def decode_symbols(tone_llr: np.ndarray, beam_width: int = 4000
                   ) -> np.ndarray | None:
    """Channel-bit LLRs (162 values, one per symbol, positive = more
    likely a 0 bit; already de-interleaved, i.e. in *transmission* order
    matching `convolve`'s output) -> the 50-bit message, via a bounded-
    beam search over the K=32 trellis. Returns ``None`` if no surviving
    path's flush tail comes out all-zero.

    State is the last 31 encoded bits (as a python int); each surviving
    path also carries its own decoded-bit history for the final
    traceback -- with `beam_width` capped in the low thousands this stays
    cheap enough for a background test run, not for a hot loop.
    """
    beams: dict[int, tuple[float, list[int]]] = {0: (0.0, [])}
    for pair in range(81):
        l0, l1 = tone_llr[2 * pair], tone_llr[2 * pair + 1]
        next_beams: dict[int, tuple[float, list[int]]] = {}
        for state, (metric, path) in beams.items():
            for bit in (0, 1):
                acc = ((state << 1) | bit) & 0xFFFFFFFF
                p0, p1 = _parity32(acc & _POLY1), _parity32(acc & _POLY2)
                branch = (l0 if p0 == 0 else -l0) + (l1 if p1 == 0 else -l1)
                new_state = acc & ((1 << (CONSTRAINT_LEN - 1)) - 1)
                m = metric + branch
                if new_state not in next_beams or m > next_beams[new_state][0]:
                    next_beams[new_state] = (m, path + [bit])
        if len(next_beams) > beam_width:
            keep = sorted(next_beams, key=lambda s: next_beams[s][0], reverse=True)[:beam_width]
            next_beams = {s: next_beams[s] for s in keep}
        beams = next_beams
        if not beams:
            return None

    best_state = max(beams, key=lambda s: beams[s][0])
    _, path = beams[best_state]
    bits = np.array(path, dtype=np.int64)
    if not np.all(bits[MESSAGE_BITS:] == 0):
        return None
    return bits[:MESSAGE_BITS]


# --------------------------------------------------------------------------
# Synthesis and detection
# --------------------------------------------------------------------------

def generate(callsign: str, grid: str, power_dbm: int, sample_rate: float,
            base_hz: float = 1500.0) -> np.ndarray:
    """Synthesize a continuous-phase 4-GFSK (modeled without Gaussian
    shaping, like this package's other ham-mode generators) recording."""
    message = pack_message(callsign, grid, power_dbm)
    tones = encode_symbols(message)
    sps = max(1, int(round(sample_rate / BAUD_HZ)))
    phase = 0.0
    chunks = []
    t = np.arange(sps) / sample_rate
    for tone in tones:
        f = base_hz + tone * TONE_SPACING_HZ
        chunks.append(np.cos(2 * np.pi * f * t + phase))
        phase = (phase + 2 * np.pi * f * sps / sample_rate) % (2 * np.pi)
    return np.concatenate(chunks)


def decode(samples: np.ndarray, sample_rate: float, base_hz: float = 1500.0,
          beam_width: int = 4000) -> WSPRMessage | None:
    """Demodulate (per-symbol 4-tone energy -> per-channel-bit LLR) and
    decode via the bounded-beam trellis search. Self-rejects if the
    convolutional code's flush tail doesn't come out clean."""
    x = np.asarray(samples, dtype=np.float64).ravel()
    sps = max(1, int(round(sample_rate / BAUD_HZ)))
    n_sym = len(x) // sps
    if n_sym < SYMBOL_COUNT:
        return None
    t = np.arange(sps) / sample_rate
    refs = np.array([np.exp(-2j * np.pi * (base_hz + k * TONE_SPACING_HZ) * t)
                     for k in range(4)])

    tone_energy = np.zeros((SYMBOL_COUNT, 4))
    for k in range(SYMBOL_COUNT):
        seg = x[k * sps:(k + 1) * sps]
        tone_energy[k] = np.abs(refs @ seg)

    # silence/pure noise gives near-equal, near-zero tone energies, which
    # the trellis search can still "explain" with some path (LLRs near
    # zero don't penalize any bit choice) -- the flush-tail check alone
    # doesn't catch this, so gate on there being real per-symbol tone
    # contrast at all first (the same "is this actually a signal" role
    # sigint.ham.morse's on/off contrast check and sigint.detect's own
    # prominence checks play elsewhere).
    if float(tone_energy.max()) < 1e-9 * sps or float(tone_energy.std()) < 1e-6 * tone_energy.mean() + 1e-12:
        return None

    # per-symbol tone -> sync bit (LSB) and channel bit (the rest):
    # tone = 2*channel_bit + sync_bit, so channel-bit LLR = energy of
    # {tones with channel bit 0} minus {tones with channel bit 1}.
    channel_llr = (np.maximum(tone_energy[:, 0], tone_energy[:, 1])
                  - np.maximum(tone_energy[:, 2], tone_energy[:, 3]))
    deinterleaved = _deinterleave_llr(channel_llr)

    message_bits = decode_symbols(deinterleaved, beam_width=beam_width)
    if message_bits is None:
        return None
    try:
        return unpack_message(message_bits)
    except (ValueError, IndexError):
        return None
