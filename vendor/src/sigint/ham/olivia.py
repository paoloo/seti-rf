"""Olivia / Contestia: incoherent MFSK with an optional Walsh-Hadamard FEC.

Both modes share the same physical layer -- `tones` orthogonally-spaced
FSK tones across `bandwidth_hz`, 7-bit-ASCII characters -- and differ only
in whether Olivia's forward error correction layer runs on top. That FEC
(:class:`MFSKConfig` with ``fec=True``) is a real biorthogonal (64,7)
Walsh-Hadamard block code: each 7-bit character is placed as a single +1
or -1 impulse into a 64-point buffer (the impulse position is the low 6
bits, its sign the 7th) and inverse-Hadamard-transformed into a
pseudo-noise-like 64-symbol codeword; ``bits_per_symbol`` (= log2(tones))
independent characters are interleaved into the same 64 symbol slots by
rotating which output bit position each character's transform occupies at
each time step, and a fixed 64-bit scrambling constant whitens each
character's own codeword before transmission. This whole FEC/interleave
structure -- the Fast Hadamard Transform butterfly, the
``0xE257E6D0291574EC`` scrambling constant, the per-character rotation --
is transcribed from ``gmfsk``'s real Olivia implementation (Pawel
Jalocha's original algorithm, ``src/olivia/mfsk.h``'s ``MFSK_Encoder``/
``MFSK_SoftDecoder`` classes), not reconstructed from memory: a wrong
scrambling constant or rotation offset would silently produce a codec
that only ever talks to itself.

Contestia is Olivia's ``fec=False`` sibling: the same tone layout, no
Hadamard coding, characters packed directly into the tone bitstream.
"MFSK" configurations here mean the same generic incoherent-tone core at
MFSK-family tone counts -- not a byte-for-byte clone of the original,
separately-specified 1999 MFSK16 mode (its own bit-interleave and
character set), which is out of scope here.

Demodulation is hard-decision (each symbol resolved to its single
strongest tone before FEC recombination) rather than the soft per-tone-
energy combining a real weak-signal receiver uses -- correct and fully
self-testing on a clean or moderately noisy channel, just with less
processing gain at very low SNR than a production decoder. Self-
rejecting: a block that produces no in-range character reports it in
`unrecognized` rather than fabricating text.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_SCRAMBLE = 0xE257E6D0291574EC
_BITS_PER_CHARACTER = 7                 # fixed by the protocol
_SYMBOLS_PER_BLOCK = 1 << (_BITS_PER_CHARACTER - 1)   # 64, always


@dataclass(frozen=True)
class MFSKConfig:
    name: str
    tones: int              # must be a power of two
    bandwidth_hz: float
    fec: bool = True

    def __post_init__(self):
        if self.tones & (self.tones - 1) or self.tones < 2:
            raise ValueError("tones must be a power of two >= 2")

    @property
    def bits_per_symbol(self) -> int:
        return self.tones.bit_length() - 1

    @property
    def tone_spacing_hz(self) -> float:
        return self.bandwidth_hz / self.tones

    @property
    def symbol_s(self) -> float:
        return 1.0 / self.tone_spacing_hz


OLIVIA_8_250 = MFSKConfig("Olivia 8/250", 8, 250.0, fec=True)
OLIVIA_16_500 = MFSKConfig("Olivia 16/500", 16, 500.0, fec=True)
OLIVIA_32_1000 = MFSKConfig("Olivia 32/1000", 32, 1000.0, fec=True)  # the common default
OLIVIA_64_2000 = MFSKConfig("Olivia 64/2000", 64, 2000.0, fec=True)
CONTESTIA_4_125 = MFSKConfig("Contestia 4/125", 4, 125.0, fec=False)
CONTESTIA_8_250 = MFSKConfig("Contestia 8/250", 8, 250.0, fec=False)
MFSK16 = MFSKConfig("MFSK16 (generic)", 16, 250.0, fec=False)


def _fht(buf: np.ndarray) -> np.ndarray:
    """In-place forward Fast Hadamard Transform (Jalocha's butterfly)."""
    n = len(buf)
    step = 1
    while step < n:
        for ptr in range(0, n, 2 * step):
            b1 = buf[ptr:ptr + step].copy()
            b2 = buf[ptr + step:ptr + 2 * step]
            buf[ptr:ptr + step] = b1 + b2
            buf[ptr + step:ptr + 2 * step] = b2 - b1
        step *= 2
    return buf


def _ifht(buf: np.ndarray) -> np.ndarray:
    """In-place inverse Fast Hadamard Transform."""
    n = len(buf)
    step = n // 2
    while step >= 1:
        for ptr in range(0, n, 2 * step):
            b1 = buf[ptr:ptr + step].copy()
            b2 = buf[ptr + step:ptr + 2 * step]
            buf[ptr:ptr + step] = b1 - b2
            buf[ptr + step:ptr + 2 * step] = b1 + b2
        step //= 2
    return buf


def _scramble_signs(freq_bit: int) -> np.ndarray:
    """+/-1 per time slot for this character's scrambling (CodeOffset =
    freq_bit * 13, per the reference encoder)."""
    code_bit = (freq_bit * 13) % _SYMBOLS_PER_BLOCK
    signs = np.empty(_SYMBOLS_PER_BLOCK)
    for t in range(_SYMBOLS_PER_BLOCK):
        signs[t] = -1.0 if (_SCRAMBLE >> code_bit) & 1 else 1.0
        code_bit = (code_bit + 1) % _SYMBOLS_PER_BLOCK
    return signs


_SCRAMBLE_CACHE: dict[int, np.ndarray] = {}


def _scramble(freq_bit: int) -> np.ndarray:
    if freq_bit not in _SCRAMBLE_CACHE:
        _SCRAMBLE_CACHE[freq_bit] = _scramble_signs(freq_bit)
    return _SCRAMBLE_CACHE[freq_bit]


def _encode_character(char7: int) -> np.ndarray:
    """One 7-bit character -> its 64-value Hadamard codeword (unscrambled)."""
    buf = np.zeros(_SYMBOLS_PER_BLOCK)
    if char7 < _SYMBOLS_PER_BLOCK:
        buf[char7] = 1.0
    else:
        buf[char7 - _SYMBOLS_PER_BLOCK] = -1.0
    return _ifht(buf)


def _encode_block_fec(chars7: list[int], bits_per_symbol: int) -> np.ndarray:
    """`bits_per_symbol` characters -> 64 tone indices (0..tones-1)."""
    tones = np.zeros(_SYMBOLS_PER_BLOCK, dtype=np.int64)
    for freq_bit, char7 in enumerate(chars7):
        codeword = _encode_character(char7) * _scramble(freq_bit)
        neg = codeword < 0
        out_bit_pos = (freq_bit + np.arange(_SYMBOLS_PER_BLOCK)) % bits_per_symbol
        tones |= (neg.astype(np.int64) << out_bit_pos)
    return tones


def _decode_block_fec(tones: np.ndarray, bits_per_symbol: int) -> list[int]:
    """64 hard-decision tone indices -> `bits_per_symbol` characters."""
    out = []
    for freq_bit in range(bits_per_symbol):
        out_bit_pos = (freq_bit + np.arange(_SYMBOLS_PER_BLOCK)) % bits_per_symbol
        bit = (tones >> out_bit_pos) & 1
        buf = np.where(bit == 1, -1.0, 1.0) * _scramble(freq_bit)
        buf = _fht(buf)
        peak_idx = int(np.argmax(np.abs(buf)))
        char7 = peak_idx if buf[peak_idx] >= 0 else peak_idx + _SYMBOLS_PER_BLOCK
        out.append(char7)
    return out


# --------------------------------------------------------------------------
# Text <-> tone-index streams
# --------------------------------------------------------------------------

def text_to_tones(text: str, cfg: MFSKConfig) -> np.ndarray:
    """7-bit-ASCII text -> a stream of tone indices (0..cfg.tones-1)."""
    chars = [b & 0x7F for b in text.encode("ascii", "replace")]
    if not cfg.fec:
        bits = "".join(format(c, "07b") for c in chars)
        pad = (-len(bits)) % cfg.bits_per_symbol
        bits += "0" * pad
        return np.array([int(bits[i:i + cfg.bits_per_symbol], 2)
                         for i in range(0, len(bits), cfg.bits_per_symbol)])
    pad = (-len(chars)) % cfg.bits_per_symbol
    chars = chars + [0x20] * pad   # pad with spaces (a harmless character)
    blocks = [chars[i:i + cfg.bits_per_symbol]
             for i in range(0, len(chars), cfg.bits_per_symbol)]
    return np.concatenate([_encode_block_fec(b, cfg.bits_per_symbol)
                           for b in blocks])


def tones_to_text(tones: np.ndarray, cfg: MFSKConfig) -> tuple[str, int]:
    """The reverse: a hard-decision tone-index stream -> (text,
    unrecognized_count). `unrecognized` counts characters recovered
    outside the printable-ASCII range (7 bits covers 0-127; a real
    character is essentially always within 0x09-0x7E)."""
    if not cfg.fec:
        bits = "".join(format(int(t), f"0{cfg.bits_per_symbol}b") for t in tones)
        nchars = len(bits) // 7
        chars = [int(bits[i * 7:i * 7 + 7], 2) for i in range(nchars)]
    else:
        chars = []
        for i in range(0, len(tones) - _SYMBOLS_PER_BLOCK + 1, _SYMBOLS_PER_BLOCK):
            chars.extend(_decode_block_fec(tones[i:i + _SYMBOLS_PER_BLOCK],
                                           cfg.bits_per_symbol))
    unrecognized = sum(1 for c in chars if not (0x09 <= c <= 0x7E))
    text = "".join(chr(c) for c in chars)
    return text, unrecognized


# --------------------------------------------------------------------------
# Synthesis (ground truth for testing)
# --------------------------------------------------------------------------

def generate(text: str, sample_rate: float, cfg: MFSKConfig,
            base_hz: float = 500.0) -> np.ndarray:
    """Synthesize a continuous-phase incoherent-MFSK recording of `text`."""
    tones = text_to_tones(text, cfg)
    sps = max(1, int(round(sample_rate * cfg.symbol_s)))
    phase = 0.0
    chunks = []
    for tone in tones:
        f = base_hz + tone * cfg.tone_spacing_hz
        t = np.arange(sps) / sample_rate
        chunks.append(np.cos(2 * np.pi * f * t + phase))
        phase = (phase + 2 * np.pi * f * sps / sample_rate) % (2 * np.pi)
    return np.concatenate(chunks) if chunks else np.zeros(0)


# --------------------------------------------------------------------------
# Detection + decode
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MFSKMessage:
    text: str
    config: str            # cfg.name
    unrecognized: int = 0


def decode(samples: np.ndarray, sample_rate: float, cfg: MFSKConfig,
          base_hz: float = 500.0, phase_steps: int = 8) -> MFSKMessage | None:
    """Detect and decode an Olivia/Contestia/MFSK-family signal already
    known to be at `base_hz`..`base_hz + cfg.bandwidth_hz` (this family
    has no fixed waterfall offset -- an operator tunes it manually, so
    there's no carrier to auto-detect the way :mod:`sigint.hfdl` or
    :mod:`sigint.ham.psk31` can).

    Self-rejecting: fewer than one full block's worth of symbols (or the
    resulting decode is empty) returns ``None``.
    """
    x = np.asarray(samples, dtype=np.float64).ravel()
    sps = max(1, int(round(sample_rate * cfg.symbol_s)))
    min_symbols = _SYMBOLS_PER_BLOCK if cfg.fec else 1

    best_tones, best_quality = None, -1.0
    for off in np.linspace(0, sps - 1, phase_steps).astype(int):
        n_sym = (len(x) - off) // sps
        if n_sym < min_symbols:
            continue
        t = np.arange(sps) / sample_rate
        refs = np.array([np.exp(-2j * np.pi * (base_hz + k * cfg.tone_spacing_hz) * t)
                         for k in range(cfg.tones)])
        tones = np.empty(n_sym, dtype=np.int64)
        mags = np.empty(n_sym)
        for k in range(n_sym):
            seg = x[off + k * sps: off + (k + 1) * sps]
            corr = np.abs(refs @ seg)
            tones[k] = int(np.argmax(corr))
            mags[k] = float(corr[tones[k]]) / sps
        quality = float(mags.mean())
        if quality > best_quality:
            best_tones, best_quality = tones, quality

    if best_tones is None or best_quality <= 0:
        return None
    if cfg.fec:
        usable = (len(best_tones) // _SYMBOLS_PER_BLOCK) * _SYMBOLS_PER_BLOCK
        best_tones = best_tones[:usable]
    text, unrecognized = tones_to_text(best_tones, cfg)
    if not text:
        return None
    return MFSKMessage(text=text, config=cfg.name, unrecognized=unrecognized)
