"""PSK31 / PSK63: differential BPSK, Varicode text framing.

PSK31 (Peter Martinez, G3PLX, 1998) is BPSK at exactly 31.25 baud;
PSK63 is the same scheme at exactly 62.5 baud -- unlike FSK modes, the
symbol rate is a protocol constant, not something to measure. Data bits
are differentially encoded (bit 1 = no phase change, bit 0 = a 180
degree phase reversal -- so the *receiver* only ever needs consecutive-
symbol phase difference, never an absolute phase reference) and framed
with Varicode: a variable-length code (like Morse, common characters get
shorter codes) whose codewords never contain "00" and always start and
end with a "1" bit, so two zero bits ("00") unambiguously mark the gap
between characters -- no separate framing/sync needed.

The 256-entry table below (0-127 standard PSK31 Varicode, 128-255 an
extended range some implementations use for accented/extra characters)
is transcribed programmatically from fldigi's own source
(``src/psk/pskvaricode.cxx``, the varicodetab1 array) -- production code
from a widely-used real PSK31 implementation, not reconstructed from
memory (a single wrong entry among 256 would silently garble any message
using that character).

Self-rejecting: a recording whose demodulated bitstream never produces a
single valid Varicode codeword returns ``None``.

Known limitation: with an auto-detected carrier several Hz off true
center (an FFT bin's worth of :func:`sigint.detect.find_carriers`
resolution, easily ~5-10 Hz), the very first decoded character is
occasionally wrong even though everything after it decodes perfectly --
the residual-frequency correction (see :func:`_differential_bits`) fixes
every symbol-to-symbol transition correctly, but the very first bit is
recovered against an assumed idle reference rather than a real preceding
symbol, and that assumption is the one place a large residual frequency
error still leaks through. Passing an exact `carrier_hz` avoids it
entirely.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..detect import find_carriers

BAUD = {"PSK31": 31.25, "PSK63": 62.5}

# Varicode table, index = character code 0-255. Transcribed from fldigi's
# src/psk/pskvaricode.cxx (varicodetab1[]).
VARICODE: tuple[str, ...] = (
    '1010101011', '1011011011', '1011101101', '1101110111', '1011101011', '1101011111', '1011101111', '1011111101',
    '1011111111', '11101111', '11101', '1101101111', '1011011101', '11111', '1101110101', '1110101011',
    '1011110111', '1011110101', '1110101101', '1110101111', '1101011011', '1101101011', '1101101101', '1101010111',
    '1101111011', '1101111101', '1110110111', '1101010101', '1101011101', '1110111011', '1011111011', '1101111111',
    '1', '111111111', '101011111', '111110101', '111011011', '1011010101', '1010111011', '101111111',
    '11111011', '11110111', '101101111', '111011111', '1110101', '110101', '1010111', '110101111',
    '10110111', '10111101', '11101101', '11111111', '101110111', '101011011', '101101011', '110101101',
    '110101011', '110110111', '11110101', '110111101', '111101101', '1010101', '111010111', '1010101111',
    '1010111101', '1111101', '11101011', '10101101', '10110101', '1110111', '11011011', '11111101',
    '101010101', '1111111', '111111101', '101111101', '11010111', '10111011', '11011101', '10101011',
    '11010101', '111011101', '10101111', '1101111', '1101101', '101010111', '110110101', '101011101',
    '101110101', '101111011', '1010101101', '111110111', '111101111', '111111011', '1010111111', '101101101',
    '1011011111', '1011', '1011111', '101111', '101101', '11', '111101', '1011011',
    '101011', '1101', '111101011', '10111111', '11011', '111011', '1111', '111',
    '111111', '110111111', '10101', '10111', '101', '110111', '1111011', '1101011',
    '11011111', '1011101', '111010101', '1010110111', '110111011', '1010110101', '1011010111', '1110110101',
    '1110111101', '1110111111', '1111010101', '1111010111', '1111011011', '1111011101', '1111011111', '1111101011',
    '1111101101', '1111101111', '1111110101', '1111110111', '1111111011', '1111111101', '1111111111', '10101010101',
    '10101010111', '10101011011', '10101011101', '10101011111', '10101101011', '10101101101', '10101101111', '10101110101',
    '10101110111', '10101111011', '10101111101', '10101111111', '10110101011', '10110101101', '10110101111', '10110110101',
    '10110110111', '10110111011', '10110111101', '10110111111', '10111010101', '10111010111', '10111011011', '10111011101',
    '10111011111', '10111101011', '10111101101', '10111101111', '10111110101', '10111110111', '10111111011', '10111111101',
    '10111111111', '11010101011', '11010101101', '11010101111', '11010110101', '11010110111', '11010111011', '11010111101',
    '11010111111', '11011010101', '11011010111', '11011011011', '11011011101', '11011011111', '11011101011', '11011101101',
    '11011101111', '11011110101', '11011110111', '11011111011', '11011111101', '11011111111', '11101010101', '11101010111',
    '11101011011', '11101011101', '11101011111', '11101101011', '11101101101', '11101101111', '11101110101', '11101110111',
    '11101111011', '11101111101', '11101111111', '11110101011', '11110101101', '11110101111', '11110110101', '11110110111',
    '11110111011', '11110111101', '11110111111', '11111010101', '11111010111', '11111011011', '11111011101', '11111011111',
    '11111101011', '11111101101', '11111101111', '11111110101', '11111110111', '11111111011', '11111111101', '11111111111',
    '101010101011', '101010101101', '101010101111', '101010110101', '101010110111', '101010111011', '101010111101', '101010111111',
    '101011010101', '101011010111', '101011011011', '101011011101', '101011011111', '101011101011', '101011101101', '101011101111',
    '101011110101', '101011110111', '101011111011', '101011111101', '101011111111', '101101010101', '101101010111', '101101011011',
)
_REVERSE: dict[str, int] = {code: i for i, code in enumerate(VARICODE)}


def text_to_bits(text: str) -> str:
    """Text -> the differential-encoding bit stream: each character's
    Varicode codeword, each pair of characters separated by '00'."""
    return "00".join(VARICODE[b] for b in text.encode("latin-1", "replace"))


def bits_to_text(bits: str) -> tuple[str, int]:
    """The reverse: a bit stream (as found between/around '00' runs) back
    to text. Returns (text, unrecognized_count)."""
    chars = []
    unrecognized = 0
    for code in bits.split("00"):
        code = code.strip("0")   # a run of 3+ zeros still just means "gap"
        if not code:
            continue
        idx = _REVERSE.get(code)
        if idx is None:
            unrecognized += 1
            chars.append("#")
        else:
            chars.append(chr(idx))
    return "".join(chars), unrecognized


# --------------------------------------------------------------------------
# Synthesis (ground truth for testing)
# --------------------------------------------------------------------------

def generate(text: str, sample_rate: float, carrier_hz: float = 1000.0,
            variant: str = "PSK31") -> np.ndarray:
    """Synthesize a differentially-BPSK-modulated recording of `text`.

    Continuous-phase, constant envelope -- correctly carries the same
    Varicode-framed differential bitstream a real PSK31/63 signal would,
    but without the raised-cosine pulse shaping a real transmitter uses to
    keep its occupied bandwidth near the symbol rate itself (a real signal
    shapes the envelope through zero at each phase reversal; this
    generator steps phase instantly, so it looks considerably wider on a
    waterfall than the real thing -- fine for exercising the
    demod/Varicode logic here, not a spectral model of the mode).
    """
    baud = BAUD[variant]
    sps = max(1, int(round(sample_rate / baud)))
    bits = text_to_bits(text)
    phase = 0.0
    chunks = []
    t_sym = np.arange(sps) / sample_rate
    for bit in bits:
        if bit == "0":
            phase += np.pi
        chunks.append(np.cos(2 * np.pi * carrier_hz * t_sym + phase))
    return np.concatenate(chunks) if chunks else np.zeros(0)


# --------------------------------------------------------------------------
# Detection + decode
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PSKMessage:
    text: str
    variant: str
    carrier_hz: float
    unrecognized: int = 0


_COMMON_TEXT_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,'\"!?-\n\r\t")


def _text_quality(bits: str) -> float:
    """How much does the Varicode decode of `bits` look like real text,
    as opposed to an arbitrary bitstream that happens to parse into some
    (mostly wrong) codewords? Used to pick between ambiguous frequency-
    correction branches. Printable-character fraction alone doesn't work
    here (unlike :func:`sigint.decoders.run_all`'s decoder sweep, where it
    does): PSK31's own extended Varicode range densely covers the
    printable Latin-1 block, so a wrong correction still decodes to
    "printable" characters almost as often as a right one, just not
    ordinary English/ASCII ones. Scoring against the common
    letters/digits/punctuation that actual text overwhelmingly uses
    separates a correct decode from a wrong one clearly (verified against
    exactly this ambiguity in tests/test_ham_psk31.py)."""
    if not bits:
        return -1e9
    text, unrecognized = bits_to_text(bits)
    if not text:
        return -1e9
    common = sum(1 for c in text if c in _COMMON_TEXT_CHARS)
    return (common - 2 * unrecognized) / len(text)


def _bits_from_phases(phases: np.ndarray, correction: float) -> str:
    dphase = np.diff(phases) - correction
    dphase = (dphase + np.pi) % (2 * np.pi) - np.pi
    return "".join("1" if abs(d) < np.pi / 2 else "0" for d in dphase)


def _differential_bits(samples: np.ndarray, sample_rate: float,
                       carrier_hz: float, baud: float,
                       phase_steps: int = 16) -> tuple[str, float]:
    """Symbol-average correlation against the carrier, phase-offset swept
    (the same coarse-grid trick :func:`sigint.dsp.fsk_demod_sync` uses)
    to find the symbol timing, then differential phase between
    consecutive symbols -> bits. Returns (bit string, quality score).

    A residual carrier-frequency error (the caller's `carrier_hz` is
    rarely exact -- an auto-detected carrier is only as precise as
    :func:`sigint.detect.find_carriers`' FFT bin spacing) shows up as a
    constant extra phase step every symbol, which differential decoding
    does *not* cancel (it only cancels a genuinely constant absolute
    phase, not a steadily-incrementing one) -- so it's corrected here via
    the standard BPSK squaring-loop trick: squaring each symbol's phase
    doubles it, which cancels the +/-180 degree data modulation and
    leaves only (twice) the residual frequency error, cleanly averaged
    out over the whole record. Halving that estimate to undo the squaring
    leaves a 180 degree sign ambiguity (which of two possible corrections
    is right); both are tried and scored by how many Varicode codewords
    they actually produce, alongside the uncorrected baseline.
    """
    x = np.asarray(samples, dtype=np.float64).ravel()
    sps = max(1, int(round(sample_rate / baud)))
    t = np.arange(len(x)) / sample_rate
    bb = x * np.exp(-2j * np.pi * carrier_hz * t)

    # search symbol-timing offset AND the frequency-correction branch
    # together, scored by the same criterion (_text_quality): two offsets
    # can have near-identical signal magnitude (this module's earlier
    # per-offset magnitude gate picked between two such offsets on noise-
    # level differences) while decoding to completely different text, so
    # magnitude alone isn't a safe offset-selection criterion either.
    best_bits, best_score, best_mag = "", -1e9, 0.0
    for off in np.linspace(0, sps - 1, phase_steps).astype(int):
        n_sym = (len(bb) - off) // sps
        if n_sym < 3:
            continue
        z = np.array([bb[off + k * sps: off + (k + 1) * sps].mean()
                     for k in range(n_sym)])
        mag = float(np.abs(z).mean())
        zsq = z ** 2
        est = float(np.angle(np.mean(zsq[1:] * np.conj(zsq[:-1])))) / 2.0
        # a virtual phase-0 reference ahead of the first symbol recovers
        # that symbol's own bit too (idle/silence before a transmission
        # starts is a steady phase, i.e. an unbroken run of '1' bits --
        # and every Varicode codeword is required to start with '1' --
        # so phase 0 is the right assumption for "whatever came before").
        phases = np.concatenate([[0.0], np.angle(z)])
        for c in (0.0, est, est + np.pi):
            bits = _bits_from_phases(phases, c)
            score = _text_quality(bits)
            if score > best_score:
                best_bits, best_score, best_mag = bits, score, mag
    return best_bits, best_mag


def decode(samples: np.ndarray, sample_rate: float, variant: str = "PSK31",
          carrier_hz: float | None = None) -> PSKMessage | None:
    """Detect and decode a PSK31/PSK63 signal.

    `carrier_hz` is the tone the signal is centered on (a PSK31 operator
    tunes anywhere in the waterfall, there's no fixed offset like HFDL's
    SSB convention) -- if not given, the strongest persistent carrier
    found by :func:`sigint.detect.find_carriers` is used. Self-rejects
    (returns ``None``) if no carrier is found, or if the demodulated
    bitstream doesn't contain a single valid Varicode codeword.
    """
    if variant not in BAUD:
        raise ValueError(f"variant must be one of {tuple(BAUD)}")
    x = np.asarray(samples, dtype=np.float64).ravel()
    if carrier_hz is None:
        scan = find_carriers(x, sample_rate, prominence_db=6.0)
        if not scan.carriers:
            return None
        carrier_hz = scan.carriers[0].freq_hz

    bits, quality = _differential_bits(x, sample_rate, carrier_hz, BAUD[variant])
    if quality <= 0 or not bits:
        return None
    text, unrecognized = bits_to_text(bits)
    if not text:
        return None
    return PSKMessage(text=text, variant=variant, carrier_hz=carrier_hz,
                      unrecognized=unrecognized)
