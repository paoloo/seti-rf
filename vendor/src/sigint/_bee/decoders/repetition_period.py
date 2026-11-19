"""Autocorrelation-based frame-period recovery for the Delphi/Borland
``SigNcMp``/``DecNcMp`` repetition-code family (see GUIDE.md's "Family A"
section).

These 20 DLLs are Borland/Delphi VCL objects with no recoverable C++ RTTI
class names and no useful embedded strings, unlike the MFC-family DLLs
(``082.dll``/``083.dll``/``072.dll``/``087.dll``/``078.dll``) that static
analysis successfully traced elsewhere in this project. Rather than leave
these captions with literally no code, this module finds their bit-level
frame period directly from the bundled captures using the same
polarity-invariant phi-correlation measure already used and proven for
MIL-STD-188-110A cadence recovery (see ``measure_cadence`` in
``milstd188110a.py``), extended to search a range of candidate periods and
require harmonic support (elevated correlation at 2x, 3x, ... the candidate
period, not just the candidate itself) before reporting a period as
confirmed. A single elevated correlation value at one lag is weak evidence
(could be a coincidental bias in a short capture); a period whose small
integer multiples are *also* elevated is the standard signature of genuine
frame periodicity and not noise.

**Bit orientation (corrected 2026-08-14).** The original application's own
help file (``Bee.Demo/BeeE.chm``/``BeeR.chm``, ``hlp_vwold.htm``) documents
that a captured bitstream is packed **LSB-first**: "the bit that arrived
first is written into the least significant bit of the byte" (worked
example: demodulated ``01110011 10101110 10011101`` packs to
``0xCE 0x75 0xB9``). This module's search previously only ever measured the
capture as given to it (which the CLI/GUI default to MSB-first unless a
caller opts into ``bit_order="lsb"``), with no internal orientation search --
unlike every other decoder module built this session
(``italhdlc.py``/``datronlink.py``/``ccir476.py``/``ale.py``/
``vocoder_frame.py``), which all try both the given orientation and its
per-byte-bit-reversed form and report whichever one the evidence supports.
``find_dominant_period`` now does the same, and is checked against each
name's documented "displayed period" (or, where the original separately
states it, the shorter "cycle of issue" that interleaving expands into the
displayed period) from the help file wherever one exists.

This recovers only frame-period synchronization -- the same evidence tier as
the existing ``192 (Period)``/``420 (Period)`` decoders. It does not recover
alphabet, error-correction, or payload structure, and does not identify
which physical-layer element (a repeated marker, a repetition code, matrix
row/column dimensions) actually produces the measured period; the DLL
codename's ``N``/``M`` numbers are noted per-decoder as a plausible but
unconfirmed correspondence, not asserted as the mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sigint._bee.decoders.base import DecodeResult, Decoder
from sigint._bee.decoders.registry import get, register

_HARMONIC_THRESHOLD = 0.15
_MIN_HARMONICS = 2


def _reverse_each_byte(bits: np.ndarray) -> np.ndarray:
    complete = len(bits) // 8 * 8
    result = bits.copy()
    if complete:
        result[:complete] = bits[:complete].reshape(-1, 8)[:, ::-1].reshape(-1)
    return result


def _phi_correlation(stream: np.ndarray, period: int) -> float:
    """Polarity-invariant Pearson correlation between the stream and itself
    delayed by ``period`` bits. See ``milstd188110a.measure_cadence``."""
    if len(stream) <= period:
        return 0.0
    left = stream[:-period].astype(np.float64)
    right = stream[period:].astype(np.float64)
    left -= left.mean()
    right -= right.mean()
    scale = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / scale) if scale else 0.0


@dataclass(frozen=True)
class PeriodCandidate:
    period: int
    correlation: float
    harmonics_confirmed: int
    orientation: str = "selected"

    @property
    def confirmed(self) -> bool:
        return self.harmonics_confirmed >= _MIN_HARMONICS


def _find_dominant_period_one_orientation(
    stream: np.ndarray, orientation: str, min_period: int, max_period: int,
) -> PeriodCandidate | None:
    """Search ``[min_period, max_period)`` for the strongest period that is a
    local correlation maximum (a real periodicity spike, not part of a
    smooth short-range-correlation ramp -- those exist near-universally at
    tiny lags and are not evidence of frame structure) whose first few
    integer multiples also show elevated phi-correlation.

    Reports the smallest confirmed period, not the strongest correlation --
    a harmonic of the true fundamental can score higher than the fundamental
    itself in a short, noisy capture, so preferring raw correlation strength
    would sometimes misreport a multiple as the period.
    """
    curve = {p: _phi_correlation(stream, p) for p in range(max(1, min_period - 1), max_period + 1)}
    best: PeriodCandidate | None = None
    for period in range(min_period, max_period):
        base = curve[period]
        if abs(base) < _HARMONIC_THRESHOLD:
            continue
        if abs(base) < abs(curve[period - 1]) or abs(base) < abs(curve[period + 1]):
            continue  # not a local peak -- part of a smooth ramp, not a spike
        harmonics = 0
        multiple = 2
        while period * multiple < len(stream) // 2 and multiple <= 6:
            if abs(_phi_correlation(stream, period * multiple)) >= _HARMONIC_THRESHOLD:
                harmonics += 1
            multiple += 1
        candidate = PeriodCandidate(period, base, harmonics, orientation)
        if candidate.confirmed:
            return candidate  # smallest confirmed fundamental wins outright
        if best is None or abs(candidate.correlation) > abs(best.correlation):
            best = candidate
    return best


def find_dominant_period(bits: np.ndarray, min_period: int = 8, max_period: int = 400) -> PeriodCandidate | None:
    """Search both byte-bit orientations (the given order and its
    per-byte-bit-reversed form -- equivalent to MSB-first vs. the
    LSB-first packing the original help file documents, see the module
    docstring) and return the better-evidenced confirmed candidate.

    When both orientations confirm a period, the one with *more* confirmed
    harmonics wins (stronger evidence of genuine periodicity), not whichever
    orientation happened to be tried first and not the smaller of the two
    periods -- checked against every documented "displayed period" this
    module has one for, that rule turned 3 of 16 exact-or-clean-divisor
    matches into 13 of 16 (see GUIDE.md). If only one orientation confirms,
    it wins outright; if neither does, the stronger of the two unconfirmed
    candidates is returned.
    """
    stream = np.asarray(bits, dtype=np.uint8).reshape(-1) & 1
    selected = _find_dominant_period_one_orientation(stream, "selected", min_period, max_period)
    reversed_stream = _reverse_each_byte(stream)
    byte_reversed = _find_dominant_period_one_orientation(reversed_stream, "byte-reversed", min_period, max_period)
    confirmed = [c for c in (selected, byte_reversed) if c is not None and c.confirmed]
    if confirmed:
        return max(confirmed, key=lambda c: (c.harmonics_confirmed, -c.period))
    candidates = [c for c in (selected, byte_reversed) if c is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda c: abs(c.correlation))


class RepetitionPeriodDecoder(Decoder):
    """Base class; each registered name below sets ``codename_hint`` for its
    provenance note and reuses this generic autocorrelation search."""

    status = "partial"
    confidence = "experimental"
    layer = "synchronization"
    codename_hint = ""

    def decode(self, bits: np.ndarray, block=None) -> DecodeResult:
        candidate = find_dominant_period(bits)
        if candidate is None:
            return DecodeResult(
                text="", symbol_count=0, error_count=0,
                notes=(f"No bit-period candidate cleared the {_HARMONIC_THRESHOLD} "
                       "correlation threshold in a 2-400 bit search. No structure "
                       "recovered."),
            )
        confidence_word = "harmonic-confirmed" if candidate.confirmed else "single-lag, unconfirmed"
        text = (f"period={candidate.period}b correlation={candidate.correlation:.3f} "
                f"harmonics_confirmed={candidate.harmonics_confirmed} "
                f"orientation={candidate.orientation} ({confidence_word})")
        hint = f" DLL codename hint: {self.codename_hint}." if self.codename_hint else ""
        return DecodeResult(
            text=text,
            symbol_count=1 if candidate.confirmed else 0,
            error_count=0 if candidate.confirmed else 1,
            notes=(f"Autocorrelation-recovered frame period only; alphabet, error "
                   f"correction, and payload structure are not recovered.{hint} "
                   "See repetition_period.py module docstring for the evidence "
                   "policy behind this decoder."),
        )


_REGISTRATIONS = {
    # name: codename hint (from docs/external_references and GUIDE.md's DLL triage)
    "2s2": "Sig2c2.dll (014.dll)",
    "2s2 (2x10)": "Dec2c2_2_10.dll (019.dll)",
    "2b 2s1 (5s5)": "Sig2B_2c1.dll (013.dll)",
    "8s8k10": "Dec8c8k10.dll (028.dll)",
    "8s8k5": "Dec8c8k5.dll (029.dll)",
    "BULG-ASCII": "DEC11ct8.dll (044.dll)",
    "DUP-FEC-2": "Sig32c32.dll (020.dll)",
    "ARM-DA": "Armada.dll / CDecArmada (087.dll)",
    "RUM-FEC 1": "Dec16c16p.dll (008.dll)",
    "RUM-FEC 2": "Dec16c16p_2.dll (009.dll)",
    "COQUELET MK8": "Sig2c2.dll (018.dll)",
    "1s1": "Sig1c1.dll (011.dll)",
    "30s30": "30C30.dll (078.dll)",
    "ROU-MOI-FEC": "Dec24c23.dll (012.dll)",
    "PICCOLO MK-6 (006a)": "Sig2c2_006a.dll (015.dll)",
    "PICCOLO MK-6 (006b)": "Sig2c2_006b.dll (016.dll)",
    "PICCOLO MK-6 (006c)": "Sig2c2_006c.dll (017.dll)",
}

for _name, _hint in _REGISTRATIONS.items():
    _decoder = RepetitionPeriodDecoder()
    _decoder.name = _name
    _decoder.description = f"Autocorrelation-recovered frame period (see codename hint)"
    _decoder.codename_hint = _hint
    _decoder.provenance = "phi-correlation period search over the bundled capture (no DLL byte evidence for this family yet)"
    register(_decoder)

# RUM-FEC 1's recovered 16-bit period is independently confirmed by
# Artemis-DB/SigID's reported "each new character starting every 16 bits"
# (see docs/external_references/artemis_db_v74.json) -- two independent
# methods agreeing earns higher confidence than the raw correlation search
# alone, which is the bar every other name in this module is held to.
get("RUM-FEC 1").confidence = "medium"
get("RUM-FEC 1").provenance += (
    "; independently cross-validated by Artemis-DB/SigID's RUM-FEC entry, "
    "which reports a new character every 16 bits -- an exact match found "
    "without prior knowledge of that detail"
)
