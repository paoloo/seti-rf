"""Frame periods recovered from the original application's own help file and
verified against the bundled captures.

`Bee.Demo/BeeE.chm` and `Bee.Demo/BeeR.chm` (the original English/Russian
HTML Help files shipped with the application, read-only bundled evidence) each
contain one page per legacy decoder under `hlp_vwold_*.htm`. Those pages state
the transmission cycle and the "displayed period" in bits for every HF
protocol the tool handled, including several names that had no recovered
algorithm at all. That is primary documentation from the original authors, not
a caption-based guess -- but the project's evidence rules require it to agree
with a bundled capture before anything is claimed, which is what this module
does.

Two things in `hlp_vwold.htm` (the "HF Protocols" index page) matter for every
decoder in the project, not just these three:

1. **Bit packing.** It specifies exactly how a `Signals/*.bin` capture stores a
   demodulated bitstream: "the demodulated bit sequence is packed into bytes,
   with the bit that arrived first written into the least significant bit of
   the byte", with a worked example (`01110011 10101110 10011101` packs to
   `0xCE 0x75 0xB9`). The transmitted bit order is therefore recovered by
   unpacking each byte **LSB-first**, not MSB-first. This is measurable, not
   just documentary: reading LSB-first moves `ALIS`'s documented 111-bit cycle
   from correlation rank 34 to rank 1, `1s1`'s documented 30-bit cycle from
   rank 206 to rank 2, and `BULG-ASCII`'s documented 66-bit cycle from rank 317
   to rank 1. See GUIDE.md's "Original capture bit packing" section.

2. **Which protocols even have a findable period.** The same page splits the
   catalog into "protocols with a determinable period" and "protocols with
   non-determinable periods", and puts `12s12p`, `AUTOSPEC`, `MEROD-USA`,
   `MEROD-FIN`, `MEROD-IND` (plus AX.25, Sitor-B and Sitor-FEC) in the second
   group, noting they "can be identified only by their structure, or by brute
   force". That independently explains this project's own repeated failures to
   autocorrelate a period out of the MEROD and `12s12p` captures.

`AUTOSPEC` is the interesting exception: the original tool classes it as
period-non-determinable because its *displayed* period (40 bits) does not
autocorrelate, but the same help page also states the primary signal has a
10-bit transmission cycle, and that 10-bit cycle is overwhelmingly present in
the capture once the documented LSB-first packing is used.

What this module claims is only the frame period -- the same evidence tier as
`repetition_period.py` and the `192 (Period)`/`420 (Period)` decoders. Alphabet,
error correction, and payload are not recovered for any of these three names.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

from sigint._bee.decoders.base import DecodeResult, Decoder
from sigint._bee.decoders.registry import register

_SEARCH_LOW = 8
_SEARCH_HIGH = 400
#: `p` below which the multiples-of-period concentration is treated as real.
#: The three registered captures clear it by 3 to 38 orders of magnitude, so
#: nothing here is sitting on the threshold.
_P_THRESHOLD = 1e-4


def _phi_correlation(stream: np.ndarray, period: int) -> float:
    """Polarity-invariant Pearson correlation at a lag, as in
    ``repetition_period._phi_correlation`` / ``milstd188110a.measure_cadence``."""
    if len(stream) <= period:
        return 0.0
    left = stream[:-period].astype(np.float64)
    right = stream[period:].astype(np.float64)
    left = left - left.mean()
    right = right - right.mean()
    scale = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / scale) if scale else 0.0


def _hypergeometric_sf(hits: int, population: int, successes: int, draws: int) -> float:
    """``P(X >= hits)`` for a hypergeometric draw; exact, no SciPy dependency."""
    total = comb(population, draws)
    if not total:
        return 1.0
    tail = sum(comb(successes, i) * comb(population - successes, draws - i)
               for i in range(hits, min(successes, draws) + 1))
    return tail / total


def byte_reverse_bits(bits: np.ndarray) -> np.ndarray:
    """Flip bit order inside each byte, converting MSB-first unpacking to
    LSB-first (the documented capture convention) and back."""
    stream = np.asarray(bits, dtype=np.uint8).reshape(-1) & 1
    usable = (len(stream) // 8) * 8
    return stream[:usable].reshape(-1, 8)[:, ::-1].reshape(-1)


@dataclass(frozen=True)
class PeriodEvidence:
    """How strongly one documented period shows up in one bitstream."""

    period: int
    orientation: str
    correlation: float
    multiples: int
    top_ranked: int
    p_value: float
    frames: int
    phase: int
    bias_spread: float

    @property
    def confirmed(self) -> bool:
        return self.p_value < _P_THRESHOLD


def harmonic_rank_evidence(bits: np.ndarray, period: int,
                           orientation: str = "as-given") -> PeriodEvidence | None:
    """Test whether the integer multiples of ``period`` sit at the top of the
    autocorrelation ranking.

    A single elevated lag is weak (a short capture can produce one by chance),
    and a fixed correlation threshold has to be tuned per capture length. Rank
    concentration needs neither: rank every lag in ``[8, 400]`` by absolute
    phi-correlation, then ask how many of the ``m`` multiples of ``period``
    land in the top ``m`` ranks. Under the null hypothesis of no periodicity
    that count is hypergeometric, which gives a calibrated p-value directly.
    """
    stream = np.asarray(bits, dtype=np.uint8).reshape(-1) & 1
    high = min(_SEARCH_HIGH, len(stream) // 3)
    if high < period or period < _SEARCH_LOW:
        return None
    lags = np.arange(_SEARCH_LOW, high + 1)
    values = np.array([abs(_phi_correlation(stream, int(lag))) for lag in lags])
    ranks = np.empty(len(lags), dtype=int)
    ranks[np.argsort(-values)] = np.arange(len(lags))
    multiples = [int(lag) for lag in lags if lag % period == 0]
    if not multiples:
        return None
    hits = sum(1 for lag in multiples if ranks[lag - _SEARCH_LOW] < len(multiples))
    p_value = _hypergeometric_sf(hits, len(lags), len(multiples), len(multiples))
    phase, spread = _best_phase(stream, period)
    return PeriodEvidence(
        period=period,
        orientation=orientation,
        correlation=float(values[period - _SEARCH_LOW]),
        multiples=len(multiples),
        top_ranked=hits,
        p_value=p_value,
        frames=len(stream) // period,
        phase=phase,
        bias_spread=spread,
    )


def _best_phase(stream: np.ndarray, period: int) -> tuple[int, float]:
    """Frame phase whose per-position mark bias is most uneven.

    A real frame has bit positions that are not equiprobable (sync, parity, and
    alphabet-constrained positions); the phase that maximises that unevenness
    is the frame boundary the capture actually supports. Reported as evidence
    of frame structure, not as a decode.
    """
    best = (0, 0.0)
    for phase in range(period):
        tail = stream[phase:]
        count = len(tail) // period
        if count < 8:
            continue
        bias = tail[: count * period].reshape(count, period).mean(axis=0)
        spread = float(bias.max() - bias.min())
        if spread > best[1]:
            best = (phase, spread)
    return best


def best_orientation_evidence(bits: np.ndarray, period: int) -> PeriodEvidence | None:
    """Evaluate the documented period in both bit-packing orientations.

    Callers may hand this decoder an MSB-first unpacking (the CLI's default),
    while the original help file specifies LSB-first. Rather than silently
    assume one, test both and report which one the capture supports.
    """
    candidates = []
    direct = harmonic_rank_evidence(bits, period, "as-given")
    if direct is not None:
        candidates.append(direct)
    stream = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if len(stream) >= 8:
        flipped = harmonic_rank_evidence(byte_reverse_bits(stream), period,
                                         "byte-reversed (documented LSB-first packing)")
        if flipped is not None:
            candidates.append(flipped)
    if not candidates:
        return None
    return min(candidates, key=lambda evidence: (evidence.p_value, -evidence.top_ranked))


class DocumentedPeriodDecoder(Decoder):
    """Verifies one help-file-documented frame period against the capture."""

    status = "partial"
    confidence = "medium"
    layer = "synchronization"
    period = 0
    citation = ""

    def decode(self, bits: np.ndarray, block=None) -> DecodeResult:
        evidence = best_orientation_evidence(bits, self.period)
        if evidence is None:
            return DecodeResult(
                text="", symbol_count=0, error_count=1,
                notes=(f"Too few bits to test the documented {self.period}-bit "
                       f"period for '{self.name}'. {self.citation}"),
            )
        verdict = "confirmed" if evidence.confirmed else "NOT confirmed"
        text = (f"period={evidence.period}b ({verdict}) correlation="
                f"{evidence.correlation:.3f} multiples_top_ranked="
                f"{evidence.top_ranked}/{evidence.multiples} p={evidence.p_value:.2e} "
                f"frames={evidence.frames} orientation={evidence.orientation} "
                f"frame_phase={evidence.phase} mark_bias_spread={evidence.bias_spread:.2f}")
        return DecodeResult(
            text=text,
            symbol_count=evidence.frames if evidence.confirmed else 0,
            error_count=0 if evidence.confirmed else 1,
            notes=(f"{self.citation} Frame period only -- alphabet, error correction "
                   "and payload structure are not recovered. The p-value is the "
                   "hypergeometric probability that this many integer multiples of "
                   "the documented period would top the autocorrelation ranking by "
                   "chance; see bee/decoders/documented_period.py."),
        )


_REGISTRATIONS = {
    "AUTOSPEC": (
        10,
        "Bee.Demo/BeeR.chm:hlp_vwold_zautospec.htm states FSK2, 850 Hz shift, "
        "67 bit/s, a 10-bit primary transmission cycle subjected to interleaving, "
        "and a 40-bit displayed period; the DLL codename is Sig10c10.dll (000.dll).",
    ),
    "HNG-FEC": (
        15,
        "Bee.Demo/BeeR.chm:hlp_vwold_hng_fec.htm states FSK2, 100 Hz shift, "
        "200 bit/s, a 15-bit transmission cycle taking one of two forms alternating "
        "every 55 bits, and a 75-bit displayed period; the DLL codename is "
        "Sig15c15p.dll (007.dll).",
    ),
}

for _name, (_period, _citation) in _REGISTRATIONS.items():
    _decoder = DocumentedPeriodDecoder()
    _decoder.name = _name
    _decoder.period = _period
    _decoder.citation = _citation
    _decoder.description = (f"{_period}-bit frame period documented by the original "
                            "help file and confirmed against the bundled capture")
    _decoder.provenance = ("original Bee help file (Bee.Demo/BeeR.chm, hlp_vwold_*.htm) "
                           "cross-checked against the bundled capture by an "
                           "autocorrelation rank test")
    register(_decoder)
