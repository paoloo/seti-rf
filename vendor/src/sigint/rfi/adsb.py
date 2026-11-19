"""ADS-B / Mode S 1090ES: preamble detection, PPM decode, CRC-24 check.

Another real, self-rejecting waveform-level probe (like sigint.hfdl and
sigint.rfi.gps): the 1090ES preamble is a fixed 8 microsecond pulse
pattern (ICAO Annex 10 Vol IV), each data bit is pulse-position-modulated
within its own microsecond, and DF17 ("extended squitter") messages carry
a 24-bit CRC that is a *pure* polynomial check of the other 88 bits (no
XOR with the aircraft address, unlike the other Mode S downlink formats)
-- so a decode either produces a message whose CRC checks out, or it
doesn't, with no ambiguity.

Only DF17/DF18 (extended squitter, ADS-B) are decoded; the other downlink
formats (DF4/5/11/20/21, whose parity XORs with the interrogated aircraft
address) are out of scope here.

Input: real-valued or complex envelope samples at >= 2 Msps (2 samples per
microsecond is the minimum to resolve the half-microsecond PPM chip; more
gives cleaner detection). Feed magnitude-detected AM samples, not raw
carrier IQ.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BIT_RATE_HZ = 1e6                 # 1 Mbit/s, 1 us per bit
PREAMBLE_US = 8.0
SHORT_BITS = 56
LONG_BITS = 112

# Mode S CRC-24 generator polynomial (ICAO Annex 10 Vol IV / DO-260,
# widely published e.g. in dump1090 and pyModeS): degree-24, feedback
# constant 0xFFF409. Verified below against a canonical DF17 message
# (tests/test_rfi_adsb.py) rather than trusted from memory alone.
_CRC_POLY = 0xFFF409


def crc24(bits: np.ndarray) -> int:
    """Mode S CRC-24 remainder of a 0/1 bit array (bit-by-bit polynomial
    division, MSB first) -- equivalent to the table-driven form, just
    without needing to carry a 112-entry lookup table."""
    reg = 0
    for b in bits:
        top = (reg >> 23) & 1
        reg = ((reg << 1) | int(b)) & 0xFFFFFF
        if top:
            reg ^= _CRC_POLY
    return reg


@dataclass(frozen=True)
class ADSBMessage:
    df: int                 # downlink format (17 or 18 for ADS-B)
    bytes_out: bytes
    crc_ok: bool
    start_sample: int


def _preamble_template(sps: int) -> np.ndarray:
    """Ideal 8 us preamble envelope at `sps` samples per microsecond:
    pulses (0.5 us wide) at 0, 1.0, 3.5 and 4.5 us."""
    n = int(round(PREAMBLE_US * sps))
    tmpl = np.zeros(n)
    half = max(1, sps // 2)
    for start_us in (0.0, 1.0, 3.5, 4.5):
        i = int(round(start_us * sps))
        tmpl[i:i + half] = 1.0
    return tmpl


def find_preambles(envelope: np.ndarray, sample_rate: float,
                   min_snr: float = 3.0) -> list[int]:
    """Sample indices where the 8 us preamble pattern correlates strongly
    against the local noise floor."""
    x = np.asarray(envelope, dtype=np.float64).ravel()
    sps = max(1, int(round(sample_rate / BIT_RATE_HZ)))
    tmpl = _preamble_template(sps)
    n = len(tmpl)
    if len(x) < n:
        return []
    # normalized sliding correlation via cumulative sums (matched filter
    # against a 0/1 template == a sum over the "on" samples)
    on = np.where(tmpl > 0)[0]
    csum = np.concatenate([[0.0], np.cumsum(x)])
    hits: list[tuple[int, float]] = []
    floor = float(np.median(x)) + 1e-9
    for start in range(len(x) - n + 1):
        on_energy = x[start + on].sum()
        window_energy = csum[start + n] - csum[start]
        off_energy = window_energy - on_energy
        off_n = n - len(on)
        mean_on = on_energy / len(on)
        mean_off = off_energy / off_n if off_n else floor
        score = mean_on / max(mean_off, floor, 1e-9)
        if score >= min_snr:
            if not hits or start - hits[-1][0] > sps:
                hits.append((start, score))
            elif score > hits[-1][1]:
                hits[-1] = (start, score)
    return [start for start, _ in hits]


def _ppm_decode(envelope: np.ndarray, start: int, sps: int,
                n_bits: int) -> np.ndarray | None:
    """PPM-decode `n_bits` starting right after the preamble: bit=1 if the
    first half of its microsecond has more energy than the second half."""
    half = max(1, sps // 2)
    bits = np.empty(n_bits, dtype=np.uint8)
    for i in range(n_bits):
        base = start + i * sps
        if base + sps > len(envelope):
            return None
        first = envelope[base:base + half].sum()
        second = envelope[base + half:base + sps].sum()
        bits[i] = 1 if first > second else 0
    return bits


def decode(envelope: np.ndarray, sample_rate: float,
          min_snr: float = 3.0) -> tuple[ADSBMessage, ...]:
    """Find and decode every DF17/DF18 message in `envelope`.

    Tries a long (112-bit) decode at every preamble hit; a message is
    reported only if its CRC-24 checks out -- garbage preambles (or a
    short 56-bit reply misread as long) simply produce no output.
    """
    x = np.asarray(envelope, dtype=np.float64).ravel()
    sps = max(1, int(round(sample_rate / BIT_RATE_HZ)))
    preamble_len = int(round(PREAMBLE_US * sps))
    out = []
    for start in find_preambles(x, sample_rate, min_snr=min_snr):
        data_start = start + preamble_len
        bits = _ppm_decode(x, data_start, sps, LONG_BITS)
        if bits is None:
            continue
        df = int(bits[0] * 16 + bits[1] * 8 + bits[2] * 4 + bits[3] * 2 + bits[4])
        if df not in (17, 18):
            continue
        remainder = crc24(bits)
        if remainder != 0:
            continue
        byte_bits = bits[:LONG_BITS - 24]   # data bits, excluding the 24-bit parity
        n_bytes = len(byte_bits) // 8
        raw = np.packbits(byte_bits[:n_bytes * 8])
        out.append(ADSBMessage(df=df, bytes_out=raw.tobytes(),
                               crc_ok=True, start_sample=start))
    return tuple(out)
