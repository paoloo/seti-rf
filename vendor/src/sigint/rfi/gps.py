"""GPS L1 C/A acquisition: a real waveform-level probe, not a band guess.

Generates the IS-GPS-200 Gold codes and runs standard parallel-code-phase
search (FFT circular correlation over a code period, at each point in a
Doppler search grid) against complex baseband IQ samples. A strong,
narrow correlation peak for a given PRN is direct evidence of that
satellite's signal -- self-rejecting, like :mod:`sigint.hfdl`: noise or
an unrelated signal simply produces no peak above the acquisition
threshold.

Requires the input to already be downconverted near 1575.42 MHz (baseband
0 Hz = the GPS L1 carrier, give or take the Doppler search range) and
sampled fast enough to resolve the 1.023 Mchip/s C/A code -- at least
~2.046 Msps (Nyquist for the main lobe), several times that for good code-
phase resolution. This is a different sample-rate regime from the rest of
`sigint`, which mostly analyzes audio-rate signals; GPS acquisition is
run separately, deliberately, not as part of the audio-rate Gate pipeline.

Validated in tests/test_rfi_gps.py against a synthetic C/A signal built
from this same code generator (encode -> BPSK -> add carrier/Doppler/noise
-> acquire), the same ground-truth discipline as sigint.dataset.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CHIP_RATE_HZ = 1.023e6
CODE_LENGTH = 1023
L1_HZ = 1575.42e6

# IS-GPS-200 Table 3-Ia: G2 code phase delay (chips) for PRN 1-32. Index 0
# is unused so PRN indexes this table directly (PRN 1 -> _G2_DELAY[1]).
_G2_DELAY = (
    None,
    5, 6, 7, 8, 17, 18, 139, 140, 141, 251,
    252, 254, 255, 256, 257, 258, 469, 470, 471, 472,
    473, 474, 509, 512, 513, 514, 515, 516, 859, 860,
    861, 862,
)


def _lfsr_sequence(taps: tuple[int, ...], length: int = CODE_LENGTH) -> np.ndarray:
    """Maximal-length sequence from a 10-bit LFSR, all-ones initial state.
    `taps` are 1-indexed feedback tap positions (MSB=1) per IS-GPS-200's
    convention; output is the value shifted out each clock, before shift."""
    reg = np.ones(10, dtype=np.int8)
    out = np.empty(length, dtype=np.int8)
    for i in range(length):
        out[i] = reg[9]
        fb = 0
        for t in taps:
            fb ^= int(reg[t - 1])
        reg[1:] = reg[:-1]
        reg[0] = fb
    return out


def generate_ca_code(prn: int) -> np.ndarray:
    """The 1023-chip C/A Gold code for `prn` (1..32), as +/-1 (BPSK)."""
    if not 1 <= prn <= 32:
        raise ValueError("prn must be 1..32 (this table only covers GPS PRNs)")
    g1 = _lfsr_sequence((3, 10))
    g2 = _lfsr_sequence((2, 3, 6, 8, 9, 10))
    delay = _G2_DELAY[prn]
    g2_shifted = np.roll(g2, delay)
    chips = np.bitwise_xor(g1, g2_shifted)
    return 1 - 2 * chips.astype(np.float64)   # 0/1 -> +1/-1


@dataclass(frozen=True)
class GPSDetection:
    prn: int
    doppler_hz: float
    code_phase_chips: float
    peak_to_secondary: float     # primary peak power / best code-phase
                                 # correlation >= 2 chips away, same Doppler


def acquire(
    samples: np.ndarray,
    sample_rate: float,
    prns: tuple[int, ...] = tuple(range(1, 33)),
    doppler_span_hz: float = 5000.0,
    doppler_step_hz: float = 500.0,
    threshold: float = 4.0,
) -> tuple[GPSDetection, ...]:
    """Parallel code-phase search acquisition over one code period (1 ms).

    For each PRN and each Doppler bin in the search grid, correlate one
    code period of `samples` (complex baseband, already wiped of the
    nominal L1 carrier) against the upsampled C/A code via FFT circular
    correlation -- this tests every code phase at once. A PRN is reported
    if its primary peak stands `threshold`x above the *secondary* peak (the
    best code-phase correlation at least a few chips away from the primary,
    in the same best-Doppler bin) -- the standard GPS acquisition ratio
    test (Kaplan & Hegarty). Comparing against a secondary peak rather than
    the mean of the whole search grid keeps the test's false-alarm rate
    independent of how many Doppler bins were searched (comparing to the
    grid mean does not: more bins searched pushes the noise-only maximum
    up via order statistics, and a fixed threshold against that mean would
    silently drift with search width).
    """
    x = np.asarray(samples, dtype=np.complex128).ravel()
    n_samples_per_code = int(round(sample_rate / CHIP_RATE_HZ * CODE_LENGTH))
    if len(x) < n_samples_per_code:
        raise ValueError(
            f"need >= {n_samples_per_code} samples (one 1 ms code period "
            f"at {sample_rate:.0f} Hz), got {len(x)}")
    x = x[:n_samples_per_code]
    t = np.arange(n_samples_per_code) / sample_rate
    dopplers = np.arange(-doppler_span_hz, doppler_span_hz + 1e-9, doppler_step_hz)
    guard_samples = max(1, int(round(2 * sample_rate / CHIP_RATE_HZ)))  # +/-2 chips

    detections: list[GPSDetection] = []
    for prn in prns:
        code = generate_ca_code(prn)
        # upsample the code to the sample rate (nearest-chip replication)
        chip_idx = np.floor(np.arange(n_samples_per_code)
                            * CHIP_RATE_HZ / sample_rate).astype(int) % CODE_LENGTH
        code_up = code[chip_idx]
        code_fft_conj = np.conj(np.fft.fft(code_up))

        best_power, best_doppler, best_phase, best_power_row = 0.0, 0.0, 0, None
        for fd in dopplers:
            wiped = x * np.exp(-2j * np.pi * fd * t)
            corr = np.fft.ifft(np.fft.fft(wiped) * code_fft_conj)
            power = np.abs(corr) ** 2
            idx = int(np.argmax(power))
            if power[idx] > best_power:
                best_power, best_doppler, best_phase = power[idx], fd, idx
                best_power_row = power
        if best_power_row is None:
            continue
        n = len(best_power_row)
        excluded = (np.arange(n) - best_phase) % n
        excluded = np.minimum(excluded, n - excluded)
        rest = best_power_row[excluded > guard_samples]
        secondary = float(rest.max()) if len(rest) else 0.0
        ratio = best_power / secondary if secondary > 0 else float("inf")
        if ratio >= threshold:
            # ifft(fft(received) * conj(fft(reference))) peaks at +shift
            # samples when received = roll(reference, shift); the received
            # signal here is the reference advanced BY the code phase, i.e.
            # roll(reference, -phase_samples) -- so the peak index is
            # -phase_samples mod N, and must be negated back.
            phase_chips = -best_phase * CHIP_RATE_HZ / sample_rate
            detections.append(GPSDetection(
                prn=prn, doppler_hz=float(best_doppler),
                code_phase_chips=float(phase_chips % CODE_LENGTH),
                peak_to_secondary=float(ratio)))
    return tuple(sorted(detections, key=lambda d: -d.peak_to_secondary))
