"""Tests for sigint.rfi.gps -- validated against a synthetic signal built
from the same code generator, ground-truth style (see sigint.dataset)."""
import numpy as np
import pytest

from sigint.rfi import gps


def test_ca_codes_are_1023_chips_and_distinct_per_prn():
    codes = [gps.generate_ca_code(prn) for prn in (1, 2, 3, 32)]
    for c in codes:
        assert len(c) == gps.CODE_LENGTH
        assert set(np.unique(c)) <= {-1.0, 1.0}
    # Gold codes have low cross-correlation: distinct PRNs must not be
    # near-identical sequences
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            corr = float(np.dot(codes[i], codes[j])) / gps.CODE_LENGTH
            assert abs(corr) < 0.3, f"PRNs too correlated: {corr}"


def test_ca_code_rejects_out_of_range_prn():
    with pytest.raises(ValueError):
        gps.generate_ca_code(33)
    with pytest.raises(ValueError):
        gps.generate_ca_code(0)


def _synthesize(prn: int, sample_rate: float, doppler_hz: float,
                code_phase_chips: float, snr_db: float,
                seed: int = 0) -> np.ndarray:
    """One code period of a synthetic C/A signal at complex baseband,
    already wiped of the L1 carrier except for the given Doppler residual."""
    rng = np.random.default_rng(seed)
    n = int(round(sample_rate / gps.CHIP_RATE_HZ * gps.CODE_LENGTH))
    code = gps.generate_ca_code(prn)
    chip_idx = (np.floor(np.arange(n) * gps.CHIP_RATE_HZ / sample_rate
                         + code_phase_chips).astype(int) % gps.CODE_LENGTH)
    baseband = code[chip_idx].astype(np.complex128)
    t = np.arange(n) / sample_rate
    x = baseband * np.exp(2j * np.pi * doppler_hz * t)
    sig_power = float(np.mean(np.abs(x) ** 2))
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (rng.standard_normal(n)
                                        + 1j * rng.standard_normal(n))
    return x + noise


def test_acquire_finds_the_right_prn_doppler_and_code_phase():
    sr = 4 * gps.CHIP_RATE_HZ
    x = _synthesize(prn=7, sample_rate=sr, doppler_hz=-1350.0,
                    code_phase_chips=311.0, snr_db=-15.0)
    hits = gps.acquire(x, sr, prns=(3, 7, 12), doppler_span_hz=2000.0,
                       doppler_step_hz=250.0)
    assert hits, "expected at least one acquisition"
    best = hits[0]
    assert best.prn == 7
    assert abs(best.doppler_hz - (-1350.0)) <= 250.0
    assert abs(best.code_phase_chips - 311.0) % gps.CODE_LENGTH <= 2.0
    assert best.peak_to_secondary >= 8.0


def test_acquire_self_rejects_on_noise():
    sr = 4 * gps.CHIP_RATE_HZ
    rng = np.random.default_rng(1)
    n = int(round(sr / gps.CHIP_RATE_HZ * gps.CODE_LENGTH))
    noise = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    hits = gps.acquire(noise, sr, prns=(1, 2, 3), doppler_span_hz=1000.0,
                       doppler_step_hz=500.0)
    assert hits == ()


def test_acquire_rejects_wrong_prn_at_high_confidence():
    sr = 4 * gps.CHIP_RATE_HZ
    x = _synthesize(prn=15, sample_rate=sr, doppler_hz=500.0,
                    code_phase_chips=50.0, snr_db=-15.0)
    hits = gps.acquire(x, sr, prns=(1, 2, 3), doppler_span_hz=1000.0,
                       doppler_step_hz=500.0)
    assert hits == (), "must not fire on a PRN that isn't actually present"


def test_zero_signal_has_no_acquisition():
    assert gps.acquire(np.zeros(4096), 4_096_000, prns=(1,)) == ()
