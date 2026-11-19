import numpy as np

from seti_rf import signal, triage
from sigint import detect
from sigint_rfi.filterbank import FilterbankScan


def test_bundle_exposes_existing_packages_through_public_api():
    assert signal.detect is detect
    assert triage.FilterbankScan is FilterbankScan


def test_signal_path_preserves_a_negative_baseband_offset():
    sample_rate = 8192
    iq = np.exp(-2j * np.pi * 1000 * np.arange(8192) / sample_rate)
    result = signal.detect.find_carriers(iq, sample_rate, center_hz=1_420_405_000)
    strongest = result.carriers[0]
    assert strongest.freq_hz == -1000
    assert strongest.abs_freq_hz == 1_420_404_000
