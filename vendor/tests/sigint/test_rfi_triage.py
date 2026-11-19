from sigint import detect as det
from sigint.dataset import SignalSpec, generate_signal
from sigint.rfi import triage

SR = 16000.0


def _tone(freq_hz=1500.0, snr=30, seed=1):
    return generate_signal(SignalSpec("tone", SR, 1.0, freq_hz=freq_hz,
                                      snr_db=snr, seed=seed)).samples


def test_verdict_for_gps_l1_is_known_and_space_origin():
    v = triage.verdict_for(1575.42e6)
    assert v.is_known
    assert v.origins == ("space",)


def test_verdict_for_an_unallocated_frequency_is_unknown():
    v = triage.verdict_for(1_000.0)
    assert not v.is_known
    assert v.origins == ()
    assert v.summary() == "unmatched"


def test_triage_carrier_shifts_by_center_hz():
    x = _tone(freq_hz=1500.0)
    center_hz = 1090e6 - 1500.0
    scan = det.find_carriers(x, SR, center_hz=center_hz)
    carrier = scan.carriers[0]
    v = triage.triage_carrier(carrier)
    assert v.is_known
    assert any(b.label == "ADS-B / Mode S 1090ES" for b in v.bands)


def test_triage_carrier_defaults_to_no_shift():
    x = _tone(freq_hz=1500.0)
    scan = det.find_carriers(x, SR)   # center_hz defaults to 0.0
    carrier = scan.carriers[0]
    v = triage.triage_carrier(carrier)   # abs_freq_hz == 1500 Hz, unmatched
    assert not v.is_known


def test_triage_carriers_sorts_unmatched_first():
    x = _tone(freq_hz=1500.0)
    scan = det.find_carriers(x, SR)
    pairs = triage.triage_carriers(scan.carriers)
    assert pairs
    assert pairs[0][1].is_known is False
