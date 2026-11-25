from dataclasses import replace

import numpy as np
import pytest

from seti_rf.triage import LookupStatus, VoltageHistory, VoltageRequest, VoltageStream


def test_cutout_crosses_chunks_and_preserves_signed_rf():
    stream = VoltageStream("scan", "0", "LoB", "X", 16_000, 1.42e9, sideband=-1)
    rng = np.random.default_rng(9)
    x = (rng.normal(size=32) + 1j * rng.normal(size=32)).astype("complex64")
    history = VoltageHistory(stream, 24)
    for start in (0, 16):
        history.append(100 + start, x[start:start + 16])
    request = VoltageRequest("crossing", stream, 110, 121,
                             stream.center_hz - 1500, stream.center_hz + 500)
    status, cutout = history.lookup(request)
    assert status == LookupStatus.COMPLETE
    np.testing.assert_array_equal(cutout.samples, x[10:21])
    assert cutout.samples.dtype == x.dtype
    assert cutout.baseband_at_low_rf_hz == 1500
    assert cutout.baseband_at_high_rf_hz == -500
    assert stream.rf_hz(1500) == request.low_hz
    assert stream.rf_hz(-500) == request.high_hz


def test_failures_and_probe_evidence_preserve_candidate():
    stream = VoltageStream("scan", "0", "LoB", "X", 16_000, 1e9)
    history = VoltageHistory(stream, 16)
    samples = np.ones(8, dtype="complex64")
    history.append(0, samples)
    history.append(10, samples)
    request = VoltageRequest("found", stream, 10, 11, 1e9, 1e9)
    for candidate, expected in [
        (replace(request, first_sample=0), LookupStatus.EXPIRED),
        (replace(request, first_sample=7), LookupStatus.GAP),
        (replace(request, stream=replace(stream, tuning="LoA")), LookupStatus.STREAM_MISMATCH),
        (replace(request, high_hz=1e9 + 8000), LookupStatus.OUTSIDE_BAND),
        (replace(request, stop_sample=19), LookupStatus.NOT_YET_AVAILABLE),
    ]:
        evidence = history.annotate(candidate, lambda _: pytest.fail("probe ran after failed lookup"))
        assert evidence.request is candidate
        assert evidence.status == expected
    assert history.annotate(request, lambda cutout: cutout.samples.size).probe_result == 1

    def broken_probe(_):
        raise RuntimeError("worker error")

    evidence = history.annotate(request, broken_probe)
    assert evidence.request is request
    assert evidence.status == LookupStatus.COMPLETE
    assert evidence.probe_error == "RuntimeError: worker error"
    with pytest.raises(ValueError, match="nonoverlapping"):
        history.append(17, samples)
