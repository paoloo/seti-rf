"""Bounded voltage cutouts and candidate-linked evidence for recorded replay."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from operator import index

import numpy as np


@dataclass(frozen=True)
class VoltageStream:
    observation: str
    beam: str
    tuning: str
    polarization: str
    sample_rate_hz: float
    center_hz: float
    sideband: int = 1

    def __post_init__(self):
        if not isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0:
            raise ValueError("sample rate must be positive and finite")
        if not isfinite(self.center_hz) or self.sideband not in (-1, 1):
            raise ValueError("invalid RF center or sideband")

    def rf_hz(self, baseband_hz: float) -> float:
        return self.center_hz + self.sideband * baseband_hz

    def baseband_hz(self, rf_hz: float) -> float:
        return self.sideband * (rf_hz - self.center_hz)


@dataclass(frozen=True)
class VoltageRequest:
    candidate_id: str
    stream: VoltageStream
    first_sample: int
    stop_sample: int
    low_hz: float
    high_hz: float

    def __post_init__(self):
        index(self.first_sample)
        index(self.stop_sample)
        if self.first_sample < 0 or self.stop_sample <= self.first_sample:
            raise ValueError("require a nonempty half-open sample interval")
        if not all(isfinite(hz) for hz in (self.low_hz, self.high_hz)) or self.high_hz < self.low_hz:
            raise ValueError("require an ordered finite RF interval")


class LookupStatus(str, Enum):
    COMPLETE = "complete"
    GAP = "gap"
    EXPIRED = "expired"
    STREAM_MISMATCH = "stream_mismatch"
    OUTSIDE_BAND = "outside_band"
    NOT_YET_AVAILABLE = "not_yet_available"


@dataclass(frozen=True)
class VoltageCutout:
    request: VoltageRequest
    samples: np.ndarray
    baseband_at_low_rf_hz: float
    baseband_at_high_rf_hz: float


@dataclass(frozen=True)
class VoltageEvidence:
    request: VoltageRequest
    status: LookupStatus
    probe_result: object = None
    probe_error: str | None = None


class VoltageHistory:
    """Index monotonically appended, complex chunks for one stream.

    Sample indices use the observation clock and intervals are half-open.
    Capacity in samples describes a time span, including missing samples.
    Use one history per beam, tuning and polarization. Append and lookup are
    serialized by the caller.
    """

    def __init__(self, stream: VoltageStream, capacity_samples: int):
        capacity_samples = index(capacity_samples)
        if capacity_samples <= 0:
            raise ValueError("capacity must be positive")
        self.stream = stream
        self.capacity_samples = capacity_samples
        self._chunks: deque[tuple[int, np.ndarray]] = deque()
        self._first: int | None = None
        self._stop: int | None = None

    def append(self, first_sample: int, samples: np.ndarray) -> None:
        first_sample = index(first_sample)
        values = np.asarray(samples)
        if values.ndim != 1 or not values.size or not np.iscomplexobj(values):
            raise ValueError("expected a nonempty one-dimensional complex voltage chunk")
        if first_sample < 0 or (self._stop is not None and first_sample < self._stop):
            raise ValueError("chunks must have increasing, nonoverlapping sample intervals")
        if self._first is None:
            self._first = first_sample
        self._stop = first_sample + len(values)
        floor = self._stop - self.capacity_samples
        while self._chunks and self._chunks[0][0] + len(self._chunks[0][1]) <= floor:
            self._chunks.popleft()
        if self._chunks and self._chunks[0][0] < floor:
            start, partial = self._chunks.popleft()
            self._chunks.appendleft((floor, partial[floor - start:].copy()))
        start = max(floor, first_sample)
        self._chunks.append((start, values[start - first_sample:].copy()))

    def lookup(self, request: VoltageRequest) -> tuple[LookupStatus, VoltageCutout | None]:
        if request.stream != self.stream:
            return LookupStatus.STREAM_MISMATCH, None
        low, high = (self.stream.baseband_hz(rf) for rf in (request.low_hz, request.high_hz))
        nyquist = self.stream.sample_rate_hz / 2
        if any(not -nyquist <= bound < nyquist for bound in (low, high)):
            return LookupStatus.OUTSIDE_BAND, None
        if self._stop is None:
            return LookupStatus.NOT_YET_AVAILABLE, None
        floor = self._stop - self.capacity_samples
        if self._first is not None and floor > self._first and request.first_sample < floor:
            return LookupStatus.EXPIRED, None
        if request.stop_sample > self._stop:
            return LookupStatus.NOT_YET_AVAILABLE, None

        cursor = request.first_sample
        parts = []
        for start, values in self._chunks:
            if start + len(values) <= cursor:
                continue
            if start > cursor:
                break
            end = min(start + len(values), request.stop_sample)
            parts.append(values[cursor - start:end - start])
            cursor = end
            if cursor == request.stop_sample:
                samples = np.concatenate(parts)
                return LookupStatus.COMPLETE, VoltageCutout(request, samples, low, high)
        return LookupStatus.GAP, None

    def annotate(
        self,
        request: VoltageRequest,
        probe: Callable[[VoltageCutout], object] | None = None,
    ) -> VoltageEvidence:
        """Return the original candidate request even when its lookup or probe fails."""
        status, cutout = self.lookup(request)
        if cutout is None or probe is None:
            return VoltageEvidence(request, status)
        try:
            return VoltageEvidence(request, status, probe(cutout))
        except Exception as exc:
            return VoltageEvidence(request, status, probe_error=f"{type(exc).__name__}: {exc}")
