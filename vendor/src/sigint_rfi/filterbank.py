"""Bounded-memory power triage of single-plane radio telescope filterbanks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from sigint.rfi import bands

from .sigproc import SigprocFilterbank

PowerData = h5py.Dataset | SigprocFilterbank
FilterbankFile = h5py.File | SigprocFilterbank


@dataclass(frozen=True)
class Grid:
    """SIGPROC frequency and integration metadata."""

    start_mjd: float
    integration_seconds: float
    first_channel_mhz: float
    channel_step_mhz: float
    rows: int
    channels: int

    def frequency_hz(self, channel: int) -> float:
        return (self.first_channel_mhz + channel * self.channel_step_mhz) * 1e6


@dataclass(frozen=True)
class Candidate:
    """One on-beam power excess with optional simultaneous off-beam evidence."""

    row: int
    channel: int
    frequency_hz: float
    on_excess_db: float
    off_channel: int | None
    off_excess_db: float | None
    catalog_matches: tuple[str, ...]

    @property
    def coincident(self) -> bool:
        return self.off_channel is not None


def _read_grid(data: PowerData) -> Grid:
    if len(data.shape) != 3 or data.shape[1] != 1 or min(data.shape) == 0:
        raise ValueError("expected data shaped (time, 1, frequency)")
    if data.dtype.kind not in "uif":
        raise ValueError("expected real linear-power filterbank data")
    try:
        tstart, tsamp, fch1, foff = (
            float(data.attrs[key]) for key in ("tstart", "tsamp", "fch1", "foff")
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("data must have numeric tstart, tsamp, fch1 and foff attributes") from exc
    if not np.all(np.isfinite((tstart, tsamp, fch1, foff))) or tsamp <= 0 or foff == 0:
        raise ValueError("invalid filterbank timing or frequency metadata")
    for key, size in (("nifs", 1), ("nchans", data.shape[2])):
        if key in data.attrs and data.attrs[key] != size:
            raise ValueError(f"{key} does not match the dataset shape")
    return Grid(tstart, tsamp, fch1, foff, data.shape[0], data.shape[2])


def _dataset(file: FilterbankFile) -> PowerData:
    if isinstance(file, SigprocFilterbank):
        return file
    data = file.get("data")
    if not isinstance(data, h5py.Dataset):
        raise ValueError("filterbank requires an HDF5 dataset named data")
    return data


def _mask_for(data: PowerData) -> h5py.Dataset | None:
    if isinstance(data, SigprocFilterbank):
        return None
    mask = data.file.get("mask")
    if mask is not None and (
        not isinstance(mask, h5py.Dataset)
        or mask.shape != data.shape or mask.dtype.kind not in "biu"
    ):
        raise ValueError("filterbank mask must be a boolean or integer dataset matching data")
    return mask


def _excess(
    data: PowerData, mask: h5py.Dataset | None, row: int, start: int, stop: int,
) -> np.ndarray:
    values = (data.read(row, start, stop) if isinstance(data, SigprocFilterbank)
              else data[row, 0, start:stop])
    power = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(power) & (power > 0)
    if mask is not None:
        valid &= mask[row, 0, start:stop] == 0
    excess = np.full(len(power), -np.inf)
    if valid.any():
        log_power = 10 * np.log10(power[valid])
        excess[valid] = log_power - 10 * np.log10(np.median(power[valid]))
    return excess


class FilterbankScan:
    """Scan one filterbank, optionally comparing it to a simultaneous beam."""

    def __init__(self, on: FilterbankFile, off: FilterbankFile | None = None):
        self._on = _dataset(on)
        self._on_mask = _mask_for(self._on)
        self.grid = _read_grid(self._on)
        self._off = _dataset(off) if off is not None else None
        self._off_mask = None
        if off is not None:
            self._off_mask = _mask_for(self._off)
            if self.grid != _read_grid(self._off):
                raise ValueError("on/off filterbanks must have identical time and frequency grids")
            if on == off or (Path(on.filename).is_file() and Path(off.filename).is_file()
                             and Path(on.filename).samefile(off.filename)):
                raise ValueError("on and off must be separate filterbank files")

    def candidates(
        self, *, on_db: float = 12.0, off_db: float = 12.0,
        tolerance_channels: int = 0, block_channels: int = 8192,
        rows: tuple[int, int] | None = None,
        channels: tuple[int, int] | None = None,
    ) -> Iterator[Candidate]:
        """Yield on-beam excesses and the strongest off-beam match per pixel.

        A coincidence is a review flag. The thresholds are power ratios to
        each beam's block median, with no calibrated false-alarm probability.
        """
        if not np.all(np.isfinite((on_db, off_db))) or min(on_db, off_db) <= 0:
            raise ValueError("power excess thresholds must be positive and finite")
        if self._off is None and tolerance_channels:
            raise ValueError("channel tolerance requires an off-beam")
        if not 0 <= tolerance_channels < block_channels or block_channels < 2:
            raise ValueError("require block_channels >= 2 and 0 <= tolerance < block_channels")

        first_row, last_row = rows if rows is not None else (0, self.grid.rows)
        first_chan, last_chan = channels if channels is not None else (0, self.grid.channels)
        if not 0 <= first_row < last_row <= self.grid.rows:
            raise ValueError("row range must be nonempty and within the grid")
        if not 0 <= first_chan < last_chan <= self.grid.channels:
            raise ValueError("channel range must be nonempty and within the grid")

        first_block = first_chan // block_channels * block_channels
        for row in range(first_row, last_row):
            for start in range(first_block, last_chan, block_channels):
                stop = min(start + block_channels, self.grid.channels)
                # Anchor a short final block against preceding channels so
                # a lone channel at the file edge has a usable baseline.
                lo = max(0, min(start, stop - block_channels) - tolerance_channels)
                hi = min(self.grid.channels, stop + tolerance_channels)
                on_excess = _excess(self._on, self._on_mask, row, lo, hi)
                off_excess = (_excess(self._off, self._off_mask, row, lo, hi)
                              if self._off is not None else None)
                core = slice(max(start, first_chan) - lo, min(stop, last_chan) - lo)
                for index in np.flatnonzero(on_excess[core] >= on_db) + core.start:
                    channel = lo + int(index)
                    off_channel = off_db_value = None
                    if off_excess is not None:
                        left = max(0, int(index) - tolerance_channels)
                        right = min(len(off_excess), int(index) + tolerance_channels + 1)
                        match = left + int(np.argmax(off_excess[left:right]))
                        if off_excess[match] >= off_db:
                            off_channel = lo + match
                            off_db_value = float(off_excess[match])
                    frequency = self.grid.frequency_hz(channel)
                    yield Candidate(
                        row, channel, frequency, float(on_excess[index]),
                        off_channel, off_db_value,
                        tuple(band.label for band in bands.lookup(frequency)),
                    )


class PairedFilterbanks(FilterbankScan):
    """Compare two simultaneous beams with identical time/frequency grids."""

    def __init__(self, on: FilterbankFile, off: FilterbankFile):
        super().__init__(on, off)
