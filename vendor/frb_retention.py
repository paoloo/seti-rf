"""Measure how proposed single-beam flags overlap an observed dispersed pulse."""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from sigint_rfi.filterbank import FilterbankScan
from sigint_rfi.sigproc import SigprocFilterbank


def evaluate(path: str, dm: float, width_ms: float, thresholds: list[float]) -> dict:
    if not math.isfinite(dm) or dm < 0 or not math.isfinite(width_ms) or width_ms <= 0:
        raise ValueError("require finite nonnegative DM and positive pulse width")
    with SigprocFilterbank(path) as file:
        scan = FilterbankScan(file)
        grid = scan.grid
        if grid.rows * grid.channels > 32_000_000:
            raise ValueError("supply a cropped filterbank of at most 32 million pixels")
        values = np.stack([file.read(row, 0, grid.channels) for row in range(grid.rows)])
        frequency = grid.first_channel_mhz + np.arange(grid.channels) * grid.channel_step_mhz
        if np.any(frequency <= 0):
            raise ValueError("dispersion calculation requires positive RF frequencies")
        delay = 4148.808 * dm * (frequency**-2 - frequency.max()**-2)
        delay_rows = np.rint(delay / grid.integration_seconds)
        width_rows = width_ms / 1000 / grid.integration_seconds
        if not np.all(np.isfinite(delay_rows)) or delay_rows.max() >= grid.rows or width_rows > grid.rows:
            raise ValueError("cutout is too short for the dispersion delay and pulse width")
        shifts = delay_rows.astype(int)
        width = max(1, round(width_rows))
        common_rows = grid.rows - int(shifts.max())
        if width > common_rows:
            raise ValueError("cutout is too short for the dispersion delay and pulse width")

        valid = np.isfinite(values) & (values > 0)
        good = valid.any(axis=0)
        if not good.any():
            raise ValueError("cutout has no valid power samples")
        channel = np.flatnonzero(good)
        clean = np.where(valid[:, good], values[:, good], np.nan)
        baseline = np.nanmedian(clean, axis=0)
        excess = np.nan_to_num(clean / baseline - 1, nan=0.0)
        profile = np.zeros(common_rows)
        for column, shift in enumerate(shifts[good]):
            profile += excess[shift:shift + common_rows, column]
        cumulative = np.concatenate(([0.0], np.cumsum(profile)))
        sums = cumulative[width:] - cumulative[:-width]
        start = int(np.argmax(sums))
        track_rows = start + shifts[channel] + np.arange(width)[:, None]
        track = excess[track_rows, np.arange(len(channel))]
        signal = float(track.sum())
        reports = []
        for threshold in thresholds:
            flags = 0
            overlap = 0
            removed = 0.0
            for hit in scan.candidates(on_db=threshold):
                flags += 1
                relative = hit.row - shifts[hit.channel] - start
                if good[hit.channel] and 0 <= relative < width:
                    column = int(np.searchsorted(channel, hit.channel))
                    overlap += 1
                    removed += float(track[relative, column])
            reports.append({
                "threshold_db": threshold, "flags": flags, "burst_pixels_flagged": overlap,
                "excess_retained_fraction": (signal - removed) / signal if signal > 0 else None,
            })
        return {
            "file": path, "dm_pc_cm3": dm, "peak_selected_from_cutout": True,
            "pulse_first_row_at_highest_frequency": start, "pulse_width_rows": width,
            "pulse_window_pixels": int(track.size), "total_pixels": int(values.size),
            "tstart_mjd": grid.start_mjd, "sample_seconds": grid.integration_seconds,
            "baseline_subtracted_normalized_excess": signal, "reports": reports,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    parser.add_argument("--dm", type=float, required=True)
    parser.add_argument("--width-ms", type=float, default=5.0)
    parser.add_argument("--threshold-db", type=float, nargs="+", default=[6.0, 9.0, 12.0])
    args = parser.parse_args()
    try:
        result = evaluate(args.file, args.dm, args.width_ms, args.threshold_db)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
