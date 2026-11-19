import json
import subprocess
import sys

import h5py
import numpy as np
import pytest

from sigint_rfi.filterbank import PairedFilterbanks


def write_filterbank(path, values, **metadata):
    defaults = dict(tstart=60859.0, tsamp=16.78, fch1=1090.0, foff=-0.000002)
    with h5py.File(path, "w") as file:
        data = file.create_dataset("data", data=values[:, None, :], compression="gzip")
        data.attrs.update(defaults | metadata)


def test_shared_track_flags_and_target_track_survives(tmp_path):
    on = np.ones((5, 33))
    off = np.full_like(on, 10.0)
    for row in range(4):
        on[row, 6 + row] = 100        # on-target-only drifting injection
        on[row, 15 + row] = 100       # shared drifting interference
        off[row, 16 + row] = 1000    # one-channel offset, including a block boundary
    on[4, 32] = 100                  # final short block
    off[4, 32] = np.nan
    paths = [tmp_path / name for name in ("on.h5", "off.h5")]
    write_filterbank(paths[0], on)
    write_filterbank(paths[1], off)
    with h5py.File(paths[0]) as on_file, h5py.File(paths[1]) as off_file:
        pair = PairedFilterbanks(on_file, off_file)
        hits = list(pair.candidates(tolerance_channels=1, block_channels=16))
        assert {(hit.row, hit.channel) for hit in hits if hit.coincident} == {
            (row, 15 + row) for row in range(4)
        }
        assert {(hit.row, hit.channel) for hit in hits if not hit.coincident} == {
            (row, 6 + row) for row in range(4)
        } | {(4, 32)}
        assert hits[0].frequency_hz == pytest.approx(1_090_000_000 - 12)
        assert any("ADS-B" in label for label in hits[0].catalog_matches)
        window = list(pair.candidates(rows=(4, 5), channels=(32, 33), block_channels=16))
        assert window == hits[-1:]

    result = subprocess.run(
        [sys.executable, "-m", "sigint_rfi.cli", *map(str, paths),
         "--tolerance-channels", "1", "--block-channels", "16"],
        capture_output=True, text=True, check=True,
    )
    report = [json.loads(line) for line in result.stdout.splitlines()]
    assert report[0]["grid"]["channel_step_mhz"] < 0
    assert report[-1] == {"type": "summary", "candidates": 9, "coincident": 4}


@pytest.mark.parametrize("metadata", [{"tstart": 60859.001}, {"foff": 0.000002}])
def test_mismatched_grids_are_rejected(tmp_path, metadata):
    shape = np.ones((2, 32))
    on, off = tmp_path / "on.h5", tmp_path / "off.h5"
    write_filterbank(on, shape)
    write_filterbank(off, shape, **metadata)
    with h5py.File(on) as on_file, h5py.File(off) as off_file:
        with pytest.raises(ValueError, match="identical"):
            PairedFilterbanks(on_file, off_file)
        with pytest.raises(ValueError, match="separate"):
            PairedFilterbanks(on_file, on_file)


def test_masks_exclude_bad_input_pixels(tmp_path):
    on = np.ones((1, 16))
    off = np.ones_like(on)
    on[0, 5:7] = off[0, 5:7] = 100
    paths = [tmp_path / name for name in ("on.h5", "off.h5")]
    for path, values, flagged in zip(paths, (on, off), (5, 6)):
        write_filterbank(path, values)
        with h5py.File(path, "a") as file:
            mask = np.zeros((1, 1, 16), dtype=np.uint8)
            mask[0, 0, flagged] = 1
            file.create_dataset("mask", data=mask)
    with h5py.File(paths[0]) as on_file, h5py.File(paths[1]) as off_file:
        hits = list(PairedFilterbanks(on_file, off_file).candidates())
    assert [hit.channel for hit in hits] == [6]
    assert not hits[0].coincident
