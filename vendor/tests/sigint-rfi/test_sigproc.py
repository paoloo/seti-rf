import json
import struct
import subprocess
import sys

import numpy as np
import pytest

from sigint_rfi.filterbank import FilterbankScan
from sigint_rfi.sigproc import SigprocFilterbank


def _text(value):
    raw = value.encode("ascii")
    return struct.pack("<i", len(raw)) + raw


def write_filterbank(path, data, bits=32):
    ints = {"data_type": 1, "nchans": data.shape[1], "nifs": 1, "nbits": bits}
    doubles = {"fch1": 1500.0, "foff": -0.5, "tstart": 59872.0, "tsamp": 0.001}
    with path.open("wb") as file:
        file.write(_text("HEADER_START"))
        file.write(_text("rawdatafile") + _text(""))
        for key, value in ints.items():
            file.write(_text(key) + struct.pack("<i", value))
        for key, value in doubles.items():
            file.write(_text(key) + struct.pack("<d", value))
        file.write(_text("HEADER_END"))
        file.write(data.tobytes())


@pytest.mark.parametrize(("bits", "dtype"), [(8, "u1"), (16, "<u2"), (32, "<f4")])
def test_sigproc_scan_uses_signed_frequency_and_selected_rows(tmp_path, bits, dtype):
    data = np.ones((4, 64), dtype=dtype)
    data[2, 13] = 32
    path = tmp_path / "ata.fil"
    write_filterbank(path, data, bits=bits)
    with SigprocFilterbank(path) as file:
        assert np.array_equal(file.read(2, 12, 15), [1, 32, 1])
        scan = FilterbankScan(file)
        hit, = scan.candidates(rows=(2, 3), channels=(12, 15))
        assert (hit.row, hit.channel, hit.frequency_hz) == (2, 13, 1_493_500_000.0)
        assert hit.off_channel is None


def test_single_beam_cli_leaves_coincidence_unevaluated(tmp_path):
    path = tmp_path / "ata.fil"
    data = np.ones((2, 64), dtype="<f4")
    data[0, 5] = 32
    write_filterbank(path, data)
    completed = subprocess.run(
        [sys.executable, "-m", "sigint_rfi.cli", str(path)],
        capture_output=True, text=True, check=True,
    )
    lines = [json.loads(line) for line in completed.stdout.splitlines()]
    assert lines[1]["coincident"] is None
    assert lines[2] == {"type": "summary", "candidates": 1, "coincident": None}


def test_incomplete_sigproc_row_is_rejected(tmp_path):
    path = tmp_path / "short.fil"
    write_filterbank(path, np.ones((2, 64), dtype="<f4"))
    with path.open("ab") as file:
        file.write(b"\x00")
    with pytest.raises(ValueError, match="incomplete"):
        SigprocFilterbank(path)
