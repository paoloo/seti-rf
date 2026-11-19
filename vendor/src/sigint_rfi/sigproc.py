"""Read bounded slices from byte-aligned, single-plane SIGPROC filterbanks."""
from __future__ import annotations

import os
from pathlib import Path
import struct

import numpy as np

_STRINGS = {"source_name", "rawdatafile"}
_INTS = {
    "machine_id", "telescope_id", "data_type", "nchans", "nbeams", "ibeam",
    "nbits", "nifs", "barycentric", "pulsarcentric", "sumifs", "nsamples",
}
_DOUBLES = {
    "az_start", "za_start", "src_raj", "src_dej", "fch1", "foff", "tstart",
    "tsamp", "refdm", "period",
}


def _read_header(file) -> dict[str, int | float | str]:
    def unpack(fmt):
        size = struct.calcsize(fmt)
        raw = file.read(size)
        if len(raw) != size:
            raise ValueError("truncated SIGPROC header")
        return struct.unpack(fmt, raw)[0]

    def string():
        length = unpack("<i")
        if not 0 <= length <= 4096:
            raise ValueError("invalid SIGPROC header string length")
        raw = file.read(length)
        if len(raw) != length:
            raise ValueError("truncated SIGPROC header string")
        return raw.decode("ascii")

    if string() != "HEADER_START":
        raise ValueError("missing SIGPROC HEADER_START")
    header: dict[str, int | float | str] = {}
    for _ in range(256):
        key = string()
        if key == "HEADER_END":
            return header
        if key in header:
            raise ValueError(f"duplicate SIGPROC header field {key!r}")
        if key in _STRINGS:
            header[key] = string()
        elif key in _INTS:
            header[key] = unpack("<i")
        elif key in _DOUBLES:
            header[key] = unpack("<d")
        elif key == "signed":
            header[key] = unpack("<b")
        else:
            raise ValueError(f"unsupported SIGPROC header field {key!r}")
    raise ValueError("SIGPROC header exceeds 256 fields")


class SigprocFilterbank:
    """Read an 8/16-bit integer or 32-bit float SIGPROC spectrogram."""

    def __init__(self, path: str | Path):
        self.filename = os.fspath(path)
        self._file = open(path, "rb")
        try:
            self.attrs = _read_header(self._file)
            if self.attrs.get("data_type") != 1 or self.attrs.get("nifs") != 1:
                raise ValueError("expected single-plane SIGPROC filterbank power")
            bits = self.attrs.get("nbits")
            signed = self.attrs.get("signed", 0)
            if bits not in (8, 16, 32) or signed not in (0, 1):
                raise ValueError("expected byte-aligned 8/16-bit integers or 32-bit float")
            self.dtype = np.dtype("<f4" if bits == 32 else f"<{'i' if signed else 'u'}{bits // 8}")
            channels = self.attrs.get("nchans", 0)
            if channels <= 0:
                raise ValueError("invalid SIGPROC channel count")
            self._offset = self._file.tell()
            rows, remainder = divmod(
                os.fstat(self._file.fileno()).st_size - self._offset,
                channels * self.dtype.itemsize,
            )
            if not rows or remainder:
                raise ValueError("empty or incomplete SIGPROC integrations")
            self.shape = (rows, 1, channels)
        except BaseException:
            self._file.close()
            raise

    def __enter__(self) -> SigprocFilterbank:
        return self

    def __exit__(self, *exc) -> None:
        self._file.close()

    def read(self, row: int, start: int, stop: int) -> np.ndarray:
        rows, _, channels = self.shape
        if not 0 <= row < rows or not 0 <= start <= stop <= channels:
            raise ValueError("SIGPROC slice lies outside the data")
        offset = self._offset + (row * channels + start) * self.dtype.itemsize
        self._file.seek(offset)
        values = np.fromfile(self._file, dtype=self.dtype, count=stop - start)
        if len(values) != stop - start:
            raise ValueError("truncated SIGPROC integration")
        return values
