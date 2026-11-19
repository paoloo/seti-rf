"""Core bit-level data model for a loaded capture.

A :class:`Bitstream` wraps a raw byte buffer and exposes it as an addressable
array of individual bits (0/1), matching how BEE treats any file: not as
structured records, but as a flat sequence of bits that gets reshaped by a
"period" for visual inspection.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

BitOrder = str  # "msb" or "lsb"

_NP_BITORDER = {"msb": "big", "lsb": "little"}


class Bitstream:
    """A raw byte capture exposed as an addressable array of individual bits."""

    def __init__(self, data: bytes, bit_order: BitOrder = "msb"):
        _validate_bit_order(bit_order)
        self._data = bytes(data)
        self._bit_order = bit_order
        self._bits = _unpack(self._data, bit_order)

    @classmethod
    def from_file(cls, path: str | Path, bit_order: BitOrder = "msb") -> "Bitstream":
        return cls(Path(path).read_bytes(), bit_order=bit_order)

    @property
    def bit_order(self) -> BitOrder:
        return self._bit_order

    @bit_order.setter
    def bit_order(self, value: BitOrder) -> None:
        _validate_bit_order(value)
        if value != self._bit_order:
            self._bit_order = value
            self._bits = _unpack(self._data, value)

    @property
    def raw(self) -> bytes:
        """The original, untouched byte buffer as loaded from disk."""
        return self._data

    @property
    def bits(self) -> np.ndarray:
        """The full bit array (uint8, values 0/1), ordered per :attr:`bit_order`."""
        return self._bits

    @property
    def n_bytes(self) -> int:
        return len(self._data)

    def __len__(self) -> int:
        return self._bits.size

    def bits_slice(self, start: int, length: int) -> np.ndarray:
        """Bits [start, start+length), zero-padded past the end of the stream."""
        end = start + length
        if start >= self._bits.size:
            return np.zeros(length, dtype=np.uint8)
        chunk = self._bits[start:end]
        if chunk.size < length:
            chunk = np.pad(chunk, (0, length - chunk.size))
        return chunk

    def reshape(
        self,
        period: int,
        start: int = 0,
        polarity_invert: bool = False,
    ) -> np.ndarray:
        """Reshape bits ``[start:]`` into rows of ``period`` bits.

        This is the data behind the "Graph" matrix view: each row is one
        repetition of the signal period, so periodic structure (sync words,
        framing) lines up in columns. The final row is zero-padded if the
        stream length isn't a multiple of ``period``.
        """
        if period <= 0:
            raise ValueError("period must be positive")
        bits = self._bits[start:]
        if polarity_invert:
            bits = 1 - bits
        n = bits.size
        n_rows = -(-n // period) if n else 1  # ceil div, at least one row
        padded = np.zeros(n_rows * period, dtype=np.uint8)
        padded[:n] = bits
        return padded.reshape(n_rows, period)

    def to_bytes(self) -> bytes:
        """Repack the (possibly edited) bit array back into bytes."""
        return pack_bits(self._bits, self._bit_order)

    def byte_at_bit(self, bit_index: int) -> int | None:
        """The raw byte value containing ``bit_index``, or None past the end."""
        byte_index = bit_index // 8
        if 0 <= byte_index < len(self._data):
            return self._data[byte_index]
        return None


def _validate_bit_order(bit_order: BitOrder) -> None:
    if bit_order not in _NP_BITORDER:
        raise ValueError(f"bit_order must be 'msb' or 'lsb', got {bit_order!r}")


def _unpack(data: bytes, bit_order: BitOrder) -> np.ndarray:
    if not data:
        return np.zeros(0, dtype=np.uint8)
    return np.unpackbits(
        np.frombuffer(data, dtype=np.uint8), bitorder=_NP_BITORDER[bit_order]
    )


def pack_bits(bits: np.ndarray, bit_order: BitOrder = "msb") -> bytes:
    """Pack a 0/1 bit array back into bytes, zero-padding to a byte boundary."""
    _validate_bit_order(bit_order)
    return np.packbits(bits, bitorder=_NP_BITORDER[bit_order]).tobytes()
