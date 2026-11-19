"""Hamming(7,4) single-error-correcting code (Tier A: public standard).

Each 7-bit codeword (positions 1-7, parity bits at positions 1/2/4) encodes
4 data bits (positions 3/5/6/7). A single-bit error is detected and
corrected via the syndrome; data nibbles are packed two-per-byte and shown
as hex.
"""
from __future__ import annotations

import numpy as np

from sigint._bee.decoders.base import DecodeResult, Decoder
from sigint._bee.decoders.registry import register

_DATA_POS = (2, 4, 5, 6)  # 0-indexed positions 3, 5, 6, 7


def _decode_block(bits7: np.ndarray) -> tuple[list[int], bool]:
    syndrome = 0
    for i in range(3):
        parity = 0
        for pos in range(1, 8):
            if pos & (1 << i):
                parity ^= int(bits7[pos - 1])
        syndrome |= parity << i
    corrected = bits7.copy()
    had_error = syndrome != 0
    if had_error:
        corrected[syndrome - 1] ^= 1
    data = [int(corrected[p]) for p in _DATA_POS]
    return data, had_error


class Hamming74Decoder(Decoder):
    name = "Hamming(7,4)"
    description = "7-bit codewords, single-error-correcting, 4 data bits/block"

    def decode(self, bits: np.ndarray, block=None) -> DecodeResult:
        n_blocks = bits.size // 7
        trimmed = bits[: n_blocks * 7].reshape(n_blocks, 7)

        nibbles: list[int] = []
        corrected_count = 0
        for row in trimmed:
            data_bits, had_error = _decode_block(row)
            value = (data_bits[0] << 3) | (data_bits[1] << 2) | (data_bits[2] << 1) | data_bits[3]
            nibbles.append(value)
            corrected_count += int(had_error)

        hex_text = "".join(f"{n:X}" for n in nibbles)
        return DecodeResult(
            text=hex_text,
            symbol_count=n_blocks,
            error_count=corrected_count,
            notes=f"{corrected_count} block(s) had a corrected single-bit error",
        )


register(Hamming74Decoder())

_hamming_alias = Hamming74Decoder()
_hamming_alias.name = "Hamming"  # matches Bee.exe's own decoder name string
register(_hamming_alias)
