"""8-bit ASCII decoder (Tier A: unambiguous public standard)."""
from __future__ import annotations

import numpy as np

from sigint._bee.decoders.base import DecodeResult, Decoder
from sigint._bee.decoders.registry import register


class AsciiDecoder(Decoder):
    name = "ASCII"
    description = "8-bit groups, MSB first, printable range 0x20-0x7E"

    def decode(self, bits: np.ndarray, block=None) -> DecodeResult:
        n_symbols = bits.size // 8
        trimmed = bits[: n_symbols * 8].reshape(n_symbols, 8)
        weights = (1 << np.arange(7, -1, -1)).astype(np.int64)
        values = trimmed.astype(np.int64) @ weights

        chars = []
        errors = 0
        for v in values:
            if 32 <= v < 127:
                chars.append(chr(v))
            else:
                chars.append(".")
                errors += 1
        return DecodeResult(text="".join(chars), symbol_count=n_symbols, error_count=errors)


register(AsciiDecoder())
