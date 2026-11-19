"""ITA2 / Baudot-Murray (CCITT No. 2) 5-bit teleprinter code (Tier A).

Public, unambiguous standard. Each character is 5 bits, transmitted LSB
first, with two shift states (LTRS/FIGS) toggled by dedicated codes — the
same code can mean a letter or a digit/punctuation mark depending on which
shift state was last selected.
"""
from __future__ import annotations

import numpy as np

from sigint._bee.decoders.base import DecodeResult, Decoder
from sigint._bee.decoders.registry import register

_LTRS = [
    "\x00", "E", "\n", "A", " ", "S", "I", "U",
    "\r", "D", "R", "J", "N", "F", "C", "K",
    "T", "Z", "L", "W", "H", "Y", "P", "Q",
    "O", "B", "G", "\x0f", "M", "X", "V", "\x0e",
]
_FIGS = [
    "\x00", "3", "\n", "-", " ", "\x07", "8", "7",
    "\r", "\x05", "4", "'", ",", "!", ":", "(",
    "5", '"', ")", "2", "#", "6", "0", "1",
    "9", "?", "&", "\x0f", ".", "/", ";", "\x0e",
]
_FIGS_SHIFT = 0x1B  # 11011
_LTRS_SHIFT = 0x1F  # 11111


class BaudotDecoder(Decoder):
    name = "Baudot/ITA2"
    description = "5-bit ITA2 teleprinter code, LSB first, LTRS/FIGS shift states"

    def decode(self, bits: np.ndarray, block=None) -> DecodeResult:
        n_symbols = bits.size // 5
        trimmed = bits[: n_symbols * 5].reshape(n_symbols, 5)
        weights = (1 << np.arange(5)).astype(np.int64)  # LSB first
        values = trimmed.astype(np.int64) @ weights

        chars = []
        figs_mode = False
        for v in values:
            if v == _FIGS_SHIFT:
                figs_mode = True
                continue
            if v == _LTRS_SHIFT:
                figs_mode = False
                continue
            chars.append((_FIGS if figs_mode else _LTRS)[v])
        text = "".join(chars)
        return DecodeResult(text=text, symbol_count=n_symbols, error_count=0)


register(BaudotDecoder())

_baudot_alias = BaudotDecoder()
_baudot_alias.name = "BAUDOT"  # matches Bee.exe's own decoder name string
register(_baudot_alias)
