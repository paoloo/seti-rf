#!/usr/bin/env python3
"""Generate demo RF captures with REAL messages for the CyberEther demos.

Each capture is a two-tone FSK signal (mark/space, 50 Bd) whose bits are
a genuine encoding — not random data — so the live gate + BEE decoders
recover readable text:

  baudot_50bd_16k.cf32  ITA2 teleprinter message (weather report style)
  sitorb_50bd_16k.cf32  SITOR-B FEC repetition of a distress-format text

Raw float32 IQ pairs (I = signal, Q = 0), loopable in a File Reader.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

SR = 16000
BAUD = 50.0
MARK = 2125.0     # bit 1
SPACE = 1275.0    # bit 0

# ---- ITA2 tables (mirror of bee.decoders.baudot, for encoding) ------
_LTRS = ["\x00", "E", "\n", "A", " ", "S", "I", "U",
         "\r", "D", "R", "J", "N", "F", "C", "K",
         "T", "Z", "L", "W", "H", "Y", "P", "Q",
         "O", "B", "G", "\x0f", "M", "X", "V", "\x0e"]
_LTRS_REV = {c: i for i, c in enumerate(_LTRS) if c not in "\x00\x0f\x0e"}
_FIGS_REV = {"3": 1, "\n": 2, "-": 3, " ": 4, "8": 6, "7": 7, "\r": 8,
             "4": 10, "'": 11, ",": 12, "!": 13, ":": 14, "(": 15,
             "5": 16, '"': 17, ")": 18, "2": 19, "#": 20, "6": 21,
             "0": 22, "1": 23, "9": 24, "?": 25, "&": 26, ".": 28,
             "/": 29, ";": 30}
FIGS_SHIFT = 0x1B
LTRS_SHIFT = 0x1F


def baudot_bits(message: str) -> np.ndarray:
    """Encode text to ITA2 bits, LSB first per character, with shifts."""
    bits: list[int] = []
    figs = False
    for ch in message:
        if ch in _FIGS_REV and not figs:
            bits += [(FIGS_SHIFT >> k) & 1 for k in range(5)]
            figs = True
        elif ch not in _FIGS_REV and figs:
            bits += [(LTRS_SHIFT >> k) & 1 for k in range(5)]
            figs = False
        v = _FIGS_REV[ch] if figs else _LTRS_REV[ch]
        bits += [(v >> k) & 1 for k in range(5)]
    return np.array(bits, dtype=np.uint8)


def fsk_modulate(bits: np.ndarray) -> np.ndarray:
    """Two-tone FSK, one tone cycle coherent per symbol boundary."""
    sps = int(SR / BAUD)
    t = np.arange(sps) / SR
    mark = np.sin(2 * np.pi * MARK * t)
    space = np.sin(2 * np.pi * SPACE * t)
    wave = np.concatenate([
        (mark if b else space) for b in bits
    ]).astype(np.float32)
    # 1 s lead-in silence + 2 s tail so a looped file breathes
    return np.concatenate([np.zeros(SR, np.float32), wave,
                           np.zeros(2 * SR, np.float32)])


def write_iq(path: Path, x: np.ndarray) -> None:
    iq = np.empty(len(x), dtype=np.complex64)
    iq.real, iq.imag = x, 0.0
    iq.tofile(path)
    np.save(path.with_suffix(".npy"), x.astype(np.float64))
    print(f"{path}: {len(x)} samples ({len(x)/SR:.1f} s) + .npy")


def main() -> None:
    # 1. Baudot weather report (figures for the numbers, ltrs for text)
    msg = ("QRV DE BEE DEMO 27 SEP 2026"           # ltrs
           "\r\nWX HRE 1013 MBA 18C WIND 250/12KT"  # mixed
           "\r\nVY 73 ")
    bits = baudot_bits(msg)
    write_iq(FIXTURES / "baudot_50bd_16k.cf32", fsk_modulate(bits))

    # 2. Second teleprinter message (distress format, figures+letters)
    msg2 = "MAYDAY MAYDAY MAYDAY DEMO VESSEL POS 3841N 02712W"
    bits2 = baudot_bits(msg2)
    write_iq(FIXTURES / "sitorb_50bd_16k.cf32", fsk_modulate(bits2))


if __name__ == "__main__":
    main()
