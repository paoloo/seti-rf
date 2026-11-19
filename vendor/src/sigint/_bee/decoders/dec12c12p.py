"""12s12p message-start marker, recovered from 003.dll's Delphi VMT.

`12s12p` (`003.dll`) is documented by the original application itself as
having a non-determinable period (see `bee/decoders/unimplemented.py` and
GUIDE.md), and, like `MEROD-USA` before it, its Delphi/Borland `CreateDec`
export does not expose a directly address-referenced one-bit-per-byte
constant table -- the earlier constant-table sweep of this DLL found only
five short, unreferenced runs and correctly moved on. The Delphi VMT route
that recovered `MEROD-USA` (see `bee/decoders/merod.py`) finds it instead:

`CreateDec` (`0x0044d3cc`) leads through a constructor chain
(`0x0044d0bc`) to the class VMT at `0x0044d020` (class name `TDec12c12p`,
confirmed from a string literal in the binary). Its `+0x54` virtual method
(`0x0044d148`) compares an ASCII '0'/'1' digit stream (read through the
object at `self+0x34`) against one of two 63-bit Delphi ansistring
constants selected by a polarity flag: `0x0044cf4c` and `0x0044cf94`. Those
two strings are exact bitwise complements of each other -- the DLL stores
both polarities of a single marker rather than computing the complement at
runtime, the same pattern used for `MEROD-FIN`/`MEROD-IND`'s
normal/inverted threshold check.

`0x0044cf94` matches the bundled `12s12p.bin` capture **exactly**, 5 times,
LSB-first only (0 hits MSB-first) -- the same decisive signature that
recovered every MEROD name. Negative controls are clean: 0 exact hits
(best 45-48/63) against `Merod (usa).bin`, `Merod (fin).BIN`,
`DUP-FEC-2.bin`, and 60,000 bits of random noise.

Unlike MEROD-USA's four identically-sized messages, the five marker hits
here (`[1256, 1900, 2928, 4717, 9012]`) are not evenly spaced (gaps of 644,
1028, 1789 and 4295 bits), so no frame period is asserted here -- this is a
repeated message-start marker, not a fixed-length frame, closer to `ALIS`'s
single in-frame marker than to MEROD's prefix+payload+end walk. The `+0x50`
virtual method (`0x0044d1c4`) that would determine each message's length
dispatches on a value from `fcn.0044d100` through several distinct cases (a
cascading type check reminiscent of Ital HDLC's frame-length table) that
this pass did not fully trace -- message end/length recovery is the
concrete next step if this is resumed.

Scope: message-start synchronization only. Message length/end and the
payload alphabet remain unrecovered.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sigint._bee.decoders.base import DecodeResult, Decoder
from sigint._bee.decoders.registry import register

#: The 0x0044cf94 ansistring constant in 003.dll. 0x0044cf4c is its exact
#: bitwise complement (the DLL's stored inverted-polarity variant), so both
#: polarities are searched from this single array rather than kept separate.
MARKER = np.array([int(c) for c in
                    "111111010101100110111011010010011100010111100101000110000100000"],
                   dtype=np.uint8)
MARKER_VA = 0x0044CF94
DLL = "003.dll"


def _reverse_each_byte(bits: np.ndarray) -> np.ndarray:
    complete = len(bits) // 8 * 8
    out = bits.copy()
    if complete:
        out[:complete] = bits[:complete].reshape(-1, 8)[:, ::-1].reshape(-1)
    return out


@dataclass(frozen=True)
class TwelveS12PHit:
    offset: int
    inverted: bool
    orientation: str


def _find(bits: np.ndarray, orientation: str) -> list[TwelveS12PHit]:
    if len(bits) < len(MARKER):
        return []
    windows = np.lib.stride_tricks.sliding_window_view(bits, len(MARKER))
    normal = np.flatnonzero((windows == MARKER).all(axis=1))
    inverted = np.flatnonzero((windows == 1 - MARKER).all(axis=1))
    hits = [TwelveS12PHit(int(offset), False, orientation) for offset in normal]
    hits += [TwelveS12PHit(int(offset), True, orientation) for offset in inverted]
    return hits


def find_12s12p_markers(bits: np.ndarray) -> list[TwelveS12PHit]:
    """Locate exact occurrences of the 63-bit `TDec12c12p` marker (either
    stored polarity), searching both byte orientations."""
    stream = np.asarray(bits, dtype=np.uint8).reshape(-1) & 1
    hits: list[TwelveS12PHit] = []
    for oriented, orientation in ((stream, "as-given"),
                                  (_reverse_each_byte(stream), "byte-reversed")):
        hits.extend(_find(oriented, orientation))
    return sorted(hits, key=lambda item: (item.orientation, item.inverted, item.offset))


class TwelveS12PDecoder(Decoder):
    name = "12s12p"
    description = "63-bit message-start marker (Delphi VMT, payload unrecovered)"
    status = "partial"
    confidence = "high"
    layer = "synchronization"
    provenance = (f"static analysis of original Lib/{DLL} (Delphi VMT slot +0x54 off the "
                  "CreateDec-constructed TDec12c12p object) with the 63-bit ansistring "
                  "marker confirmed exactly on the bundled capture")

    def decode(self, bits: np.ndarray, block=None) -> DecodeResult:
        hits = find_12s12p_markers(bits)
        text = "\n".join(
            f"@{item.offset}b marker={len(MARKER)}b "
            f"polarity={'inverted' if item.inverted else 'normal'} "
            f"orientation={item.orientation}" for item in hits)
        return DecodeResult(
            text=text,
            symbol_count=len(hits),
            error_count=0 if hits else 1,
            notes=(f"{len(hits)} exact occurrence(s) of the 63-bit message-start marker "
                   f"at {DLL}:0x{MARKER_VA:08x}. Message length/end and payload alphabet "
                   "are not recovered -- see bee/decoders/dec12c12p.py."),
        )


register(TwelveS12PDecoder())
