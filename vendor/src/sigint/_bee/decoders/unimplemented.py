"""Placeholder registrations for names with no recovered decode algorithm.

For these, neither a public specification nor a feasible from-scratch
static-analysis recovery of the corresponding ``Lib/*.dll`` (Delphi/MSVC
binaries with no useful embedded strings or symbols -- see the project's
research notes) was achieved. Rather than fabricate a plausible-looking but
unverified decode -- actively harmful for a signal-analysis tool, since
wrong output that *looks* right is worse than no output -- these register a
decoder that says so plainly and points at the manual workflow (Graph/Hex/
Find views, matching the original tool's own "eyeball the bit matrix,
locate sync, then apply a decoder" design) that still works for them today.

The most reliable path to a real implementation for any of these is the
ground-truth oracle described in the project plan (run the real Bee.exe
in a Windows VM against the matching Signals/*.bin sample and compare).
"""
from __future__ import annotations

import numpy as np

from sigint._bee.decoders.base import DecodeResult, Decoder
from sigint._bee.decoders.registry import register

#: Empty as of 2026-08-15: `12s12p`, the last member, was recovered via the
#: Delphi VMT route (see `bee/decoders/dec12c12p.py`). Kept as live
#: infrastructure -- a name that fails both period recovery and VMT tracing
#: in a future session belongs here, registered honestly rather than left
#: out of the inventory.
_NAMES: list[str] = []

#: The original application's own help file (`Bee.Demo/BeeR.chm`,
#: `hlp_vwold.htm`) lists `MEROD-USA` and `12s12p` (among others) under
#: "protocols with non-determinable periods", saying they "can be identified
#: only by their structure, or by brute force", and gives each a displayed
#: period (`MEROD-USA` 56, `12s12p` 48 bits) that does not survive
#: verification against its own bundled capture: with the documented
#: LSB-first bit packing, the multiples of each documented period are no
#: better than randomly placed in the autocorrelation ranking (`p >= 0.15`
#: for `12s12p`, `p = 1.0` for `MEROD-USA`; see
#: `documented_period.harmonic_rank_evidence`). That agreement between the
#: original documentation and this project's measurements -- no *period* is
#: findable for either -- is exactly what `MEROD-FIN`/`MEROD-IND`/`MEROD-USA`
#: and `12s12p` turned out to have in common: all four are framed by a
#: **marker**, not a period, recovered from their DLLs instead (see
#: `bee/decoders/merod.py` and `bee/decoders/dec12c12p.py`). Kept here for
#: any future name that lands in `_NAMES` with a documented-but-unverified
#: period of its own.
_NON_DETERMINABLE_PERIOD: dict[str, int] = {}


class UnimplementedDecoder(Decoder):
    description = "No recovered algorithm -- see module docstring"
    status = "unrecovered"
    confidence = "unrecovered"
    layer = "none"
    provenance = "original decoder name only; algorithm not yet recovered"

    def decode(self, bits: np.ndarray, block=None) -> DecodeResult:
        documented = _NON_DETERMINABLE_PERIOD.get(self.name)
        extra = ""
        if documented:
            extra = (f" The original help file lists '{self.name}' among the "
                     "protocols with non-determinable periods (identifiable only by "
                     f"structure or brute force) and gives a displayed period of "
                     f"{documented} bits, which does not verify against the bundled "
                     "capture -- see bee/decoders/unimplemented.py and GUIDE.md.")
        return DecodeResult(
            text="",
            symbol_count=0,
            error_count=0,
            notes=(
                f"No decode algorithm recovered for '{self.name}' (no public spec "
                "found, and static analysis of the original Lib DLL didn't yield a "
                "recoverable algorithm in the time available -- see "
                "bee/decoders/unimplemented.py). Use the Graph/Hex/Sig views and "
                "Find dialog to locate sync words and framing by hand, the same "
                "workflow the original tool was built around." + extra
            ),
        )


for _name in _NAMES:
    _decoder = UnimplementedDecoder()
    _decoder.name = _name
    register(_decoder)
