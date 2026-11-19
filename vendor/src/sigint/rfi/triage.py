"""Tie sigint.detect's carrier/band-scan output to the known-RFI-band
catalog: the actual "first, rule it out as Earth" step.

Uses `Carrier.abs_freq_hz` / `ScanHit.abs_center_hz` directly -- `sigint`
itself carries the absolute-RF-frequency translation (an optional
`center_hz` on `detect.find_carriers`/`band_scan` and
`realtime.GateConfig`, additive and 0.0 by default), so this module is
just the catalog lookup on top of it, not a second translation layer.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..detect import Carrier, ScanHit

from .bands import RFIBand, lookup


@dataclass(frozen=True)
class Verdict:
    freq_hz: float
    bands: tuple[RFIBand, ...]

    @property
    def has_catalog_match(self) -> bool:
        return bool(self.bands)

    @property
    def is_known(self) -> bool:
        """Compatibility alias for a frequency allocation match."""
        return self.has_catalog_match

    @property
    def origins(self) -> tuple[str, ...]:
        return tuple(sorted({b.origin for b in self.bands}))

    def summary(self) -> str:
        if not self.bands:
            return "unmatched"
        return ", ".join(b.label for b in self.bands)


def verdict_for(freq_hz: float) -> Verdict:
    return Verdict(freq_hz=freq_hz, bands=lookup(freq_hz))


def triage_carrier(carrier: Carrier) -> Verdict:
    return verdict_for(carrier.abs_freq_hz)


def triage_scan_hit(hit: ScanHit) -> Verdict:
    return verdict_for(hit.abs_center_hz)


def triage_carriers(carriers: tuple[Carrier, ...]) -> tuple[tuple[Carrier, Verdict], ...]:
    """Every carrier paired with its RFI-band verdict, unmatched ones
    (candidates worth a second look) sorted first."""
    pairs = [(c, triage_carrier(c)) for c in carriers]
    return tuple(sorted(pairs, key=lambda p: p[1].has_catalog_match))
