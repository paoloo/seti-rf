"""sigint.rfi -- known-RFI-source detection: GNSS, satcom, ISM/WiFi,
cellular, radar, broadcast, aviation.

Identifying a GPS or ADS-B signal is the same kind of work as identifying
a Baudot or STANAG signal -- general RF SIGINT, not something intrinsically
radio-astronomy-specific -- so it lives in `sigint` itself rather than a
separate package. The sibling `sigint-rfi` project builds the
radio-telescope-specific orchestration on top of this (multi-beam
cross-correlation, a persistent local RFI-fingerprint log, radio-astronomy
I/O formats), the same way `sigint` itself has no telescope-specific
concepts.

Two kinds of evidence, deliberately not conflated:

- :mod:`sigint.rfi.bands` -- a static frequency-allocation lookup table
  (GNSS, satellite comms, ISM/WiFi, cellular, radar, broadcast, aviation).
  A hit means "this frequency sits where that service normally lives," not
  a decode -- real, useful triage evidence on its own.
- :mod:`sigint.rfi.gps` and :mod:`sigint.rfi.adsb` -- real waveform-level
  probes (Gold-code acquisition, PPM+CRC-24 decode) that either produce a
  positive, explainable identification or self-reject, the same standard
  `sigint.hfdl` already holds itself to.
- :mod:`sigint.rfi.triage` -- ties `sigint.detect`'s carrier/band-scan
  output (via `Carrier.abs_freq_hz` / `ScanHit.abs_center_hz`) to the band
  catalog.
"""
from __future__ import annotations

from . import adsb, bands, gps, triage
from .bands import RFIBand, KNOWN_BANDS, lookup, lookup_range
from .triage import Verdict, verdict_for, triage_carrier, triage_carriers, triage_scan_hit

__all__ = ["adsb", "bands", "gps", "triage", "RFIBand", "KNOWN_BANDS", "lookup",
          "lookup_range", "Verdict", "verdict_for", "triage_carrier",
          "triage_carriers", "triage_scan_hit"]
