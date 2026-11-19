"""sigint.ham -- amateur-radio and adjacent human-made-signal detection
and decoding.

Detecting and decoding one of these is the same kind of work as
detecting and decoding any other RF signal (HFDL, GPS, ADS-B, the 76
BEE-derived HF/military formats): a waveform-level probe that either
produces a confident, checkable result or self-rejects. This package
covers the modes that weren't already served by the rest of `sigint`
(RTTY is just Baudot/ITA2 FSK, already covered by `sigint.decoders` +
`sigint.dsp.fsk_demod_sync`; AX.25/APRS packet framing is already in the
BEE-derived decoder catalog):

- :mod:`sigint.ham.morse` -- CW: keying-timing detection and decode,
  clustering the actual on/off durations seen rather than assuming a
  fixed WPM (a human keys it, so there's no fixed symbol clock).
- :mod:`sigint.ham.psk31` -- PSK31/PSK63: differential BPSK, Varicode
  text framing (the real 256-entry table, from fldigi's production
  source).
- :mod:`sigint.ham.olivia` -- Olivia/Contestia: incoherent MFSK with
  Olivia's real Walsh-Hadamard FEC (Pawel Jalocha's algorithm, from
  gmfsk's real implementation) as an optional layer.
- :mod:`sigint.ham.sstv` -- SSTV: VIS header detect/decode (identifies
  the mode) plus full image encode/decode for Martin M1, Scottie S1 and
  Robot 36 (real VIS codes and per-mode timing, from pySSTV's real
  encoder).
- :mod:`sigint.ham.ft8` -- FT8/FT4 (the WSJT-X weak-signal family):
  Costas sync, the real LDPC(174,91) parity-check/generator matrices, and
  the standard callsign+grid message format -- all transcribed
  programmatically from a real, working reference decoder (see the
  module's own docstring for the exact provenance and what's out of
  scope).
- :mod:`sigint.ham.wspr` -- WSPR: the third WSJT-X mode, a K=32
  convolutional code decoded via a bounded-beam trellis search rather
  than infeasible full Viterbi (real, verified encoder; less processing
  gain than WSJT-X's own tuned sequential decoder).
- :mod:`sigint.ham.ais` -- AIS: maritime position reports over standard
  HDLC framing (reusing `sigint._bee.decoders.hdlc`'s already-tested
  bit-destuffing and CRC-16/X-25 directly).
- :mod:`sigint.ham.pocsag` -- POCSAG: pager addresses and alphanumeric/
  numeric text, BCH(31,21)-checked codewords.
- :mod:`sigint.ham.rds` -- RDS/RBDS: the FM broadcast subcarrier data
  service (station name, PTY, TP/TA) essentially every FM station sends.
- :mod:`sigint.ham.acars` -- ACARS: the aircraft VHF datalink, HFDL's
  much simpler and much more common cousin.
- :mod:`sigint.ham.apt` -- NOAA APT: weather-satellite images (AM
  subcarrier, not SSTV's FM -- a different mode entirely despite both
  being "an image over audio").
- :mod:`sigint.ham.trunked` -- DMR/P25/NXDN: frame-sync detection and
  cleartext metadata (P25's NAC/DUID) for the common digital
  trunked-radio families -- deliberately not voice decode (the vocoders
  are patented), by design, not as a limitation to work around.

Every module documents its own validation level and honest limitations
in its docstring -- some of this family (Olivia's FEC, FT8's LDPC,
WSPR's trellis search) is a real, verified coding-gain implementation;
some (this package's demod stages generally) trades the last decade of
production-receiver engineering for a correct, testable, independently
useful first version.
"""
from __future__ import annotations

from . import acars, ais, apt, ft8, morse, olivia, pocsag, psk31, rds, sstv, trunked, wspr

__all__ = ["acars", "ais", "apt", "ft8", "morse", "olivia", "pocsag", "psk31",
          "rds", "sstv", "trunked", "wspr"]
