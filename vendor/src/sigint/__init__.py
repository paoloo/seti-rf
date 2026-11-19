"""sigint -- combined RF signal-intelligence suite.

Self-contained synthesis of the two recovered Python codebases:

- DSP surface from the Signals Analyzer (SA v6.2.6.7) clean-room port
  (windowed sliding-FFT spectrum, FSK baud measurement and demodulation,
  phase plane, in-band inversion, OFDM CP-correlation sync), reimplemented
  here so `sigint` has no import-time dependency on `XTRA`.
- Bitstream decoding conventions from the BEE project (registered decoders,
  DecodeResult/DecodeArtifact, 76 signal formats from Baudot and ALE to
  MIL-STD-188-110A and STANAG-4285), bridged through a thin adapter that
  works on raw captures.

On top of both sits what a human RF analyst actually needs and neither
tool provided together: automatic signal detection (energy/PSTN carrier
find, SNR estimation, per-band scan), a modulation classifier, synthetic
dataset generation with known ground truth for validating the detection
stack, dataset integrity validation, and one tkinter+ttk GUI that shows
waterfall, instantaneous spectrum, and detection/decode results side by
side.

Layout:
    sigint.dsp          -- spectra, waterfalls, demodulation primitives
    sigint.detect       -- signal detection, SNR, classification, band scan
    sigint.decoders     -- bitstream decoders (bridge over bee's registry)
    sigint.dataset      -- synthetic RF generation + dataset validation
    sigint.gui          -- tkinter+ttk analyst workbench
    sigint.rfi          -- known-RFI-source detection: a frequency-band
                           catalog (GNSS/satcom/ISM/cellular/radar/
                           broadcast/aviation) plus real GPS L1 C/A and
                           ADS-B/Mode S waveform-level probes, tied to
                           sigint.detect's carrier/band-scan output via
                           the optional absolute-frequency `center_hz` on
                           find_carriers/band_scan/GateConfig
    sigint.ham          -- amateur-radio digital modes not already
                           covered elsewhere: CW/Morse, PSK31/PSK63,
                           Olivia/Contestia, SSTV, and FT8/FT4
"""
from __future__ import annotations

__version__ = "0.2.0"
