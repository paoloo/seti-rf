"""Known RFI frequency allocations: GNSS, satellite comms, ISM/WiFi,
cellular, radar, broadcast, aviation.

This is a "first, rule it out as Earth" lookup table, not a decoder: given
an absolute RF frequency (Hz), :func:`lookup` reports every known
allocation that overlaps it. A hit doesn't prove the signal you saw *is*
that service (a real GPS/ADS-B verdict needs :mod:`sigint.rfi.gps` /
:mod:`sigint.rfi.adsb`'s actual waveform probes) -- it means the frequency
sits inside spectrum where that service is normally present, which is
still real, useful evidence for a human or an automated triage step.

Sources: public ICDs and spectrum-management references (IS-GPS-200,
Galileo OS-SIS-ICD, BeiDou ICD, GLONASS ICD, FCC filings for Iridium and
Starlink, ITU/national band plans for ISM/cellular/radar/broadcast). Entries
marked `region="varies"` (mostly cellular and broadcast) shift by country;
treat those as a starting point to prune/extend for your own site, not
ground truth. Frequencies are nominal band edges or channel centers plus
typical occupied bandwidth, not legal-precision allocation boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass

Origin = str    # "space" (satellite-based) | "terrestrial"


@dataclass(frozen=True)
class RFIBand:
    label: str
    lo_hz: float
    hi_hz: float
    category: str          # "GNSS" | "satcom" | "ISM" | "cellular" | "radar"
                            # | "broadcast" | "aviation"
    origin: Origin
    region: str = "global"  # "global" | "varies" | a specific region note
    notes: str = ""

    @property
    def center_hz(self) -> float:
        return (self.lo_hz + self.hi_hz) / 2.0

    def contains(self, freq_hz: float) -> bool:
        return self.lo_hz <= freq_hz <= self.hi_hz


def _mhz(lo: float, hi: float) -> tuple[float, float]:
    return lo * 1e6, hi * 1e6


def _ghz(lo: float, hi: float) -> tuple[float, float]:
    return lo * 1e9, hi * 1e9


_B = []


def _add(label, lo_hz, hi_hz, category, origin, region="global", notes=""):
    _B.append(RFIBand(label, lo_hz, hi_hz, category, origin, region, notes))


# --------------------------------------------------------------------------
# GNSS -- IS-GPS-200 (GPS), Galileo OS-SIS-ICD, BeiDou ICD, GLONASS ICD.
# Ranges are the main-lobe (or documented channel) occupied bandwidth
# around each published carrier; several constellations legitimately
# overlap around 1.16-1.61 GHz.
# --------------------------------------------------------------------------
_add("GPS L1 C/A", *_mhz(1574.397, 1576.443), "GNSS", "space",
     notes="carrier 1575.42 MHz, 1.023 Mcps C/A main lobe; "
           "see sigint.rfi.gps for a real acquisition probe")
_add("GPS L1 P(Y)/M", *_mhz(1563.42, 1587.42), "GNSS", "space",
     notes="wider military-code occupied band around the same carrier")
_add("GPS L2", *_mhz(1215.6, 1239.6), "GNSS", "space", notes="carrier 1227.60 MHz")
_add("GPS L5", *_mhz(1164.45, 1188.45), "GNSS", "space", notes="carrier 1176.45 MHz")
_add("GLONASS L1 (FDMA)", *_mhz(1598.0625, 1605.375), "GNSS", "space",
     notes="channels k=-7..+6 @ 0.5625 MHz spacing around 1602 MHz")
_add("GLONASS L2 (FDMA)", *_mhz(1242.9375, 1248.625), "GNSS", "space",
     notes="channels k=-7..+6 @ 0.4375 MHz spacing around 1246 MHz")
_add("Galileo E1", *_mhz(1573.42, 1577.42), "GNSS", "space",
     notes="carrier 1575.42 MHz, overlaps GPS L1")
_add("Galileo E5a", *_mhz(1164.45, 1188.45), "GNSS", "space", notes="carrier 1176.45 MHz")
_add("Galileo E5b", *_mhz(1195.14, 1219.14), "GNSS", "space", notes="carrier 1207.14 MHz")
_add("Galileo E6", *_mhz(1266.75, 1290.75), "GNSS", "space", notes="carrier 1278.75 MHz")
_add("BeiDou B1I", *_mhz(1559.052, 1563.144), "GNSS", "space", notes="carrier 1561.098 MHz")
_add("BeiDou B1C", *_mhz(1573.42, 1577.42), "GNSS", "space", notes="carrier 1575.42 MHz")
_add("BeiDou B2a", *_mhz(1164.45, 1188.45), "GNSS", "space", notes="carrier 1176.45 MHz")
_add("BeiDou B3", *_mhz(1256.52, 1280.52), "GNSS", "space", notes="carrier 1268.52 MHz")
_add("QZSS L1", *_mhz(1574.397, 1576.443), "GNSS", "space",
     notes="Japan regional augmentation, mirrors GPS L1")

# --------------------------------------------------------------------------
# Satellite comms -- FCC filings / operator ICDs.
# --------------------------------------------------------------------------
_add("Iridium", *_mhz(1616.0, 1626.5), "satcom", "space",
     notes="L-band TDD, ~41.667 kHz channels; see NEXT-PLAN.md -- no "
           "waveform probe implemented yet, band-match only")
_add("Globalstar uplink", *_mhz(1610.0, 1626.5), "satcom", "space")
_add("Globalstar downlink", *_mhz(2483.5, 2500.0), "satcom", "space")
_add("Inmarsat space-to-earth", *_mhz(1525.0, 1559.0), "satcom", "space")
_add("Inmarsat earth-to-space", *_mhz(1626.5, 1660.5), "satcom", "space")
_add("GPS/GLONASS DME/L-band ATC overlap", *_mhz(960.0, 1215.0), "aviation", "terrestrial",
     notes="DME + L-band ATC radar; a well-known real interferer near GPS L2/L5")
_add("Starlink user downlink (Ku)", *_ghz(10.7, 12.7), "satcom", "space",
     region="varies",
     notes="FCC-filed user-terminal band, subject to change across "
           "satellite generations -- treat as approximate")
_add("Starlink user uplink (Ku)", *_ghz(14.0, 14.5), "satcom", "space", region="varies")
_add("Starlink gateway (Ka)", *_ghz(17.8, 20.2), "satcom", "space", region="varies")

# --------------------------------------------------------------------------
# ISM / WiFi / Bluetooth / Zigbee.
# --------------------------------------------------------------------------
_add("ISM 433 MHz (region 1)", *_mhz(433.05, 434.79), "ISM", "terrestrial", region="ITU region 1")
_add("ISM 915 MHz (region 2)", *_mhz(902.0, 928.0), "ISM", "terrestrial", region="ITU region 2")
_add("ISM/WiFi/Bluetooth 2.4 GHz", *_mhz(2400.0, 2483.5), "ISM", "terrestrial")
_add("WiFi U-NII 5 GHz", *_mhz(5150.0, 5895.0), "ISM", "terrestrial")
_add("WiFi 6E", *_mhz(5925.0, 7125.0), "ISM", "terrestrial")

# --------------------------------------------------------------------------
# Cellular -- broad, commonly-cited ranges; genuinely region-dependent.
# --------------------------------------------------------------------------
_add("Cellular 700 MHz (LTE 12/13/14/17)", *_mhz(698.0, 806.0), "cellular", "terrestrial", region="varies")
_add("Cellular 850 MHz (GSM/CDMA)", *_mhz(824.0, 894.0), "cellular", "terrestrial", region="varies")
_add("GSM 900", *_mhz(880.0, 960.0), "cellular", "terrestrial", region="varies")
_add("GSM 1800 / DCS", *_mhz(1710.0, 1880.0), "cellular", "terrestrial", region="varies")
_add("PCS 1900", *_mhz(1850.0, 1990.0), "cellular", "terrestrial", region="varies")
_add("LTE 2600 (band 7)", *_mhz(2500.0, 2690.0), "cellular", "terrestrial", region="varies")
_add("5G C-band (n77/n78)", *_ghz(3.3, 4.2), "cellular", "terrestrial", region="varies")
_add("5G mmWave", *_ghz(24.25, 29.5), "cellular", "terrestrial", region="varies")

# --------------------------------------------------------------------------
# Radar.
# --------------------------------------------------------------------------
_add("Airport surveillance radar (ASR)", *_ghz(2.7, 2.9), "radar", "terrestrial")
_add("Weather radar (NEXRAD-class S-band)", *_ghz(2.7, 3.0), "radar", "terrestrial")
_add("SSR/ADS-B interrogation", *_mhz(1029.0, 1031.0), "radar", "terrestrial",
     notes="1030 MHz interrogation; replies are the 1090 MHz band below")
_add("ADS-B / Mode S 1090ES", *_mhz(1089.0, 1091.0), "radar", "terrestrial",
     notes="see sigint.rfi.adsb for a real preamble+CRC probe")
_add("Marine radar (X-band)", *_ghz(9.3, 9.5), "radar", "terrestrial")

# --------------------------------------------------------------------------
# Broadcast.
# --------------------------------------------------------------------------
_add("MW AM broadcast", *_mhz(0.531, 1.611), "broadcast", "terrestrial", region="varies")
_add("Shortwave broadcast", *_mhz(2.3, 26.1), "broadcast", "terrestrial", region="varies",
     notes="spread across several allocated SW broadcast sub-bands, not "
           "one contiguous block -- a coarse envelope")
_add("FM broadcast", *_mhz(87.5, 108.0), "broadcast", "terrestrial")
_add("DAB (Band III)", *_mhz(174.0, 240.0), "broadcast", "terrestrial", region="varies")
_add("UHF TV / DVB-T", *_mhz(470.0, 862.0), "broadcast", "terrestrial", region="varies")

# --------------------------------------------------------------------------
# Aviation.
# --------------------------------------------------------------------------
_add("VOR/ILS localizer", *_mhz(108.0, 118.0), "aviation", "terrestrial")
_add("VHF air traffic comm", *_mhz(118.0, 137.0), "aviation", "terrestrial")

KNOWN_BANDS: tuple[RFIBand, ...] = tuple(_B)


def lookup(freq_hz: float) -> tuple[RFIBand, ...]:
    """Every known band overlapping `freq_hz`, widest-context first."""
    hits = [b for b in KNOWN_BANDS if b.contains(freq_hz)]
    return tuple(sorted(hits, key=lambda b: b.hi_hz - b.lo_hz, reverse=True))


def lookup_range(lo_hz: float, hi_hz: float) -> tuple[RFIBand, ...]:
    """Every known band overlapping [lo_hz, hi_hz]."""
    hits = [b for b in KNOWN_BANDS if b.lo_hz <= hi_hz and b.hi_hz >= lo_hz]
    return tuple(sorted(hits, key=lambda b: b.lo_hz))
