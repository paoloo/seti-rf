"""AIS (Automatic Identification System): maritime position reports.

The physical/link layer is standard HDLC -- NRZI-encoded, bit-stuffed,
flag-delimited (``0x7E``), CRC-16/X-25 FCS -- exactly what
:mod:`sigint._bee.decoders.hdlc` (the same framing AX.25 packet radio
uses) already implements and tests, so this module reuses it directly
rather than re-deriving bit-destuffing and the FCS a second time. What's
new here is AIS-specific: NRZI decode (HDLC frames are bit-stuffed
*before* NRZI encoding, so this has to happen first), the physical layer
(GMSK at 9600 bps, modeled here as continuous-phase binary FSK -- correct
for exercising the framing/field logic, not a spectral model of real
GMSK), and the field layout of Common Navigation Block messages (types 1,
2, 3 -- position reports, the most common AIS traffic by far).

Field widths and scaling (MMSI 30 bits, lat/lon 1/600000 degree, speed
0.1 knot, course 0.1 degree, etc.) are cross-checked against `pyais`
(github.com/M0r13n/pyais, `pyais/messages.py`'s `MessageType1`), a real,
widely-used AIS decoder -- ITU-R M.1371's own bit-level layout is exactly
this, transcribing it from a working implementation just removes the
risk of a transcription error going unnoticed.

Self-rejecting: no HDLC flag found, a CRC failure, or a message type
other than 1/2/3, all return ``None``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._bee.core.crc import crc16_x25
from .._bee.decoders.hdlc import _bits_to_bytes, destuff_and_split_frames

BIT_RATE_HZ = 9600.0
_FLAG = [0, 1, 1, 1, 1, 1, 1, 0]


# --------------------------------------------------------------------------
# HDLC framing: bit-stuff + NRZI (the encode-side mirror of what
# sigint._bee.decoders.hdlc already implements for decode)
# --------------------------------------------------------------------------

def _stuff_bits(bits: list[int]) -> list[int]:
    out = []
    ones_run = 0
    for b in bits:
        out.append(b)
        if b == 1:
            ones_run += 1
            if ones_run == 5:
                out.append(0)
                ones_run = 0
        else:
            ones_run = 0
    return out


def _nrzi_encode(bits: list[int]) -> list[int]:
    """0 = transition, 1 = no transition -- the HDLC/AX.25/AIS convention."""
    level = 1
    out = []
    for b in bits:
        if b == 0:
            level ^= 1
        out.append(level)
    return out


def _nrzi_decode(levels: np.ndarray) -> np.ndarray:
    prev = np.concatenate([[1], levels[:-1]])
    return (levels == prev).astype(np.int64)   # no transition -> 1, transition -> 0


def _bytes_to_bits(data: bytes) -> list[int]:
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> i) & 1)
    return bits


def frame_message(payload_bits: np.ndarray) -> list[int]:
    """Payload bits (a multiple of 8) -> a complete flag-delimited,
    bit-stuffed, NRZI-encoded HDLC frame ready to FSK-modulate."""
    payload_bytes = bytes(_bits_to_bytes(list(payload_bits)))
    fcs = crc16_x25(payload_bytes)
    frame_bits = list(payload_bits) + _bytes_to_bits(bytes([fcs & 0xFF, (fcs >> 8) & 0xFF]))
    stuffed = _stuff_bits(frame_bits)
    return _FLAG + stuffed + _FLAG


# --------------------------------------------------------------------------
# Message type 1/2/3 (Common Navigation Block -- position reports)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PositionReport:
    msg_type: int          # 1, 2, or 3
    mmsi: int
    nav_status: int        # 0-15, see NAV_STATUS
    lon: float             # degrees
    lat: float             # degrees
    speed_knots: float
    course_deg: float
    heading_deg: int | None   # None if not available (511)

    def summary(self) -> str:
        status = NAV_STATUS.get(self.nav_status, f"status {self.nav_status}")
        return (f"MMSI {self.mmsi} @ {self.lat:.5f},{self.lon:.5f} "
               f"{self.speed_knots:.1f} kn, course {self.course_deg:.1f} deg, {status}")


NAV_STATUS: dict[int, str] = {
    0: "under way using engine", 1: "at anchor", 2: "not under command",
    3: "restricted maneuverability", 4: "constrained by draught", 5: "moored",
    6: "aground", 7: "engaged in fishing", 8: "under way sailing",
    11: "power-driven vessel towing astern", 12: "power-driven vessel pushing ahead",
    14: "AIS-SART/MOB/EPIRB", 15: "not defined",
}


def _bits_int(bits: np.ndarray, signed: bool = False) -> int:
    v = int("".join(str(int(b)) for b in bits), 2) if len(bits) else 0
    if signed and bits[0]:
        v -= 1 << len(bits)
    return v


def pack_position_report(msg_type: int, mmsi: int, nav_status: int, lon: float,
                         lat: float, speed_knots: float, course_deg: float,
                         heading_deg: int | None = None) -> np.ndarray:
    """Pack a type 1/2/3 Common Navigation Block -> 168 payload bits."""
    def bits_of(v: int, n: int) -> list[int]:
        return [(v >> (n - 1 - i)) & 1 for i in range(n)]

    lon_i = int(round(lon * 600000)) & ((1 << 28) - 1)
    lat_i = int(round(lat * 600000)) & ((1 << 27) - 1)
    speed_i = min(1022, int(round(speed_knots * 10)))
    course_i = int(round(course_deg * 10)) & ((1 << 12) - 1)
    heading_i = 511 if heading_deg is None else heading_deg

    bits = (bits_of(msg_type, 6) + bits_of(0, 2) + bits_of(mmsi, 30)
           + bits_of(nav_status, 4) + bits_of(0x80, 8)   # turn: "not available"
           + bits_of(speed_i, 10) + bits_of(0, 1)
           + bits_of(lon_i, 28) + bits_of(lat_i, 27)
           + bits_of(course_i, 12) + bits_of(heading_i, 9)
           + bits_of(60, 6) + bits_of(0, 2) + bits_of(0, 3)
           + bits_of(0, 1) + bits_of(0, 19))
    assert len(bits) == 168
    return np.array(bits, dtype=np.int64)


def unpack_position_report(bits: np.ndarray) -> PositionReport | None:
    if len(bits) < 168:
        return None
    msg_type = _bits_int(bits[0:6])
    if msg_type not in (1, 2, 3):
        return None
    mmsi = _bits_int(bits[8:38])
    nav_status = _bits_int(bits[38:42])
    speed = _bits_int(bits[50:60]) / 10.0
    lon = _bits_int(bits[61:89], signed=True) / 600000.0
    lat = _bits_int(bits[89:116], signed=True) / 600000.0
    course = _bits_int(bits[116:128]) / 10.0
    heading = _bits_int(bits[128:137])
    return PositionReport(msg_type=msg_type, mmsi=mmsi, nav_status=nav_status,
                          lon=lon, lat=lat, speed_knots=speed, course_deg=course,
                          heading_deg=None if heading == 511 else heading)


# --------------------------------------------------------------------------
# Synthesis and detection
# --------------------------------------------------------------------------

def generate(msg_type: int, sample_rate: float, mmsi: int, nav_status: int,
            lon: float, lat: float, speed_knots: float, course_deg: float,
            heading_deg: int | None = None, base_hz: float = 20000.0,
            deviation_hz: float = 5000.0, training_bits: int = 24) -> np.ndarray:
    """Synthesize a GMSK-modeled (continuous-phase binary FSK) recording
    of one AIS position report, preceded by the standard alternating-bit
    training sequence used for bit-clock recovery.

    `sample_rate` needs real headroom over the 9600 Bd symbol rate --
    both to resolve `base_hz +/- deviation_hz` at all (Nyquist) and,
    less obviously, to give each symbol enough *cycles* of its tone to
    correlate against (a tone under about one cycle per symbol doesn't
    demodulate reliably no matter how finely it's sampled -- this isn't a
    sample-rate problem once Nyquist is satisfied, it's a `tone_hz *
    symbol_s` problem). 96 kHz or higher is a safe default to test with.
    """
    payload = pack_position_report(msg_type, mmsi, nav_status, lon, lat,
                                   speed_knots, course_deg, heading_deg)
    frame_bits = frame_message(payload)
    nrzi = _nrzi_encode(frame_bits)
    training = [i % 2 for i in range(training_bits)]
    all_bits = training + nrzi

    sps = max(1, int(round(sample_rate / BIT_RATE_HZ)))
    phase = 0.0
    chunks = []
    t = np.arange(sps) / sample_rate
    for bit in all_bits:
        f = base_hz + (deviation_hz if bit else -deviation_hz)
        chunks.append(np.cos(2 * np.pi * f * t + phase))
        phase = (phase + 2 * np.pi * f * sps / sample_rate) % (2 * np.pi)
    return np.concatenate(chunks)


def decode(samples: np.ndarray, sample_rate: float, base_hz: float = 20000.0,
          deviation_hz: float = 5000.0, phase_steps: int = 8) -> PositionReport | None:
    """Bit-synchronize, FSK-demod, NRZI-decode, and destuff/CRC-check via
    the shared HDLC framing, then unpack a type 1/2/3 position report.
    Self-rejects if no flag-delimited, CRC-valid frame is found."""
    x = np.asarray(samples, dtype=np.float64).ravel()
    sps = max(1, int(round(sample_rate / BIT_RATE_HZ)))
    t = np.arange(sps) / sample_rate
    mark = np.exp(-2j * np.pi * (base_hz + deviation_hz) * t)
    space = np.exp(-2j * np.pi * (base_hz - deviation_hz) * t)

    best_bits, best_quality = None, -1.0
    for off in np.linspace(0, sps - 1, phase_steps).astype(int):
        n_sym = (len(x) - off) // sps
        if n_sym < 32:
            continue
        levels = np.empty(n_sym, dtype=np.int64)
        mags = np.empty(n_sym)
        for k in range(n_sym):
            seg = x[off + k * sps: off + (k + 1) * sps]
            m, s = abs(np.dot(seg, mark)), abs(np.dot(seg, space))
            levels[k] = 1 if m > s else 0
            mags[k] = max(m, s) / sps
        quality = float(mags.mean())
        if quality > best_quality:
            best_bits, best_quality = _nrzi_decode(levels), quality

    if best_bits is None:
        return None
    for frame_bits in destuff_and_split_frames(best_bits):
        data = _bits_to_bytes(frame_bits)
        if len(data) < 3:
            continue
        payload, fcs_bytes = data[:-2], data[-2:]
        if crc16_x25(payload) != (fcs_bytes[0] | (fcs_bytes[1] << 8)):
            continue
        # _bits_to_bytes (shared with sigint._bee's HDLC/AX.25 framing)
        # packs each byte LSB-first from the transmitted bit order, so
        # this is the exact inverse -- NOT the more common MSB-first
        # unpacking, which would silently scramble every field.
        report = unpack_position_report(np.unpackbits(np.frombuffer(payload, dtype=np.uint8),
                                                       bitorder="little"))
        if report is not None:
            return report
    return None
