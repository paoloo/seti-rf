"""SSTV (Slow-Scan Television): VIS header detect, image encode/decode.

Every SSTV mode starts with the same VIS (Vertical Interval Signaling)
header: two 300 ms leader tones at 1900 Hz around a 10 ms 1200 Hz break,
then a 30 ms 1200 Hz start bit, 7 data bits (1100 Hz = 1, 1300 Hz = 0,
LSB first) plus one even-parity bit, then a 30 ms 1200 Hz stop bit. The
7-bit VIS code identifies the mode that follows -- so VIS header
detection alone (no image decode needed) already answers "is this SSTV,
and which mode" the same way :mod:`sigint.hfdl`'s preamble correlation
answers "is this HFDL." After the header, every mode FM-modulates
brightness onto a 1500 (black) - 2300 Hz (white) tone per pixel, with a
mode-specific line/sync/channel structure.

Three modes are fully supported end to end (encode + decode): Martin M1,
Scottie S1 and Robot 36 -- chosen to cover the three structurally
different line layouts in real use (simple RGB-per-channel, RGB with the
sync pulse moved before the red channel, and 4:2:0-style YUV with
alternating chroma). VIS codes and all per-mode timing constants (sync/
scan/porch durations in ms, channel order) are transcribed from
``pySSTV``'s real encoder (github.com/dnet/pySSTV, ``pysstv/color.py``),
a working, widely-used implementation, not reconstructed from memory --
a wrong scan-line duration desyncs an entire image.

Images are plain ``np.ndarray`` (H, W, 3) ``uint8`` RGB arrays -- no PIL
dependency, consistent with the rest of this package.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dsp import _analytic

FREQ_VIS_BIT1 = 1100.0
FREQ_SYNC = 1200.0
FREQ_VIS_BIT0 = 1300.0
FREQ_BLACK = 1500.0
FREQ_VIS_START = 1900.0
FREQ_WHITE = 2300.0
FREQ_RANGE = FREQ_WHITE - FREQ_BLACK

MSEC_VIS_START = 300.0
MSEC_VIS_SYNC = 10.0
MSEC_VIS_BIT = 30.0


@dataclass(frozen=True)
class SSTVMode:
    name: str
    vis_code: int
    width: int
    height: int


MARTIN_M1 = SSTVMode("Martin M1", 0x2C, 320, 256)
SCOTTIE_S1 = SSTVMode("Scottie S1", 0x3C, 320, 256)
ROBOT36 = SSTVMode("Robot 36", 0x08, 320, 240)

_MARTIN_SYNC_MS = 4.862
_MARTIN_SCAN_MS = 146.432
_MARTIN_GAP_MS = 0.572

_SCOTTIE_SYNC_MS = 9.0
_SCOTTIE_GAP_MS = 1.5
_SCOTTIE_SCAN_MS = 138.24 - _SCOTTIE_GAP_MS

_ROBOT_SYNC_MS = 9.0
_ROBOT_PORCH_MS = 1.5
_ROBOT_SYNC_PORCH_MS = 3.0
_ROBOT_GAP_MS = 4.5
_ROBOT_Y_SCAN_MS = 88.0
_ROBOT_C_SCAN_MS = 44.0

_MODES_BY_VIS = {m.vis_code: m for m in (MARTIN_M1, SCOTTIE_S1, ROBOT36)}


def _byte_to_freq(value: float) -> float:
    return FREQ_BLACK + FREQ_RANGE * value / 255.0


def _freq_to_byte(freq: float) -> int:
    v = (freq - FREQ_BLACK) / FREQ_RANGE * 255.0
    return int(np.clip(round(v), 0, 255))


def _rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 (the JPEG/PIL 'YCbCr' convention pySSTV relies on)."""
    r, g, b = rgb[..., 0].astype(np.float64), rgb[..., 1].astype(np.float64), rgb[..., 2].astype(np.float64)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.stack([y, cb, cr], axis=-1)


def _ycbcr_to_rgb(ycc: np.ndarray) -> np.ndarray:
    y, cb, cr = ycc[..., 0], ycc[..., 1] - 128, ycc[..., 2] - 128
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Tone synthesis / instantaneous-frequency demod (shared primitives)
# --------------------------------------------------------------------------

class _ToneWriter:
    """Accumulates (freq_hz, duration_ms) segments into one continuous-
    phase waveform -- the encoder-side mirror of `_inst_freq` below."""

    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate
        self.phase = 0.0
        self.chunks: list[np.ndarray] = []

    def add(self, freq_hz: float, duration_ms: float) -> None:
        n = max(1, int(round(duration_ms / 1000.0 * self.sample_rate)))
        t = np.arange(n) / self.sample_rate
        self.chunks.append(np.cos(2 * np.pi * freq_hz * t + self.phase))
        self.phase = (self.phase + 2 * np.pi * freq_hz * n / self.sample_rate) % (2 * np.pi)

    def add_line(self, freqs_hz: np.ndarray, duration_ms: float) -> None:
        """`freqs_hz` sampled once per output sample over `duration_ms`
        total (a pixel row's worth of continuously-varying frequency)."""
        n = max(1, int(round(duration_ms / 1000.0 * self.sample_rate)))
        freqs = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(freqs_hz)), freqs_hz)
        phase_inc = 2 * np.pi * freqs / self.sample_rate
        phase = self.phase + np.cumsum(phase_inc)
        self.chunks.append(np.cos(phase))
        self.phase = float(phase[-1]) % (2 * np.pi)

    def result(self) -> np.ndarray:
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0)


def _inst_freq(x: np.ndarray, sample_rate: float) -> np.ndarray:
    """Instantaneous frequency (Hz) via the analytic signal's phase
    derivative -- one estimate per input sample (length len(x)-1, the
    same convention :func:`sigint.dsp.baud_rate_stages` uses)."""
    z = _analytic(x)
    phase = np.unwrap(np.angle(z))
    return np.diff(phase) * sample_rate / (2 * np.pi)


# --------------------------------------------------------------------------
# VIS header
# --------------------------------------------------------------------------

def _write_vis(tw: _ToneWriter, mode: SSTVMode) -> None:
    tw.add(FREQ_VIS_START, MSEC_VIS_START)
    tw.add(FREQ_SYNC, MSEC_VIS_SYNC)
    tw.add(FREQ_VIS_START, MSEC_VIS_START)
    tw.add(FREQ_SYNC, MSEC_VIS_BIT)   # start bit
    ones = 0
    for i in range(7):
        bit = (mode.vis_code >> i) & 1
        ones += bit
        tw.add(FREQ_VIS_BIT1 if bit else FREQ_VIS_BIT0, MSEC_VIS_BIT)
    parity_bit = 1 if ones % 2 == 1 else 0
    tw.add(FREQ_VIS_BIT1 if parity_bit else FREQ_VIS_BIT0, MSEC_VIS_BIT)
    tw.add(FREQ_SYNC, MSEC_VIS_BIT)   # stop bit


def _find_leader(freq: np.ndarray, sample_rate: float,
                 min_ms: float = 200.0, tol_hz: float = 60.0) -> int | None:
    """First sample index where the instantaneous frequency holds near
    1900 Hz for at least `min_ms` -- the VIS leader tone. Returns None if
    no such run exists (self-rejects: not SSTV, or too corrupted to find
    the header at all)."""
    on = np.abs(freq - FREQ_VIS_START) < tol_hz
    need = max(1, int(round(min_ms / 1000.0 * sample_rate)))
    run = 0
    for i, v in enumerate(on):
        run = run + 1 if v else 0
        if run >= need:
            return i - run + 1
    return None


def _read_vis(freq: np.ndarray, sample_rate: float, start: int) -> tuple[SSTVMode, int] | None:
    """Given the leader's start index, read the VIS code and return
    (mode, sample index the image data starts at), or None if the header
    doesn't check out (bad parity, or an unrecognized VIS code)."""
    def ms_to_n(ms: float) -> int:
        return max(1, int(round(ms / 1000.0 * sample_rate)))

    pos = start
    pos += ms_to_n(MSEC_VIS_START) + ms_to_n(MSEC_VIS_SYNC) + ms_to_n(MSEC_VIS_START)
    pos += ms_to_n(MSEC_VIS_BIT)   # start bit
    bit_n = ms_to_n(MSEC_VIS_BIT)
    bits = []
    for _ in range(8):   # 7 data bits + parity
        if pos + bit_n > len(freq):
            return None
        avg = float(freq[pos:pos + bit_n].mean())
        bits.append(1 if abs(avg - FREQ_VIS_BIT1) < abs(avg - FREQ_VIS_BIT0) else 0)
        pos += bit_n
    pos += bit_n   # stop bit
    vis_code = sum(b << i for i, b in enumerate(bits[:7]))
    if sum(bits[:7]) % 2 != bits[7]:
        return None
    mode = _MODES_BY_VIS.get(vis_code)
    if mode is None:
        return None
    return mode, pos


# --------------------------------------------------------------------------
# Per-mode image encode/decode
# --------------------------------------------------------------------------

def _encode_martin(tw: _ToneWriter, rgb: np.ndarray) -> None:
    for line in range(MARTIN_M1.height):
        tw.add(FREQ_SYNC, _MARTIN_SYNC_MS)
        for ch in (1, 2, 0):   # green, blue, red
            tw.add(FREQ_BLACK, _MARTIN_GAP_MS)
            tw.add_line(_byte_to_freq(rgb[line, :, ch].astype(np.float64)), _MARTIN_SCAN_MS)


def _encode_scottie(tw: _ToneWriter, rgb: np.ndarray) -> None:
    for line in range(SCOTTIE_S1.height):
        tw.add(FREQ_BLACK, _SCOTTIE_GAP_MS)
        tw.add_line(_byte_to_freq(rgb[line, :, 1].astype(np.float64)), _SCOTTIE_SCAN_MS)  # green
        tw.add(FREQ_BLACK, _SCOTTIE_GAP_MS)
        tw.add_line(_byte_to_freq(rgb[line, :, 2].astype(np.float64)), _SCOTTIE_SCAN_MS)  # blue
        tw.add(FREQ_SYNC, _SCOTTIE_SYNC_MS)
        tw.add(FREQ_BLACK, _SCOTTIE_GAP_MS)
        tw.add_line(_byte_to_freq(rgb[line, :, 0].astype(np.float64)), _SCOTTIE_SCAN_MS)  # red


def _encode_robot36(tw: _ToneWriter, rgb: np.ndarray) -> None:
    ycc = _rgb_to_ycbcr(rgb)
    for line in range(ROBOT36.height):
        channel = 2 - (line % 2)   # alternates Cr(2)/Cb(1) chroma line
        tw.add(FREQ_SYNC, _ROBOT_SYNC_MS)
        tw.add(FREQ_BLACK, _ROBOT_SYNC_PORCH_MS)
        tw.add_line(_byte_to_freq(np.clip(ycc[line, :, 0], 0, 255)), _ROBOT_Y_SCAN_MS)
        tw.add(FREQ_WHITE if channel == 1 else FREQ_BLACK, _ROBOT_GAP_MS)
        tw.add(FREQ_VIS_START, _ROBOT_PORCH_MS)
        tw.add_line(_byte_to_freq(np.clip(ycc[line, :, channel], 0, 255)), _ROBOT_C_SCAN_MS)


def generate(rgb: np.ndarray, sample_rate: float, mode: SSTVMode) -> np.ndarray:
    """Synthesize an SSTV recording of an (H, W, 3) uint8 RGB image."""
    if rgb.shape[:2] != (mode.height, mode.width):
        raise ValueError(f"{mode.name} needs a ({mode.height}, {mode.width}, 3) "
                        f"image, got {rgb.shape}")
    tw = _ToneWriter(sample_rate)
    _write_vis(tw, mode)
    {MARTIN_M1: _encode_martin, SCOTTIE_S1: _encode_scottie,
     ROBOT36: _encode_robot36}[mode](tw, rgb)
    return tw.result()


def _read_line(freq: np.ndarray, sample_rate: float, pos: int,
              width: int, duration_ms: float) -> tuple[np.ndarray, int]:
    n = max(1, int(round(duration_ms / 1000.0 * sample_rate)))
    seg = freq[pos:pos + n]
    if len(seg) == 0:
        return np.zeros(width, dtype=np.uint8), pos + n
    idx = np.linspace(0, len(seg) - 1, width).astype(int)
    pixels = np.array([_freq_to_byte(f) for f in seg[idx]], dtype=np.uint8)
    return pixels, pos + n


def _decode_martin(freq: np.ndarray, sample_rate: float, pos: int) -> np.ndarray:
    rgb = np.zeros((MARTIN_M1.height, MARTIN_M1.width, 3), dtype=np.uint8)
    ms = lambda v: max(1, int(round(v / 1000.0 * sample_rate)))
    for line in range(MARTIN_M1.height):
        pos += ms(_MARTIN_SYNC_MS)
        for ch in (1, 2, 0):
            pos += ms(_MARTIN_GAP_MS)
            rgb[line, :, ch], pos = _read_line(freq, sample_rate, pos,
                                              MARTIN_M1.width, _MARTIN_SCAN_MS)
    return rgb


def _decode_scottie(freq: np.ndarray, sample_rate: float, pos: int) -> np.ndarray:
    rgb = np.zeros((SCOTTIE_S1.height, SCOTTIE_S1.width, 3), dtype=np.uint8)
    ms = lambda v: max(1, int(round(v / 1000.0 * sample_rate)))
    for line in range(SCOTTIE_S1.height):
        pos += ms(_SCOTTIE_GAP_MS)
        rgb[line, :, 1], pos = _read_line(freq, sample_rate, pos, SCOTTIE_S1.width, _SCOTTIE_SCAN_MS)
        pos += ms(_SCOTTIE_GAP_MS)
        rgb[line, :, 2], pos = _read_line(freq, sample_rate, pos, SCOTTIE_S1.width, _SCOTTIE_SCAN_MS)
        pos += ms(_SCOTTIE_SYNC_MS) + ms(_SCOTTIE_GAP_MS)
        rgb[line, :, 0], pos = _read_line(freq, sample_rate, pos, SCOTTIE_S1.width, _SCOTTIE_SCAN_MS)
    return rgb


def _decode_robot36(freq: np.ndarray, sample_rate: float, pos: int) -> np.ndarray:
    ycc = np.zeros((ROBOT36.height, ROBOT36.width, 3), dtype=np.float64)
    ycc[..., 1] = 128
    ycc[..., 2] = 128
    ms = lambda v: max(1, int(round(v / 1000.0 * sample_rate)))
    for line in range(ROBOT36.height):
        channel = 2 - (line % 2)
        pos += ms(_ROBOT_SYNC_MS) + ms(_ROBOT_SYNC_PORCH_MS)
        y_pixels, pos = _read_line(freq, sample_rate, pos, ROBOT36.width, _ROBOT_Y_SCAN_MS)
        ycc[line, :, 0] = y_pixels
        pos += ms(_ROBOT_GAP_MS) + ms(_ROBOT_PORCH_MS)
        c_pixels, pos = _read_line(freq, sample_rate, pos, ROBOT36.width, _ROBOT_C_SCAN_MS)
        ycc[line, :, channel] = c_pixels
        if line + 1 < ROBOT36.height:
            ycc[line + 1, :, channel] = c_pixels   # chroma is shared across the pair of lines
    return _ycbcr_to_rgb(ycc)


@dataclass(frozen=True)
class SSTVImage:
    mode: str
    image: np.ndarray   # (H, W, 3) uint8


def decode(samples: np.ndarray, sample_rate: float) -> SSTVImage | None:
    """Find the VIS header anywhere in `samples`, identify the mode, and
    decode the image that follows. Self-rejecting: no leader tone, a
    parity failure, or an unrecognized VIS code all return ``None``."""
    x = np.asarray(samples, dtype=np.float64).ravel()
    freq = _inst_freq(x, sample_rate)
    start = _find_leader(freq, sample_rate)
    if start is None:
        return None
    found = _read_vis(freq, sample_rate, start)
    if found is None:
        return None
    mode, pos = found
    decoder = {MARTIN_M1: _decode_martin, SCOTTIE_S1: _decode_scottie,
              ROBOT36: _decode_robot36}[mode]
    image = decoder(freq, sample_rate, pos)
    return SSTVImage(mode=mode.name, image=image)
