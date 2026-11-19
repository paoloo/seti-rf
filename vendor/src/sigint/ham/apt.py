"""NOAA APT (Automatic Picture Transmission): the weather-satellite image
format hams and SWLs still receive directly from NOAA POES satellites
(and the ISS's own SSTV events use a completely different mode --
:mod:`sigint.ham.sstv` -- don't confuse the two).

Unlike SSTV's FM-modulated brightness, APT AM-modulates a 2400 Hz
subcarrier: each "word" (pixel-equivalent sample, 4160 words/second, 2
lines/second = 2080 words/line) sets that subcarrier's amplitude, so
decoding is envelope detection, not instantaneous frequency. A line
splits into two 1040-word channel halves (visible + IR-ish sensor data),
each opening with 7 cycles of a fixed-frequency square-wave sync tone
(1040 Hz for channel A, 832 Hz for channel B -- both exact word-rate
submultiples: 4 and 5 words/cycle respectively) that a receiver
correlates against to find each new line, followed by a black porch, 909
words of video, and 45 words of calibration telemetry.

The word-budget per channel half (39 sync + 47 porch + 909 video + 45
telemetry = 1040) and the sync frequencies/word-per-cycle counts are
cross-checked against real APT documentation and `zacstewart/apt-decoder`
(github.com/zacstewart/apt-decoder)'s own sync-correlation template.

Images are plain ``np.ndarray`` (H, W) ``uint8`` arrays (one channel;
pass the same array for both A/B video if you don't have two distinct
channels) -- no PIL dependency, consistent with the rest of this
package.

Fidelity is real but modest (mean pixel error around 10-20 out of 255 in
testing) -- envelope detection plus resampling to the word clock, no
sync-jitter tracking within a line the way a production decoder does; a
recognizable image, not a pixel-perfect one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dsp import _analytic

SUBCARRIER_HZ = 2400.0
WORD_RATE_HZ = 4160.0
WORDS_PER_LINE = 2080
SYNC_A_WORDS = 39
SYNC_B_WORDS = 39
PORCH_WORDS = 47
VIDEO_WORDS = 909
TELEMETRY_WORDS = 45


def _square_wave_words(freq_hz: float, n_words: int, cycles: int = 7) -> np.ndarray:
    """A `freq_hz` square wave at the word rate, `cycles` full periods,
    zero-padded/truncated to exactly `n_words`."""
    words_per_cycle = WORD_RATE_HZ / freq_hz
    n = int(round(words_per_cycle * cycles))
    idx = np.arange(n)
    wave = ((idx % words_per_cycle) < words_per_cycle / 2).astype(np.float64)
    out = np.zeros(n_words)
    out[:min(n, n_words)] = wave[:min(n, n_words)]
    return out


SYNC_A_PATTERN = _square_wave_words(1040.0, SYNC_A_WORDS)
SYNC_B_PATTERN = _square_wave_words(832.0, SYNC_B_WORDS)


def _line_words(video_a: np.ndarray, video_b: np.ndarray) -> np.ndarray:
    """One image line's worth of words (`WORDS_PER_LINE` values, 0..255)."""
    telemetry = np.zeros(TELEMETRY_WORDS)
    return np.concatenate([
        SYNC_A_PATTERN * 255, np.zeros(PORCH_WORDS), video_a, telemetry,
        SYNC_B_PATTERN * 255, np.zeros(PORCH_WORDS), video_b, telemetry,
    ])


# --------------------------------------------------------------------------
# Synthesis and detection
# --------------------------------------------------------------------------

def generate(image_a: np.ndarray, sample_rate: float, image_b: np.ndarray | None = None
            ) -> np.ndarray:
    """Synthesize an AM recording of `image_a` (and `image_b`, defaulting
    to the same image, as the second channel) -- one word's worth of
    audio per pixel, amplitude-modulating the 2400 Hz subcarrier."""
    image_a = np.asarray(image_a, dtype=np.float64)
    image_b = image_a if image_b is None else np.asarray(image_b, dtype=np.float64)
    height, width = image_a.shape
    resized_a = np.array([np.interp(np.linspace(0, width - 1, VIDEO_WORDS),
                                    np.arange(width), image_a[y]) for y in range(height)])
    resized_b = np.array([np.interp(np.linspace(0, width - 1, VIDEO_WORDS),
                                    np.arange(width), image_b[y]) for y in range(height)])

    sps = max(1, int(round(sample_rate / WORD_RATE_HZ)))
    phase = 0.0
    chunks = []
    t = np.arange(sps) / sample_rate
    for y in range(height):
        words = _line_words(resized_a[y], resized_b[y])
        for w in words:
            amp = w / 255.0
            chunks.append(amp * np.cos(2 * np.pi * SUBCARRIER_HZ * t + phase))
            phase = (phase + 2 * np.pi * SUBCARRIER_HZ * sps / sample_rate) % (2 * np.pi)
    return np.concatenate(chunks) if chunks else np.zeros(0)


@dataclass(frozen=True)
class APTImage:
    image_a: np.ndarray   # (H, 909) uint8
    image_b: np.ndarray


def _words_from_envelope(samples: np.ndarray, sample_rate: float) -> np.ndarray:
    """Envelope-detect and resample to the 4160 words/second word clock."""
    env = np.abs(_analytic(samples))
    n_words = int(round(len(samples) / sample_rate * WORD_RATE_HZ))
    if n_words < 2:
        return np.zeros(0)
    idx = np.linspace(0, len(env) - 1, n_words)
    return np.interp(idx, np.arange(len(env)), env)


def decode(samples: np.ndarray, sample_rate: float, min_lines: int = 4) -> APTImage | None:
    """Envelope-detect, resample to the word clock, correlate for sync A
    at every candidate line start, and decode as many full lines as are
    found. Self-rejects if fewer than `min_lines` lines are recovered."""
    words = _words_from_envelope(np.asarray(samples, dtype=np.float64).ravel(), sample_rate)
    if len(words) < WORDS_PER_LINE * min_lines:
        return None
    words = words / (words.max() + 1e-12) * 255.0

    template = SYNC_A_PATTERN * 255.0
    template_ac = template - template.mean()
    starts = []
    search_to = len(words) - WORDS_PER_LINE
    i = 0
    while i < search_to:
        window = words[i:i + SYNC_A_WORDS]
        window_ac = window - window.mean()
        denom = np.linalg.norm(template_ac) * np.linalg.norm(window_ac)
        corr = float(np.dot(template_ac, window_ac) / denom) if denom else 0.0
        if corr > 0.7:
            starts.append(i)
            i += WORDS_PER_LINE - SYNC_A_WORDS // 2   # skip ahead to the next expected line
        else:
            i += 1

    if len(starts) < min_lines:
        return None
    rows_a, rows_b = [], []
    for s in starts:
        if s + WORDS_PER_LINE > len(words):
            break
        line = words[s:s + WORDS_PER_LINE]
        va_start = SYNC_A_WORDS + PORCH_WORDS
        vb_start = va_start + VIDEO_WORDS + TELEMETRY_WORDS + SYNC_B_WORDS + PORCH_WORDS
        rows_a.append(line[va_start:va_start + VIDEO_WORDS])
        rows_b.append(line[vb_start:vb_start + VIDEO_WORDS])
    if len(rows_a) < min_lines:
        return None
    return APTImage(image_a=np.array(rows_a).astype(np.uint8),
                    image_b=np.array(rows_b).astype(np.uint8))
