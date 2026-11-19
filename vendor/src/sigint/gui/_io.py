"""Load a recording (wav/npy/npz/raw) into the workbench."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def load_recording(path: str | Path, *, preserve_iq: bool = False) -> tuple[np.ndarray, float]:
    """Read wav / npy / npz / raw -> (float64 samples, rate).

    .raw/.iq are headerless int16 (the legacy convention); .f32/.cf32
    are headerless float32 (real / interleaved complex, I first) —
    for complex files the I component is returned as the real signal.
    Set preserve_iq to retain complex .cf32/.npy/.npz samples.
    Sample rate defaults to 16000 Hz for headerless formats; pass an
    explicit rate via the CLI where supported.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".wav":
        import wave
        with wave.open(str(p), "rb") as w:
            sr = float(w.getframerate())
            nch = w.getnchannels()
            width = w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if width == 2:
            x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
        elif width == 4:
            x = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
        elif width == 1:
            x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
        else:
            raise ValueError(f"unsupported wav sample width {width}")
        if nch > 1:
            x = x.reshape(-1, nch).mean(axis=1)
        return x, sr
    if suffix == ".npy":
        x = np.load(p)
        return (x if preserve_iq else x.real.astype(np.float64)).ravel(), 16000.0
    if suffix == ".npz":
        z = np.load(p)
        key = "samples" if "samples" in z.files else z.files[0]
        sr = float(z["sample_rate"]) if "sample_rate" in z.files else 16000.0
        x = z[key]
        return (x if preserve_iq else x.real.astype(np.float64)).ravel(), sr
    if suffix == ".raw" or suffix == ".iq":
        raw = p.read_bytes()
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
        return x, 16000.0
    if suffix == ".f32":
        x = np.frombuffer(p.read_bytes(), dtype="<f4").astype(np.float64)
        return x, 16000.0
    if suffix == ".cf32":
        z = np.frombuffer(p.read_bytes(), dtype="<c8")
        return (z if preserve_iq else z.real.astype(np.float64)), 16000.0
    raise ValueError(f"unsupported format {suffix!r} (wav/npy/npz/raw)")


def save_recording(path: str | Path, samples: np.ndarray, sample_rate: float) -> None:
    p = Path(path)
    peak = np.abs(samples).max() + 1e-12
    pcm = np.clip(samples / peak * 32767, -32768, 32767).astype("<i2")
    import wave
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())
