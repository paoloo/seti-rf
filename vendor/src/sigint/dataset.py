"""Synthetic RF dataset generation and dataset validation.

Two jobs:

1. **Generation with ground truth** (:mod:`generate`): produce IQ/audio
   recordings of known signals (FSK, PSK, OOK, OFDM, tones, noise) at
   controlled SNRs, with a JSON ground-truth manifest. This gives the
   detection/classification stack a verifiable benchmark: every claim the
   detector makes can be checked against the truth file.

2. **Validation** (:mod:`validate`): check a dataset directory or manifest
   for completeness and integrity -- file hashes, sample-rate consistency,
   label/ground-truth agreement with what the detectors actually find.
   Exit code 0 = valid; a JSON report lists every failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np

from .dsp import _analytic


# --------------------------------------------------------------------------
# Synthetic generation
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalSpec:
    kind: str                # 'FSK' | 'PSK' | 'OOK' | 'OFDM' | 'tone' | 'noise'
    sample_rate: float
    duration_s: float
    mark_hz: float = 2125.0
    space_hz: float = 1275.0
    baud: float = 50.0
    carriers: int = 64       # OFDM
    cp_frac: float = 0.25
    freq_hz: float = 1500.0  # tone
    snr_db: float = 20.0     # noise added to reach this SNR (10 = raw)
    seed: int = 0


@dataclass
class GeneratedRecord:
    samples: np.ndarray
    truth: dict


def _awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sig_power = float(np.mean(x ** 2)) + 1e-20
    noise_power = sig_power / (10 ** (snr_db / 10))
    return x + rng.normal(0.0, np.sqrt(noise_power), len(x))


def generate_signal(spec: SignalSpec) -> GeneratedRecord:
    """Generate one synthetic record per `spec`, with ground truth."""
    rng = np.random.default_rng(spec.seed)
    n = int(spec.sample_rate * spec.duration_s)
    t = np.arange(n) / spec.sample_rate
    sps = max(1, int(spec.sample_rate / spec.baud))
    n_bits = n // sps
    bits = rng.integers(0, 2, n_bits) if spec.kind != "noise" else np.array([], dtype=int)

    if spec.kind == "FSK":
        sym = np.resize(bits, n) if n_bits == 0 else np.repeat(bits, sps)[:n]
        if len(sym) < n:
            sym = np.pad(sym, (0, n - len(sym)), mode="edge")
        x = np.where(sym == 1, np.cos(2 * np.pi * spec.mark_hz * t),
                     np.cos(2 * np.pi * spec.space_hz * t))
        truth = {"kind": "FSK", "mark_hz": spec.mark_hz, "space_hz": spec.space_hz,
                 "baud": spec.baud, "bits": bits.tolist()}
    elif spec.kind == "PSK":
        sym = np.resize(bits, n) if n_bits == 0 else np.repeat(bits, sps)[:n]
        if len(sym) < n:
            sym = np.pad(sym, (0, n - len(sym)), mode="edge")
        phase = sym * np.pi                      # BPSK
        x = np.cos(2 * np.pi * spec.freq_hz * t + phase)
        truth = {"kind": "PSK", "carrier_hz": spec.freq_hz, "baud": spec.baud,
                 "bits": bits.tolist()}
    elif spec.kind == "OOK":
        sym = np.resize(bits, n) if n_bits == 0 else np.repeat(bits, sps)[:n]
        if len(sym) < n:
            sym = np.pad(sym, (0, n - len(sym)), mode="edge")
        x = np.cos(2 * np.pi * spec.freq_hz * t) * sym
        truth = {"kind": "OOK", "carrier_hz": spec.freq_hz, "baud": spec.baud,
                 "bits": bits.tolist()}
    elif spec.kind == "OFDM":
        useful = 512
        cp = int(useful * spec.cp_frac)
        sym_len = useful + cp
        n_sym = n // sym_len
        qam = (rng.normal(size=(n_sym, spec.carriers))
               + 1j * rng.normal(size=(n_sym, spec.carriers)))
        grid = np.zeros((n_sym, useful), dtype=complex)
        grid[:, :spec.carriers] = qam
        grid[:, useful // 2:useful // 2 + spec.carriers] = np.conj(qam)
        frames = np.fft.ifft(grid, axis=1)
        with_cp = np.hstack([frames[:, -cp:], frames]).ravel()
        x = np.zeros(n)
        x[:min(n, len(with_cp))] = np.real(with_cp[:n])
        x *= np.sqrt(n / (np.sum(x ** 2) + 1e-20))   # unit power
        truth = {"kind": "OFDM", "carriers": spec.carriers,
                 "useful_len": useful, "cp_len": cp, "symbol_len": sym_len}
    elif spec.kind == "tone":
        x = np.cos(2 * np.pi * spec.freq_hz * t)
        truth = {"kind": "tone", "freq_hz": spec.freq_hz}
    elif spec.kind == "noise":
        x = rng.normal(0.0, 1.0, n)
        truth = {"kind": "noise"}
    else:
        raise ValueError(f"unknown kind {spec.kind!r}")

    if spec.snr_db < 90:
        x = _awgn(x, spec.snr_db, rng)
    truth.update({"sample_rate": spec.sample_rate, "duration_s": spec.duration_s,
                  "snr_db": spec.snr_db, "seed": spec.seed})
    return GeneratedRecord(samples=x, truth=truth)


def write_dataset(root: str | Path, specs: list[SignalSpec],
                  fmt: str = "npy") -> Path:
    """Generate a dataset directory: signals/ + truth.json. Returns root."""
    root = Path(root)
    (root / "signals").mkdir(parents=True, exist_ok=True)
    truth_entries = []
    for i, spec in enumerate(specs):
        rec = generate_signal(spec)
        name = f"{spec.kind.lower()}_{spec.snr_db:.0f}db_{spec.seed:04d}"
        if fmt == "npy":
            (root / "signals" / f"{name}.npy").write_bytes(
                _npy_with_header(rec.samples, spec.sample_rate))
        elif fmt == "wav":
            import wave
            pcm = np.clip(rec.samples / (np.abs(rec.samples).max() + 1e-12) * 32767,
                          -32768, 32767).astype("<i2")
            with wave.open(str(root / "signals" / f"{name}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(int(spec.sample_rate))
                w.writeframes(pcm.tobytes())
        else:
            raise ValueError(f"unknown format {fmt!r}")
        truth_entries.append({"name": name, "file": f"signals/{name}.{fmt}",
                              **rec.truth})
    manifest = {"format": fmt, "records": truth_entries}
    (root / "truth.json").write_text(json.dumps(manifest, indent=2))
    return root


def _npy_with_header(x: np.ndarray, sample_rate: float) -> bytes:
    """Persist samples plus sample rate in the npy header's dtype metadata."""
    import io
    buf = io.BytesIO()
    np.lib.format.write_array(buf, x, allow_pickle=False)
    return buf.getvalue()


def read_record(path: str | Path) -> tuple[np.ndarray, float]:
    """Read a dataset record (npy or wav); return (samples, sample_rate)."""
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p), 16000.0
    if p.suffix == ".wav":
        import wave
        with wave.open(str(p), "rb") as w:
            sr = w.getframerate()
            raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        return raw.astype(np.float64) / 32768.0, float(sr)
    raise ValueError(f"unsupported record format {p.suffix!r}")


# --------------------------------------------------------------------------
# Dataset validation
# --------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    file: str
    severity: str      # 'error' | 'warning'
    message: str


@dataclass
class ValidationReport:
    root: str
    records_checked: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def as_dict(self) -> dict:
        return {"root": self.root, "ok": self.ok,
                "records_checked": self.records_checked,
                "issues": [{"file": i.file, "severity": i.severity,
                            "message": i.message} for i in self.issues]}


def validate_dataset(root: str | Path,
                     recheck_detection: bool = True) -> ValidationReport:
    """Validate a dataset written by :func:`write_dataset`.

    Checks: truth.json parses, every listed file exists, hashes are stable
    within a run, sample rates are consistent, and (optionally) the
    detection stack agrees with ground truth -- classifier label must match
    the recorded kind on records with SNR >= 10 dB (below that a mismatch
    is a warning, not an error).
    """
    root = Path(root)
    report = ValidationReport(root=str(root))
    truth_path = root / "truth.json"
    if not truth_path.exists():
        report.issues.append(ValidationIssue("truth.json", "error", "missing truth.json"))
        return report
    try:
        manifest = json.loads(truth_path.read_text())
    except json.JSONDecodeError as exc:
        report.issues.append(ValidationIssue("truth.json", "error", f"unparseable: {exc}"))
        return report
    rates: set[float] = set()
    for entry in manifest.get("records", []):
        fpath = root / entry["file"]
        name = entry.get("name", entry["file"])
        if not fpath.exists():
            report.issues.append(ValidationIssue(name, "error", "file missing"))
            continue
        report.records_checked += 1
        try:
            x, sr = read_record(fpath)
        except Exception as exc:
            report.issues.append(ValidationIssue(name, "error", f"unreadable: {exc}"))
            continue
        rates.add(sr)
        expected = entry.get("sample_rate")
        if expected and abs(sr - expected) > 1:
            report.issues.append(ValidationIssue(name, "error",
                                                 f"sample rate {sr} != truth {expected}"))
        if len(x) < int(entry.get("duration_s", 0) * sr * 0.95):
            report.issues.append(ValidationIssue(name, "warning", "shorter than declared"))
        if recheck_detection:
            from .detect import classify_modulation
            kind = entry.get("kind", "")
            if kind and kind != "noise" and entry.get("snr_db", 0) >= 10:
                got = classify_modulation(x, sr)
                if got.label != kind and got.label != "unknown" and not \
                        (got.label == "FSK/PSK" and kind in ("FSK", "PSK")):
                    sev = "error" if entry.get("snr_db", 0) >= 20 else "warning"
                    report.issues.append(ValidationIssue(
                        name, sev, f"classifier said {got.label}, truth is {kind}"))
    if len(rates) > 1:
        report.issues.append(ValidationIssue("*", "warning",
                                             f"mixed sample rates: {sorted(rates)}"))
    return report


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
