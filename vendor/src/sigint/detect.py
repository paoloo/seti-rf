"""Signal detection, estimation and classification.

The automatic layer a human analyst needs before reaching for a decoder:

- :func:`snr_estimate` -- in-band vs out-of-band noise power ratio.
- :func:`find_carriers` -- peak-picking on the PSD with prominence and
  persistence (across FFT frames) tests, so a real carrier beats a noise
  spike.
- :func:`detect_activity` -- time-domain energy activity segments
  (VAD-style: noise floor from the lowest decile, activity above
  threshold + hysteresis).
- :func:`classify_modulation` -- lightweight decision-tree classifier on
  spectral/ envelope features: tone-vs-noise, FSK (two stable tones with
  alternating energy), OOK/ASK (on-off envelope), PSK-like (constant
  envelope, spread spectrum), OFDM (flat spectrum with CP correlation).
- :func:`band_scan` -- run detection across a band and report every
  occupied channel.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dsp import Spectrogram, _analytic, psd, spectrogram


# --------------------------------------------------------------------------
# SNR estimation
# --------------------------------------------------------------------------

def snr_estimate(samples: np.ndarray, sample_rate: float,
                 band_lo_hz: float, band_hi_hz: float,
                 guard_hz: float = 100.0) -> float:
    """Narrowband SNR (dB): peak PSD inside the band vs the median noise
    PSD in guard bands beside it.

    The peak statistic is robust to how the signal spreads its energy
    across bins (a pure tone concentrates in one bin; FSK in two; noise
    spreads), while the guard-band median is a stable noise-floor
    estimate. Returns dB.
    """
    freqs, db = psd(samples, sample_rate)
    band = (freqs >= band_lo_hz) & (freqs <= band_hi_hz)
    lo_guard = (freqs >= band_lo_hz - guard_hz) & (freqs < band_lo_hz)
    hi_guard = (freqs > band_hi_hz) & (freqs <= band_hi_hz + guard_hz)
    if not band.any() or not (lo_guard.any() or hi_guard.any()):
        raise ValueError("band/guard outside the analyzed range")
    guards = []
    if lo_guard.any():
        guards.append(db[lo_guard])
    if hi_guard.any():
        guards.append(db[hi_guard])
    noise = float(np.median(np.concatenate(guards)))
    return float(db[band].max() - noise)


# --------------------------------------------------------------------------
# Carrier detection
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Carrier:
    freq_hz: float
    power_db: float           # PSD level at the peak
    prominence_db: float      # peak above the local noise floor
    frames_present: int       # of the frames analyzed
    frames_total: int
    bandwidth_hz: float       # -3 dB width of the peak region
    abs_freq_hz: float = 0.0  # freq_hz + the caller's center_hz (0.0 if
                              # the caller didn't supply one -- baseband-
                              # relative and absolute then coincide)


@dataclass(frozen=True)
class CarrierScan:
    carriers: tuple[Carrier, ...]
    noise_floor_db: float
    sample_rate: float


def find_carriers(
    samples: np.ndarray,
    sample_rate: float,
    prominence_db: float = 6.0,
    min_persistence: float = 0.5,
    nfft: int = 2048,
    fmin: float | None = None,
    fmax: float | None = None,
    center_hz: float = 0.0,
) -> CarrierScan:
    """Detect spectral peaks that persist across frames.

    A peak is reported when it stands `prominence_db` above the local
    noise floor (median of the PSD) in the mean PSD AND appears in at
    least `min_persistence` of the individual FFT frames at that bin.

    `center_hz` is the absolute RF frequency this buffer's own 0 Hz
    corresponds to (e.g. an SDR's tuned center frequency) -- purely
    additive bookkeeping, every `Carrier.abs_freq_hz` is just
    `freq_hz + center_hz`. Leave it at 0.0 (the default) when there is no
    such notion, in which case `abs_freq_hz` and `freq_hz` coincide.
    """
    spec = spectrogram(samples, sample_rate, block_size=nfft, hop=nfft // 2)
    mean_psd = (spec.magnitudes ** 2).mean(axis=0)
    mean_db = 10 * np.log10(mean_psd + 1e-20)
    freqs = spec.freqs
    sel = np.ones(len(freqs), dtype=bool)
    if fmin is not None:
        sel &= freqs >= fmin
    if fmax is not None:
        sel &= freqs <= fmax
    sel &= freqs != 0.0                               # DC
    if not sel.any():
        return CarrierScan((), float(mean_db[sel].mean() if sel.any() else 0.0), sample_rate)
    noise_floor = float(np.median(mean_db[sel]))
    frame_db = 20 * np.log10(spec.magnitudes + 1e-12)
    carriers: list[Carrier] = []
    used = np.zeros(len(freqs), dtype=bool)
    order = np.argsort(mean_db)[::-1]
    for idx in order:
        if not sel[idx] or used[idx]:
            continue
        peak_db = mean_db[idx]
        if peak_db - noise_floor < prominence_db:
            break
        # local floor: bins +/-25 bins away, excluding the peak region
        lo, hi = max(0, idx - 25), min(len(freqs), idx + 26)
        local = np.concatenate([mean_db[lo:max(lo, idx - 6)],
                                mean_db[min(hi, idx + 7):hi]])
        floor = float(np.median(local)) if len(local) else noise_floor
        prom = peak_db - floor
        if prom < prominence_db:
            continue
        # -3 dB bandwidth around the peak
        half = peak_db - 3.0
        j = idx
        while j > 0 and mean_db[j] > half:
            j -= 1
        k = idx
        while k < len(freqs) - 1 and mean_db[k] > half:
            k += 1
        bw = (k - j) * (sample_rate / nfft)
        # persistence across frames (max within the -3 dB region)
        region = frame_db[:, j:k + 1].max(axis=1)
        thresh = peak_db - 8.0
        present = float((region > thresh).mean())
        if present < min_persistence:
            continue
        used[j:k + 1] = True
        carriers.append(Carrier(freq_hz=float(freqs[idx]), power_db=float(peak_db),
                                prominence_db=float(prom), frames_present=int(present * len(region)),
                                frames_total=int(len(region)), bandwidth_hz=float(bw),
                                abs_freq_hz=float(freqs[idx]) + center_hz))
    return CarrierScan(carriers=tuple(sorted(carriers, key=lambda c: -c.power_db)),
                       noise_floor_db=noise_floor, sample_rate=sample_rate)


# --------------------------------------------------------------------------
# Time-domain activity detection
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ActivitySegment:
    start_s: float
    end_s: float
    energy_db: float        # segment mean energy above the noise floor


def detect_activity(samples: np.ndarray, sample_rate: float,
                    block_ms: float = 10.0, threshold_db: float = 6.0) -> tuple[ActivitySegment, ...]:
    """Energy-activity segmentation: block RMS energy vs the noise floor
    (10th percentile of block energies), hysteresis-merged segments."""
    x = np.asarray(samples).ravel()
    x = np.abs(x) if np.iscomplexobj(x) else x.astype(np.float64, copy=False)
    block = max(1, int(sample_rate * block_ms / 1000))
    n = len(x) // block
    if n < 4:
        return ()
    e = np.array([np.sqrt(np.mean(x[i * block:(i + 1) * block] ** 2)) for i in range(n)])
    e_db = 20 * np.log10(e + 1e-12)
    floor = float(np.percentile(e_db, 10))
    # a constant-energy record has no activity structure: one full segment
    if e_db.max() - floor < threshold_db:
        return (ActivitySegment(start_s=0.0, end_s=len(x) / sample_rate,
                                energy_db=float(e_db.mean() - floor)),)
    on = e_db > floor + threshold_db
    segs: list[ActivitySegment] = []
    i = 0
    while i < n:
        if on[i]:
            j = i
            while j < n and (on[j] or (j + 1 < n and on[j + 1])):
                j += 1
            seg_db = float(e_db[i:j + 1].mean() - floor)
            segs.append(ActivitySegment(start_s=i * block / sample_rate,
                                        end_s=(j + 1) * block / sample_rate,
                                        energy_db=seg_db))
            i = j + 1
        else:
            i += 1
    return tuple(segs)


# --------------------------------------------------------------------------
# Modulation classification
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Classification:
    label: str             # 'noise' | 'tone' | 'FSK' | 'OOK' | 'PSK-like' | 'OFDM'
    confidence: float      # 0..1
    features: dict[str, float] = field(default_factory=dict)


def classify_modulation(samples: np.ndarray, sample_rate: float,
                        fmin: float = 300.0, fmax: float | None = None) -> Classification:
    """Decision-tree modulation classifier on recovered features.

    Features (all computed from the record, no training):
    - tone_ratio: fraction of total power in the top-3 PSD bins;
    - two_tone_ratio: power in the top-2 separated tones;
    - envelope_cv: coefficient of variation of the analytic envelope
      (OOK high, PSK near zero);
    - spectral_flatness: geometric/arithmetic mean of the in-band PSD
      (OFDM/PSK high, tone low);
    - cp_score: OFDM cyclic-prefix autocorrelation score.
    """
    x = np.asarray(samples).ravel()
    complex_input = np.iscomplexobj(x)
    if not complex_input:
        x = x.astype(np.float64, copy=False)
    if len(x) < 512:
        return Classification("noise", 0.0, {})
    freqs, db = psd(x, sample_rate, nfft=min(4096, len(x)))
    band = np.ones(len(freqs), dtype=bool)
    band &= freqs != 0.0
    if fmin is not None:
        band &= freqs >= fmin
    if fmax is not None:
        band &= freqs <= fmax
    if not band.any():
        return Classification("noise", 0.5, {})
    p = 10 ** (db / 10)
    pb = p[band]
    total = pb.sum() + 1e-20
    top = np.sort(pb)[::-1][:3].sum() / total
    # two strongest well-separated peaks, measured as power in +/-50 Hz
    # bands around them (a binned top-2 metric underestimates when bit
    # transitions smear each tone across neighboring bins)
    idx_band = np.where(band)[0]
    order = np.argsort(db[band])[::-1]
    i1 = idx_band[order[0]]
    i2 = next((idx_band[o] for o in order[1:]
               if abs(idx_band[o] - i1) * (sample_rate / len(db)) > 200), None)
    two_tone = 0.0
    if i2 is not None:
        res = sample_rate / len(db)
        b1 = np.abs(freqs - freqs[i1]) <= 50.0
        b2 = np.abs(freqs - freqs[i2]) <= 50.0
        two_tone = float((p[b1].sum() + p[b2].sum()) / (p[band].sum() + 1e-20))
    env = np.abs(x if complex_input else _analytic(x))
    env_cv = float(env.std() / (env.mean() + 1e-12))
    lin = pb + 1e-20
    flatness = float(np.exp(np.log(lin).mean()) / lin.mean())
    # CP correlation score: max over candidate lags AND window positions
    # (the CP structure correlates at every symbol boundary; scoring only
    # the first position is fragile -- a boundary landing mid-symbol
    # halves the score)
    cp_score = 0.0
    for lag in (256, 512, 640, 1024, 1280, 2048):
        n = min(1024, len(x) - 2 * lag)
        if n < 128:
            continue
        for pos in range(3):
            base = pos * n
            a = x[base + lag:base + lag + n]
            b = x[base:base + n]
            if len(a) < n:
                break
            denom = np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)
            if denom:
                corr = np.vdot(b, a) / denom
                cp_score = max(cp_score, float(abs(corr) if complex_input else corr))
    # strong spectral peak count (FSK=2 tones, PSK=3 carrier+sidebands,
    # OFDM=many): local maxima within 10 dB of the band max, merged
    peak_count = 0
    band_db = db[band]
    mx = band_db.max()
    idx = [i for i in range(2, len(band_db) - 2)
           if band_db[i] >= band_db[i - 1] and band_db[i] >= band_db[i + 1]
           and band_db[i] > mx - 10]
    for j, i in enumerate(idx):
        if j == 0 or i - idx[j - 1] > 3:
            peak_count += 1
    feats = {"tone_ratio": float(top), "two_tone_ratio": float(two_tone),
             "envelope_cv": env_cv, "spectral_flatness": flatness,
             "cp_score": cp_score, "peak_count": float(peak_count)}
    if top > 0.5 and env_cv < 0.35:
        return Classification("tone", min(1.0, top), feats)
    # Constant-envelope two-tone: BPSK (carrier + sidebands) and 2FSK
    # (mark/space) are spectrally near-identical at this feature order --
    # the honest label is the family, and the decoder sweep names the
    # exact waveform. FSK keeps its own label when the envelope varies
    # (audio-band FSK with visible transition dips).
    if two_tone > 0.3 and env_cv < 0.5:
        if env_cv > 0.15:
            return Classification("FSK", min(1.0, two_tone * 2.0), feats)
        return Classification("FSK/PSK", min(1.0, two_tone), feats)
    if env_cv > 0.8:
        return Classification("OOK", min(1.0, env_cv), feats)
    if flatness > 0.05 and cp_score > 0.2:
        return Classification("OFDM", min(1.0, cp_score * 1.5), feats)
    if flatness > 0.2 and env_cv < 0.35:
        return Classification("PSK-like", min(1.0, flatness * 3), feats)
    if top < 0.2 and flatness > 0.3:
        return Classification("noise", 0.6, feats)
    return Classification("unknown", 0.3, feats)


# --------------------------------------------------------------------------
# Band scan
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ScanHit:
    center_hz: float
    bandwidth_hz: float
    snr_db: float
    classification: Classification
    activity_segments: int
    abs_center_hz: float = 0.0   # center_hz + the caller's absolute tuning
                                 # center_hz (0.0 if none given)


def band_scan(samples: np.ndarray, sample_rate: float,
              channel_bw_hz: float = 3000.0, fmin: float | None = None,
              fmax: float | None = None, min_snr_db: float = 6.0,
              center_hz: float = 0.0) -> tuple[ScanHit, ...]:
    """Scan the record in `channel_bw_hz` channels; report occupied ones.

    Each channel gets an SNR test against its neighbor average and a
    modulation classification. Only channels above `min_snr_db` are hits.

    `center_hz` is the absolute RF frequency this buffer's own 0 Hz
    corresponds to; every hit's `abs_center_hz` is its (baseband-relative)
    `center_hz` field plus this parameter. Leave at 0.0 (default) when
    there is no such notion, in which case `abs_center_hz` and
    `center_hz` coincide.
    """
    x = np.asarray(samples).ravel()
    freqs, db = psd(x, sample_rate)
    nyq = sample_rate / 2
    lo = fmin if fmin is not None else float(freqs[0])
    hi = fmax if fmax is not None else nyq
    if not np.isfinite(channel_bw_hz) or channel_bw_hz <= 0:
        raise ValueError("channel bandwidth must be positive and finite")
    hits: list[ScanHit] = []
    step = channel_bw_hz / 2
    c = lo + channel_bw_hz / 2
    while c < min(hi, nyq) - channel_bw_hz / 2:
        chan = (freqs >= c - channel_bw_hz / 2) & (freqs <= c + channel_bw_hz / 2)
        if chan.sum() > 8:
            gp = ~chan
            gp &= (freqs >= c - channel_bw_hz * 1.5) & (freqs <= c + channel_bw_hz * 1.5)
            if gp.sum() > 4:
                power = 10 ** (db[chan].mean() / 10)
                noise = 10 ** (db[gp].mean() / 10)
                snr = 10 * np.log10(max(power, 1e-20) / (noise + 1e-20))
                if snr >= min_snr_db:
                    f_lo, f_hi = c - channel_bw_hz / 2, c + channel_bw_hz / 2
                    t = np.arange(len(x)) / sample_rate
                    z = (x if np.iscomplexobj(x) else _analytic(x)) * np.exp(-2j * np.pi * c * t)
                    segs = detect_activity(np.abs(z), sample_rate)
                    cls = classify_modulation(np.abs(z), sample_rate,
                                              fmin=0.0, fmax=channel_bw_hz / 2)
                    half = (db[chan].max() - 3.0)
                    edge = np.where(db[chan] > half)[0]
                    bw = max(1.0, (edge.max() - edge.min()) * (freqs[1] - freqs[0])) if len(edge) else channel_bw_hz
                    hits.append(ScanHit(center_hz=c, bandwidth_hz=float(bw),
                                        snr_db=float(snr), classification=cls,
                                        activity_segments=len(segs),
                                        abs_center_hz=c + center_hz))
        c += step
    return tuple(sorted(hits, key=lambda h: -h.snr_db))
