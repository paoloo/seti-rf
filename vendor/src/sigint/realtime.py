"""Real-time recognition gate: stream samples in, recognition events out.

Import `Gate` into any pipeline (GNU Radio via a Python block, CyberEther,
a sounddevice reader, an SDR CLI) and feed it samples; it emits
:class:`RecognitionEvent`s whenever a signal appears, is classified, or
locks. No blocking, no threads required — call :meth:`Gate.process` with
whatever samples you have and collect the returned events.

Design:
- all state lives in the Gate; the pipeline owns the source thread/timer;
- a rolling ring buffer holds the last ~window of samples for the
  classifiers;
- the VAD (block-energy vs running noise floor) gates everything else, so
  cost is ~zero on silence;
- on sustained activity the gate classifies, tries to lock FSK baud, and
  starts the (optional) demodulator; demodulated bits accumulate in a
  rolling bit buffer for decoder sweeps;
- events are monotone: SIGNAL_START -> SIGNAL_CLASSIFIED -> FSK_LOCKED ->
  BITS (rolling) -> SIGNAL_END.

Audio-band convention: one channel, float64 (-1..1 or arbitrary scale),
sample_rate known. IQ sources channelize first (tune + decimate) and feed
the magnitude/real part or a re-tuned channel; the Gate is bandwidth-blind.

Feeding from GNU Radio: a Python block's `work()` passes its input items
to `gate.process(samples)` and publishes `gate.drain_events()` as messages.
Feeding from a file/pipe: see `sigint/rt_gate_cli.py` (stdin/ wav/ raw ->
events on stdout) — the same path a network stream would use.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time

import numpy as np

from . import detect as det
from . import dsp


@dataclass(frozen=True)
class RecognitionEvent:
    """Something the gate noticed. `kind` is one of:
    SIGNAL_START, SIGNAL_CLASSIFIED, FSK_LOCKED, SIGNAL_END."""
    kind: str
    t_s: float              # stream timestamp (seconds since gate start)
    band_hz: tuple[float, float] | None = None   # occupied band, if known
                                                  # (relative to this
                                                  # buffer's own baseband)
    abs_band_hz: tuple[float, float] | None = None  # band_hz shifted by
                                                     # GateConfig.center_hz;
                                                     # equals band_hz when
                                                     # center_hz is 0.0
    label: str | None = None                     # classification label
    confidence: float | None = None
    snr_db: float | None = None
    baud_hz: float | None = None
    detail: str = ""

    def as_dict(self) -> dict:
        d = {"kind": self.kind, "t_s": round(self.t_s, 3)}
        for k in ("band_hz", "abs_band_hz", "label", "snr_db", "baud_hz"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        if self.confidence is not None:
            d["confidence"] = round(self.confidence, 3)
        if self.detail:
            d["detail"] = self.detail
        return d


@dataclass
class GateConfig:
    sample_rate: float = 16000.0
    center_hz: float = 0.0             # absolute RF frequency this
                                        # stream's own 0 Hz corresponds to
                                        # (e.g. an SDR's tuned center
                                        # frequency); purely additive --
                                        # 0.0 (default) means no such
                                        # notion, and abs_band_hz coincides
                                        # with band_hz on every event
    vad_block_s: float = 0.010        # energy block for the VAD
    vad_threshold_db: float = 6.0     # above the running noise floor
    window_s: float = 2.0             # rolling window for classifiers
    classify_after_s: float = 0.5     # active for this long -> classify
    classify_every_s: float = 2.0     # re-classify while active
    end_after_s: float = 1.0          # quiet for this long -> SIGNAL_END
    demod: bool = False               # attempt FSK demod while active
    mark_hz: float | None = None      # required for demod
    space_hz: float | None = None
    baud_hz: float | None = None      # nominal; None = measure once active
    bit_history: int = 4096           # rolling demodulated-bit buffer
    min_snr_db: float = 5.0           # classification SNR floor
    relock_every_s: float = 2.0       # re-estimate the rate this often
                                      # until the lock is stable
    sweep: bool = True                # name waveforms via the BEE decoder
                                      # sweep (needs the bee package)
    sweep_every_s: float = 3.0        # re-sweep cadence while active
    hfdl: bool = True                 # probe PSK-classified bursts with
                                      # the HFDL PHY decoder (one attempt
                                      # per burst, ~2 s of audio)


@dataclass
class _Active:
    start_t: float = 0.0
    last_energy_t: float = 0.0
    last_classify_t: float = -1e9
    classified: bool = False
    label: str | None = None
    band: tuple[float, float] | None = None
    abs_band: tuple[float, float] | None = None
    locked_baud: float | None = None
    lock_final: bool = False
    prev_lock: float | None = None
    last_relock_t: float = 0.0
    last_sweep_t: float = -1e9
    bits: deque = field(default_factory=lambda: deque(maxlen=4096))
    samples: deque = field(default_factory=lambda: deque(maxlen=320000))
    hfdl_probed: bool = False


class Gate:
    """Streaming recognition gate. Feed samples, collect events.

    Typical pipeline loop:
        gate = Gate(sample_rate=16000)
        for chunk in source:                 # whatever cadence
            for ev in gate.process(chunk):
                handle(ev)                   # SIGNAL_START / FSK_LOCKED ...
        # latest demodulated bits, always available:
        bits = np.asarray(gate.bits)
    """

    def __init__(self, config: GateConfig | None = None,
                 sample_rate: float | None = None, demod: bool = False,
                 mark_hz: float | None = None, space_hz: float | None = None,
                 baud_hz: float | None = None):
        if sample_rate is not None:
            if config is None:
                config = GateConfig()
            config.sample_rate = sample_rate
        if demod:
            if config is None:
                config = GateConfig()
            config.demod = True
            config.mark_hz = mark_hz if mark_hz is not None else config.mark_hz
            config.space_hz = space_hz if space_hz is not None else config.space_hz
            config.baud_hz = baud_hz if baud_hz is not None else config.baud_hz
        if config is None:
            config = GateConfig()
        if config.demod and (config.mark_hz is None or config.space_hz is None):
            raise ValueError("demod requires mark_hz and space_hz")
        self.cfg = config
        self._block = max(1, int(config.sample_rate * config.vad_block_s))
        self._win = int(config.sample_rate * config.window_s)
        self._buf = np.zeros(0)
        self._t = 0.0                     # seconds of stream consumed
        self._noise_db: float | None = None
        self._active: _Active | None = None
        self._vad_energy_db: deque = deque(maxlen=200)
        self.bits: np.ndarray = np.zeros(0, dtype=np.int8)
        self.waveforms: list[tuple[str, int, int]] = []   # latest sweep
        self.hfdl_frames: list = []       # latest HFDL decode, if any
        self._demod_tail = np.zeros(0)
        self._demod_phase: int = 0
        self._window = np.zeros(0)        # rolling window for classifiers
        self._flat_buf = np.zeros(0)

    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        return self._active is not None

    @property
    def stream_time(self) -> float:
        return self._t

    def process(self, samples: np.ndarray) -> list[RecognitionEvent]:
        """Feed the next chunk of samples; return any recognition events."""
        x = np.asarray(samples, dtype=np.float64).ravel()
        if len(x) == 0:
            return []
        events: list[RecognitionEvent] = []
        # rolling classifier window
        self._window = np.concatenate([self._window, x])[-self._win:]
        # rolling buffer: keep the classifier window + block alignment
        self._buf = np.concatenate([self._buf, x])
        self._t += len(x) / self.cfg.sample_rate

        # process in VAD blocks; keep a tail of < block samples
        while len(self._buf) >= self._block:
            block = self._buf[:self._block]
            self._buf = self._buf[self._block:]
            events += self._process_block(block)
        return events

    # ------------------------------------------------------------------

    def _flat_energy_is_signal(self, block: np.ndarray) -> bool:
        """Constant-envelope discrimination: is this flat energy a signal
        (structured spectrum) or noise (flat spectrum)? White noise shows
        a couple of random ±10 dB spikes; a real signal shows a stable
        comb of carriers. Require >= 2 strong carriers as the bar."""
        try:
            scan = det.find_carriers(block, self.cfg.sample_rate,
                                     prominence_db=10.0, min_persistence=0.9)
        except ValueError:
            # block shorter than the FFT: accumulate until we can check
            return False
        strong = [c for c in scan.carriers if c.prominence_db >= 12.0]
        return len(strong) >= 2

    def _process_block(self, block: np.ndarray) -> list[RecognitionEvent]:
        cfg = self.cfg
        t_block_end = self._t - len(self._buf) / cfg.sample_rate
        e_db = 20 * np.log10(float(np.sqrt(np.mean(block ** 2))) + 1e-12)
        self._vad_energy_db.append(e_db)

        # running noise floor: 10th percentile of recent block energies
        floor = float(np.percentile(self._vad_energy_db, 10))
        self._noise_db = floor
        spread = max(self._vad_energy_db) - floor
        if spread < 3.0:
            # flat energy: either constant signal or constant noise. Use
            # spectral content to tell them apart: a real signal has
            # carriers standing above the band median. Wait for enough
            # samples (the classifier FFT needs ~2048) before deciding.
            self._flat_buf = np.concatenate([getattr(self, "_flat_buf",
                                                     np.zeros(0)), block])
            if len(self._flat_buf) < 2048:
                return []
            loud = self._flat_energy_is_signal(self._flat_buf)
            self._flat_buf = self._flat_buf[1024:]   # rolling re-check
        else:
            loud = e_db > floor + cfg.vad_threshold_db

        if loud and self._active is None:
            self._active = _Active(start_t=t_block_end - cfg.vad_block_s,
                                   last_energy_t=t_block_end)
            self._active.bits = deque(maxlen=cfg.bit_history)
            self._active.samples.extend(block.tolist())  # probe needs
                                                         # the burst head
            return [RecognitionEvent("SIGNAL_START", t_s=t_block_end)]
        if not loud:
            if self._active is not None:
                a = self._active
                # keep accumulating through the quiet tail: a fading
                # frame tail must reach the waveform probe
                a.samples.extend(block.tolist())
                if t_block_end - a.last_energy_t > cfg.end_after_s:
                    ev = RecognitionEvent("SIGNAL_END", t_s=t_block_end,
                                          label=a.label, baud_hz=a.locked_baud,
                                          band_hz=a.band, abs_band_hz=a.abs_band,
                                          detail=f"duration="
                                                 f"{t_block_end - a.start_t:.1f}s")
                    # final waveform probe on the complete burst audio
                    if cfg.hfdl:
                        a.hfdl_probed = False   # allow the final attempt
                        end_events = self._hfdl_probe(t_block_end, a)
                    else:
                        end_events = []
                    self._active = None
                    return [ev] + end_events
            return []
        # loud and active
        a = self._active
        a.last_energy_t = t_block_end
        a.samples.extend(block.tolist())
        window = self._current_window()
        out: list[RecognitionEvent] = []

        # occupied band from the classifier window's carriers. Computed
        # before classification so the band it reports is fresh for
        # events emitted later in this same cycle, not the previous
        # cycle's.
        try:
            scan = det.find_carriers(window, cfg.sample_rate,
                                     prominence_db=8.0, center_hz=cfg.center_hz)
            if scan.carriers:
                lo = min(c.freq_hz for c in scan.carriers)
                hi = max(c.freq_hz for c in scan.carriers)
                a.band = (lo, hi)
                a.abs_band = (lo + cfg.center_hz, hi + cfg.center_hz)
        except ValueError:
            pass

        # classify once after classify_after_s, then every classify_every_s
        # The FIRST confident label is sticky: a random-data re-classify
        # on a later window must not flip the reported modulation (PSK's
        # symmetric sidebands read as FSK on some windows). Later
        # classifications are still emitted but only overwrite when they
        # agree with a different, consistently-seen family.
        if (not a.classified and
                t_block_end - a.start_t >= cfg.classify_after_s) or \
                (a.classified and
                 t_block_end - a.last_classify_t >= cfg.classify_every_s):
            cls, snr = self._classify(window)
            a.last_classify_t = t_block_end
            if not a.classified:
                a.classified = True
                a.label = cls.label
                out.append(RecognitionEvent(
                    "SIGNAL_CLASSIFIED", t_s=t_block_end, label=cls.label, band_hz=a.band, abs_band_hz=a.abs_band,
                    confidence=cls.confidence, snr_db=snr))
            elif cls.label == a.label:
                a.last_classify_t = t_block_end
                out.append(RecognitionEvent(
                    "SIGNAL_CLASSIFIED", t_s=t_block_end, label=cls.label, band_hz=a.band, abs_band_hz=a.abs_band,
                    confidence=cls.confidence, snr_db=snr))
            elif a.label in (None, "unknown", "noise") and \
                    cls.label not in (None, "unknown", "noise") and \
                    cls.confidence >= 0.5:
                # upgrade: the first window was too short to resolve
                # (e.g. OFDM's CP score needs ~2 s) — promote the label
                a.label = cls.label
                out.append(RecognitionEvent(
                    "SIGNAL_CLASSIFIED", t_s=t_block_end, label=cls.label, band_hz=a.band, abs_band_hz=a.abs_band,
                    confidence=cls.confidence, snr_db=snr))
            # else: label disagrees; keep the sticky one, refresh nothing

        # waveform-level probe: named PHY decoders (HFDL) on the
        # accumulated burst audio. One attempt mid-burst once ~2.1 s is
        # up (catches long frames / bursts already in progress); the
        # burst END always gets a final attempt on the complete audio
        # (a 1.8 s frame is incomplete before SIGNAL_END).
        # No label gate: the PHY decoder self-rejects via its preamble
        # correlation threshold (family classifiers mislabel PSK bursts
        # as OOK too often to trust here).
        if cfg.hfdl and not a.hfdl_probed and \
                t_block_end - a.start_t >= 2.1:
            out += self._hfdl_probe(t_block_end, a)

        # FSK lock + demod (lock before classify so FSK_LOCKED precedes
        # the classification in the event stream)
        if cfg.demod:
            out += self._demod_step(window, t_block_end, a)

        # waveform naming: sweep the BEE decoders on the rolling bits
        if cfg.sweep and t_block_end - a.last_sweep_t >= min(
                cfg.sweep_every_s, 1.0) and len(self.bits) >= 64:
            a.last_sweep_t = t_block_end
            out += self._sweep_step(t_block_end)
        return out

    def _hfdl_probe(self, t: float, a: _Active) -> list[RecognitionEvent]:
        """One HFDL PHY decode attempt per burst on the accumulated
        activity audio. Emits a WAVEFORM event on a frame match and
        keeps the frames in gate.hfdl_frames."""
        a.hfdl_probed = True
        try:
            from . import hfdl as hfdl_mod
            from scipy import signal as _sp
            x = np.asarray(a.samples, dtype=np.float64)
            sr = self.cfg.sample_rate
            f, P = _sp.welch(x, sr, nperseg=1024)
            tone = f[int(np.argmax(P))]
            if not (800 < tone < 2600):
                tone = hfdl_mod.SSB_CARRIER_OFFSET_HZ
            ana = _sp.hilbert(x)
            bb = ana * np.exp(-2j * np.pi * tone * np.arange(len(x)) / sr)
            frames = hfdl_mod.decode_frame(bb, sr)
            frames = [frames] if frames else []
        except Exception:
            return []
        if not frames:
            return []
        self.hfdl_frames = frames
        fr = frames[0]
        return [RecognitionEvent(
            "WAVEFORM", t_s=t, label=f"HFDL {fr.label}",
            detail=f"{fr.symbols} sym, {len(fr.bytes_out)} bytes, "
                   f"{'CRC OK' if fr.crc_ok else 'CRC bad'}")]

    def _sweep_step(self, t: float) -> list[RecognitionEvent]:
        """Run the decoder sweep on the rolling bits; publish the top
        named waveforms as a WAVEFORM event (and gate.waveforms)."""
        from . import decoders as sd
        bits = np.asarray(self.bits)
        if len(bits) < 64:
            return []
        outcomes = sd.run_all(bits, top=5)
        if not outcomes:
            return []
        self.waveforms = [(o.decoder, o.symbol_count, o.error_count)
                          for o in outcomes]
        best = outcomes[0]
        return [RecognitionEvent(
            "WAVEFORM", t_s=t, label=best.decoder,
            detail=f"candidates: "
                   f"{', '.join(o.decoder for o in outcomes[:3])}"
                   + (f" ({len(bits)} bits)" if len(bits) else ""))]

    def _relock(self, window: np.ndarray, t: float,
                a: _Active) -> list[RecognitionEvent]:
        """Re-estimate the symbol rate on the grown activity window.

        The coarse estimator's quality depends on how much signal it has
        seen; a bad first lock self-corrects here because the relock
        searches a WIDE range (0.5x..3x of the current estimate) with the
        phase-swept confidence score, and repeats until the rate is
        stable across two consecutive relocks.
        """
        cfg = self.cfg
        out: list[RecognitionEvent] = []

        def rate_score(r: float) -> float:
            sps = max(2, int(round(cfg.sample_rate / r)))
            n = len(window) // sps
            if n < 16:
                return 0.0
            t_sym = np.arange(sps) / cfg.sample_rate
            m_ref = np.exp(-2j * np.pi * cfg.mark_hz * t_sym)
            s_ref = np.exp(-2j * np.pi * cfg.space_hz * t_sym)
            # phase-swept AND window-offset-swept: a window boundary that
            # lands mid-symbol depresses the score by ~30% and can make a
            # harmonic outscore the fundamental. Try a few window starts.
            best = 0.0
            for off in (0, sps // 4, sps // 2, 3 * sps // 4):
                w2 = window[off:]
                n2 = len(w2) // sps
                if n2 < 16:
                    continue
                for ph in np.linspace(0, sps - 1, 10).astype(int):
                    vals = []
                    for k in range(n2):
                        seg = w2[ph + k * sps:ph + (k + 1) * sps]
                        if len(seg) < sps:
                            break
                        vals.append(abs(np.dot(seg, m_ref))
                                    - abs(np.dot(seg, s_ref)))
                    sc = float(np.mean(np.abs(np.asarray(vals)))) / np.sqrt(sps)
                    best = max(best, sc)
            return best

        # wide search around the current estimate; the confidence peak is
        # sharp so the grid must be fine (1.5% steps)
        lo = max(5.0, a.locked_baud * 0.5)
        hi = min(cfg.sample_rate / 4, a.locked_baud * 3.0)
        n_steps = max(40, int(np.log(hi / lo) / np.log(1.015)) + 1)
        rates = np.exp(np.linspace(np.log(lo), np.log(hi), n_steps))
        p1 = max(((rate_score(r), r) for r in rates),
                 default=(0.0, a.locked_baud))
        p2 = max(((rate_score(r), r)
                  for r in p1[1] * np.linspace(0.97, 1.03, 13)),
                 default=(0.0, a.locked_baud))
        new_rate = p2[1]
        changed = abs(new_rate - a.locked_baud) / a.locked_baud > 0.01
        if changed:
            a.locked_baud = new_rate
            a.bits.clear()
            self.bits = np.zeros(0, dtype=np.int8)
            self._demod_tail = np.zeros(0)
            self._demod_phase = 0
            out.append(RecognitionEvent("FSK_LOCKED", t_s=t,
                                        baud_hz=a.locked_baud,
                                        detail="refined"))
        # stability: two relocks agreeing within 1% mark the lock final
        if a.prev_lock is not None and \
                abs(new_rate - a.prev_lock) / a.prev_lock <= 0.01:
            a.lock_final = True
        a.prev_lock = new_rate
        return out

    def _current_window(self) -> np.ndarray:
        """Most recent `window_s` of the stream."""
        return self._window

    def _classify(self, window: np.ndarray) -> tuple[det.Classification, float]:
        cfg = self.cfg
        try:
            snr = det.snr_estimate(window, cfg.sample_rate,
                                   1.0, cfg.sample_rate / 2 - 1.0)
        except ValueError:
            snr = float("nan")
        cls = det.classify_modulation(window, cfg.sample_rate)
        return cls, snr

    def _demod_step(self, window: np.ndarray, t: float,
                    a: _Active) -> list[RecognitionEvent]:
        cfg = self.cfg
        out: list[RecognitionEvent] = []
        if a.locked_baud is None:
            # lock once: coarse measure on the accumulated activity window
            try:
                ev = dsp.baud_rate_stages(window, cfg.sample_rate,
                                          required_baud=cfg.baud_hz)
                a.locked_baud = ev.rate_hz
                a.last_sweep_t = t  # first waveform sweep ~1 s after lock
                out.append(RecognitionEvent(
                    "FSK_LOCKED", t_s=t, baud_hz=a.locked_baud,
                    detail=f"mark {cfg.mark_hz:.0f}/space {cfg.space_hz:.0f} Hz"
                           f" corr {ev.correlation:.2f}"))
            except ValueError:
                return out
        elif not a.lock_final and \
                t - a.last_relock_t >= cfg.relock_every_s and \
                t - a.start_t >= 1.0:
            # periodic relock on the grown window until the rate is stable
            out += self._relock(window, t, a)
            a.last_relock_t = t
        # streaming demod: correlate only the NEW samples since the last
        # step against the tone references, tracking the phase within the
        # symbol. The decision is per full symbol period; partial symbols
        # carry over via _demod_tail.
        sps = max(2, int(round(cfg.sample_rate / a.locked_baud)))
        tail = np.concatenate([self._demod_tail, window])
        n_sym = len(tail) // sps
        if n_sym < 1:
            return out
        t_sym = np.arange(sps) / cfg.sample_rate
        m_ref = np.exp(-2j * np.pi * cfg.mark_hz * t_sym)
        s_ref = np.exp(-2j * np.pi * cfg.space_hz * t_sym)
        ph = int(self._demod_phase)
        new_bits = []
        for k in range(n_sym):
            seg = tail[ph + k * sps:ph + (k + 1) * sps]
            if len(seg) < sps:
                break
            m = abs(np.dot(seg, m_ref))
            s = abs(np.dot(seg, s_ref))
            new_bits.append(1 if m > s else 0)
        # consume the whole symbols; keep the remainder for next time
        consumed = ph + n_sym * sps
        self._demod_tail = tail[consumed:]
        self._demod_phase = 0.0
        if new_bits:
            for b in new_bits:
                a.bits.append(int(b))
            self.bits = np.asarray(a.bits, dtype=np.int8)
        return out

    # ------------------------------------------------------------------

    def drain_events(self) -> list[RecognitionEvent]:
        """Pipelines that prefer pulling: no-op placeholder (events are
        returned from process()); kept for message-block symmetry."""
        return []
