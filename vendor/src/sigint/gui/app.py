"""The sigint workbench: one window, an analyst's workflow.

Layout (left -> right):
- Control panel: file open/save, playback region, FFT/window controls,
  detection actions (Find Carriers, Classify, Band Scan, Decode Sweep),
  dataset validation launcher.
- Waterfall (time x frequency, dB color) with a synced instantaneous
  spectrum strip above it; click/drag on either selects a band and a time
  region; scroll zooms time.
- Analysis panel: SNR + classification of the current selection,
  carrier list, scan hits table, decoder sweep results, constellation and
  phase views of the selection.

Everything runs on the main thread with computation chunked via
`after()` jobs so the UI stays responsive on multi-second records.

Runnable standalone (from inside sigint/) as:  python3 gui/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # standalone: sigint/gui/app.py
    _PKG_PARENT = str(Path(__file__).resolve().parent.parent.parent)
    if _PKG_PARENT not in sys.path:
        sys.path.insert(0, _PKG_PARENT)
    __package__ = "sigint.gui"

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from .. import dataset as ds
from .. import detect as det
from .. import decoders as sd
from .. import dsp
from ..dsp import SlideStep, WindowType, window_label
from ._io import load_recording, save_recording
from ._render import PointsView, SpectrumLine, WaterfallImage


class Workbench(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("sigint -- RF workbench")
        self.geometry("1420x880")

        self.samples: np.ndarray = np.zeros(0)
        self.sample_rate: float = 16000.0
        self.path: str | None = None

        # selection state (sample indices)
        self.sel_band: tuple[float, float] | None = None
        self.sel_time: tuple[float, float] | None = None

        self._build_ui()

    # ---------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # left control panel
        left = ttk.Frame(self, padding=6)
        left.grid(row=0, column=0, sticky="ns")
        self._build_controls(left)

        # center: spectrum strip + waterfall
        center = ttk.Frame(self, padding=4)
        center.grid(row=0, column=1, sticky="nsew")
        center.rowconfigure(1, weight=1)
        center.columnconfigure(0, weight=1)

        self.spec_canvas = tk.Canvas(center, height=140, bg="black", highlightthickness=0)
        self.spec_canvas.grid(row=0, column=0, sticky="ew")
        self.spec_view = SpectrumLine(self.spec_canvas, 800, 140)
        self.spec_canvas.bind("<Button-1>", self._on_spec_click)
        self.spec_canvas.bind("<B1-Motion>", self._on_spec_drag)
        self.spec_canvas.bind("<ButtonRelease-1>", self._on_spec_release)

        self.wf_canvas = tk.Canvas(center, bg="black", highlightthickness=0)
        self.wf_canvas.grid(row=1, column=0, sticky="nsew")
        self.wf_view = WaterfallImage(self.wf_canvas)
        self.wf_canvas.bind("<Button-1>", self._on_wf_click)
        self.wf_canvas.bind("<B1-Motion>", self._on_wf_drag)
        self.wf_canvas.bind("<ButtonRelease-1>", self._on_wf_release)
        self.wf_canvas.bind("<Configure>", lambda e: self.refresh_views())

        # right analysis panel (notebook)
        right = ttk.Frame(self, padding=6)
        right.grid(row=0, column=2, sticky="ns")
        self._build_analysis(right)

        # status bar
        self.status = ttk.Label(self, text="Open a recording (wav/npy/npz) or generate a synthetic one",
                                anchor="w", padding=(8, 2))
        self.status.grid(row=1, column=0, columnspan=3, sticky="ew")

    def _build_controls(self, parent: ttk.Frame) -> None:
        def section(title: str) -> ttk.LabelFrame:
            f = ttk.LabelFrame(parent, text=title, padding=6)
            f.pack(fill="x", pady=4)
            return f

        f = section("Recording")
        ttk.Button(f, text="Open...", command=self.open_file).pack(fill="x")
        ttk.Button(f, text="Save selection as wav...", command=self.save_selection).pack(fill="x", pady=2)
        self.var_demo = tk.BooleanVar(value=False)
        ttk.Button(f, text="Generate synthetic FSK demo", command=self.load_demo).pack(fill="x")

        f = section("Analysis band")
        row = ttk.Frame(f); row.pack(fill="x")
        ttk.Label(row, text="lo Hz").pack(side="left")
        self.var_flo = tk.StringVar(value="0")
        ttk.Entry(row, textvariable=self.var_flo, width=8).pack(side="left", padx=2)
        ttk.Label(row, text="hi Hz").pack(side="left")
        self.var_fhi = tk.StringVar(value="8000")
        ttk.Entry(row, textvariable=self.var_fhi, width=8).pack(side="left", padx=2)
        ttk.Button(f, text="Apply / refresh views", command=self.refresh_views).pack(fill="x", pady=2)

        f = section("Spectrum")
        row = ttk.Frame(f); row.pack(fill="x")
        ttk.Label(row, text="FFT").pack(side="left")
        self.var_nfft = tk.StringVar(value="1024")
        ttk.Combobox(row, textvariable=self.var_nfft, width=6, state="readonly",
                     values=[str(1 << k) for k in range(6, 14)]).pack(side="left", padx=2)
        ttk.Label(row, text="Window").pack(side="left")
        self.var_window = tk.StringVar(value=window_label(WindowType.KAISER_B8))
        ttk.Combobox(row, textvariable=self.var_window, width=10, state="readonly",
                     values=[window_label(w) for w in WindowType]).pack(side="left", padx=2)
        ttk.Label(f, text="Slide").pack(anchor="w")
        self.var_slide = tk.StringVar(value="Size/2")
        ttk.Combobox(f, textvariable=self.var_slide, width=8, state="readonly",
                     values=["Size", "Size/2", "Size/4", "Size/8", "Size/16"]).pack(fill="x")
        ttk.Label(f, text="Dynamic range dB").pack(anchor="w")
        self.var_dyn = tk.StringVar(value="60")
        ttk.Scale(f, from_=20, to=100, variable=tk.DoubleVar())
        scale = ttk.Scale(f, from_=20, to=100, command=self._on_dyn)
        scale.set(60)
        scale.pack(fill="x")

        f = section("Detection")
        ttk.Button(f, text="Find carriers", command=self.find_carriers).pack(fill="x")
        ttk.Button(f, text="Classify selection", command=self.classify_selection).pack(fill="x", pady=2)
        ttk.Button(f, text="Band scan", command=self.band_scan).pack(fill="x")
        ttk.Button(f, text="Detect waveforms", command=self.detect_waveforms).pack(fill="x", pady=2)
        ttk.Button(f, text="Decode sweep (selection bits)", command=self.decode_sweep).pack(fill="x", pady=2)
        ttk.Button(f, text="Validate dataset dir...", command=self.validate_dataset).pack(fill="x")
        ttk.Button(f, text="Generate test dataset...", command=self.generate_dataset).pack(fill="x", pady=2)

        f = section("Demodulation")
        ttk.Label(f, text="FSK mark Hz").pack(anchor="w")
        self.var_mark = tk.StringVar(value="2125")
        ttk.Entry(f, textvariable=self.var_mark, width=8).pack(fill="x")
        ttk.Label(f, text="space Hz").pack(anchor="w")
        self.var_space = tk.StringVar(value="1275")
        ttk.Entry(f, textvariable=self.var_space, width=8).pack(fill="x")
        ttk.Label(f, text="baud Hz").pack(anchor="w")
        self.var_baud = tk.StringVar(value="50")
        ttk.Entry(f, textvariable=self.var_baud, width=8).pack(fill="x")
        self.var_auto_baud = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="measure baud (4-stage)", variable=self.var_auto_baud).pack(anchor="w")
        ttk.Button(f, text="FSK demod selection", command=self.fsk_demod_selection).pack(fill="x", pady=4)

    def _build_analysis(self, parent: ttk.Frame) -> None:
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        # Info tab
        f_info = ttk.Frame(nb, padding=4); nb.add(f_info, text="Info")
        self.txt_info = tk.Text(f_info, width=44, height=20, wrap="word")
        self.txt_info.pack(fill="both", expand=True)

        # Carriers / scan tab
        f_car = ttk.Frame(nb, padding=4); nb.add(f_car, text="Detections")
        cols = ("freq", "bw", "snr", "class", "conf")
        self.tree = ttk.Treeview(f_car, columns=cols, show="headings", height=12)
        for cid, head, w in (("freq", "Hz", 80), ("bw", "BW Hz", 70),
                             ("snr", "SNR dB", 60), ("class", "Class", 70),
                             ("conf", "Conf", 50)):
            self.tree.heading(cid, text=head)
            self.tree.column(cid, width=w, anchor="e")
        self.tree.pack(fill="both", expand=True)

        # Decoder tab
        f_dec = ttk.Frame(nb, padding=4); nb.add(f_dec, text="Decoders")
        self.txt_dec = tk.Text(f_dec, width=44, height=20, wrap="word")
        self.txt_dec.pack(fill="both", expand=True)

        # IQ tab
        f_iq = ttk.Frame(nb, padding=4); nb.add(f_iq, text="I/Q")
        self.const_canvas = tk.Canvas(f_iq, width=200, height=200, bg="black",
                                      highlightthickness=0)
        self.const_canvas.pack()
        self.const_view = PointsView(self.const_canvas, 200)
        self.phase_canvas = tk.Canvas(f_iq, width=200, height=200, bg="black",
                                      highlightthickness=0)
        self.phase_canvas.pack(pady=4)
        self.phase_view = PointsView(self.phase_canvas, 200)

        # Dataset tab
        f_ds = ttk.Frame(nb, padding=4); nb.add(f_ds, text="Dataset")
        self.txt_ds = tk.Text(f_ds, width=44, height=20, wrap="word")
        self.txt_ds.pack(fill="both", expand=True)

    # ------------------------------------------------------------ helpers

    def say(self, msg: str) -> None:
        self.status.config(text=msg)
        self.update_idletasks()

    def _window(self) -> WindowType:
        for w in WindowType:
            if window_label(w) == self.var_window.get():
                return w
        return WindowType.KAISER_B8

    def _slide(self) -> SlideStep:
        return {"Size": SlideStep.FULL, "Size/2": SlideStep.HALF,
                "Size/4": SlideStep.QUARTER, "Size/8": SlideStep.EIGHTH,
                "Size/16": SlideStep.SIXTEENTH}.get(self.var_slide.get(), SlideStep.HALF)

    def _fband(self) -> tuple[float, float]:
        nyq = self.sample_rate / 2
        try:
            lo = max(0.0, float(self.var_flo.get()))
            hi = min(nyq, float(self.var_fhi.get()))
        except ValueError:
            lo, hi = 0.0, nyq
        return (lo, hi) if hi > lo else (0.0, nyq)

    def _tband(self) -> tuple[float, float]:
        if self.sel_time:
            return self.sel_time
        dur = len(self.samples) / self.sample_rate
        return (0.0, dur)

    def _selection(self) -> np.ndarray:
        t0, t1 = self._tband()
        i0 = int(t0 * self.sample_rate)
        i1 = min(len(self.samples), int(t1 * self.sample_rate))
        return self.samples[i0:i1]

    def _on_dyn(self, value: str) -> None:
        self.var_dyn.set(f"{float(value):.0f}")
        if len(self.samples):
            self.refresh_views()

    # ------------------------------------------------------------ loading

    def open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open recording", filetypes=[
                ("Recordings", "*.wav *.npy *.npz *.raw *.iq"),
                ("All files", "*.*")])
        if not path:
            return
        try:
            self.samples, self.sample_rate = load_recording(path)
        except Exception as exc:
            messagebox.showerror("Open failed", str(exc))
            return
        self.path = path
        self.var_fhi.set(f"{self.sample_rate / 2:.0f}")
        self.sel_time = None
        self.say(f"{path}: {len(self.samples)} samples @ {self.sample_rate:.0f} Hz "
                 f"({len(self.samples) / self.sample_rate:.2f} s)")
        self.refresh_views()

    def load_demo(self) -> None:
        """Synthetic FSK demo so the workbench works with zero setup."""
        spec = ds.SignalSpec("FSK", 16000, 3.0, mark_hz=2125, space_hz=1275,
                             baud=50, snr_db=18, seed=42)
        rec = ds.generate_signal(spec)
        self.samples, self.sample_rate = rec.samples, spec.sample_rate
        self.path = None
        self.var_mark.set("2125")
        self.var_space.set("1275")
        self.var_baud.set("50")
        self.var_fhi.set("8000")
        self.sel_time = None
        self.say("Synthetic FSK demo loaded (mark 2125 / space 1275, 50 Bd, 18 dB SNR)")
        self.refresh_views()

    def save_selection(self) -> None:
        if not len(self.samples):
            return
        path = filedialog.asksaveasfilename(defaultextension=".wav",
                                            filetypes=[("WAV", "*.wav")])
        if not path:
            return
        save_recording(path, self._selection(), self.sample_rate)
        self.say(f"Saved selection -> {path}")

    # ------------------------------------------------------------- views

    def refresh_views(self) -> None:
        if not len(self.samples):
            return
        try:
            nfft = int(self.var_nfft.get())
        except ValueError:
            nfft = 1024
        flo, fhi = self._fband()
        dur = len(self.samples) / self.sample_rate
        t0, t1 = self._tband()
        i0, i1 = int(t0 * self.sample_rate), int(t1 * self.sample_rate)
        seg = self.samples[i0:i1]
        if len(seg) < nfft:
            self.say("Selection shorter than FFT block; widen time selection")
            return
        # time decimation to fit canvas width
        width = max(self.wf_canvas.winfo_width(), 400)
        try:
            spec = dsp.spectrogram(seg, self.sample_rate, block_size=nfft,
                                   slide=self._slide(), window=self._window(),
                                   fmin=flo, fmax=fhi)
        except ValueError as exc:
            self.say(str(exc))
            return
        db = spec.to_db(dynamic_range_db=float(self.var_dyn.get()))
        n_frames = db.shape[0]
        dec = max(1, n_frames // width)
        self.wf_view.update(db, vmin=-float(self.var_dyn.get()), vmax=0.0,
                            time_decimate=dec)
        # spectrum strip: last frame's (strongest) row
        row = db[int(np.argmax(db.mean(axis=1)))]
        self.spec_view.width = max(self.spec_canvas.winfo_width(), 400)
        self.spec_view.update(row, vmin=-float(self.var_dyn.get()), vmax=0.0)
        self._draw_overlays(spec)

    def _draw_overlays(self, spec: dsp.Spectrogram) -> None:
        """Band-selection rectangle + selection label on both canvases."""
        for canvas, tag in ((self.wf_canvas, "ovl"), (self.spec_canvas, "ovl")):
            canvas.delete(tag)
        if not self.sel_band:
            return
        flo, fhi = self._fband()
        lo, hi = self.sel_band
        width = canvas.winfo_width()
        nfft = spec.block_size
        # map freq -> x within [flo,fhi]
        freqs = spec.freqs if spec.freqs[-1] > flo else np.array([flo, fhi])
        x = lambda f: (f - flo) / max(fhi - flo, 1e-9) * width
        canvas.create_rectangle(x(lo), 0, x(hi), canvas.winfo_height(),
                                outline="#ff7040", tags="ovl")

    # ------------------------------------------------- canvas interaction

    def _canvas_freq(self, canvas, x: float) -> float:
        flo, fhi = self._fband()
        w = canvas.winfo_width()
        return flo + (x / max(w, 1)) * (fhi - flo)

    def _canvas_time(self, x: float) -> float:
        t0, t1 = self._tband()
        w = max(self.wf_canvas.winfo_width(), 1)
        return t0 + (x / w) * (t1 - t0)

    _drag_state: dict = {}

    def _on_spec_click(self, e) -> None:
        self._drag_state["spec"] = self._canvas_freq(self.spec_canvas, e.x)
        self.sel_band = None

    def _on_spec_drag(self, e) -> None:
        f0 = self._drag_state.get("spec")
        if f0 is None:
            return
        f1 = self._canvas_freq(self.spec_canvas, e.x)
        self.sel_band = (min(f0, f1), max(f0, f1))
        self.refresh_views()

    def _on_spec_release(self, e) -> None:
        if self.sel_band:
            lo, hi = self.sel_band
            self.say(f"Band selected: {lo:.0f} .. {hi:.0f} Hz "
                     f"({hi - lo:.0f} Hz wide)")

    def _on_wf_click(self, e) -> None:
        self._drag_state["wf"] = self._canvas_time(e.x)

    def _on_wf_drag(self, e) -> None:
        t0 = self._drag_state.get("wf")
        if t0 is None:
            return
        t1 = self._canvas_time(e.x)
        self.sel_time = (max(0.0, min(t0, t1)), min(len(self.samples) / self.sample_rate,
                                                    max(t0, t1)))
        self.refresh_views()

    def _on_wf_release(self, e) -> None:
        if self.sel_time:
            a, b = self.sel_time
            self.say(f"Time selected: {a:.2f} .. {b:.2f} s ({b - a:.2f} s)")

    # ---------------------------------------------------------- detection

    def find_carriers(self) -> None:
        if not len(self.samples):
            return
        self.say("Finding carriers...")
        self._find_carriers_job()

    def _find_carriers_job(self) -> None:
        seg = self._selection()
        flo, _ = self._fband()
        try:
            scan = det.find_carriers(seg, self.sample_rate, fmin=flo or None)
        except Exception as exc:
            messagebox.showerror("Detection", str(exc)); return
        for row in self.tree.get_children():
            self.tree.delete(row)
        for c in scan.carriers:
            self.tree.insert("", "end", values=(
                f"{c.freq_hz:.0f}", f"{c.bandwidth_hz:.0f}", f"{c.prominence_db:.1f}",
                "carrier", f"{c.frames_present}/{c.frames_total}"))
        self.txt_info.delete("1.0", "end")
        self.txt_info.insert("end", f"Carrier scan of {len(seg)} samples\n"
                             f"noise floor {scan.noise_floor_db:.1f} dB\n"
                             f"{len(scan.carriers)} carrier(s):\n\n")
        for c in scan.carriers:
            self.txt_info.insert("end",
                                 f"  {c.freq_hz:9.1f} Hz  prom {c.prominence_db:5.1f} dB  "
                                 f"bw ~{c.bandwidth_hz:6.0f} Hz  "
                                 f"persist {c.frames_present}/{c.frames_total}\n")
        self.say(f"{len(scan.carriers)} carrier(s) found")

    def classify_selection(self) -> None:
        if not len(self.samples):
            return
        seg = self._selection()
        flo, fhi = self.sel_band or self._fband()
        cls = det.classify_modulation(seg, self.sample_rate, fmin=flo or None,
                                      fmax=fhi if self.sel_band else None)
        snr = det.snr_estimate(seg, self.sample_rate, flo + 1, fhi - 1) if self.sel_band else float("nan")
        self.txt_info.delete("1.0", "end")
        self.txt_info.insert("end", "Selection analysis\n==================\n")
        self.txt_info.insert("end", f"time: {self._tband()[0]:.2f}..{self._tband()[1]:.2f} s\n")
        self.txt_info.insert("end", f"band: {flo:.0f}..{fhi:.0f} Hz\n")
        if self.sel_band:
            self.txt_info.insert("end", f"SNR (guard-band): {snr:.1f} dB\n")
        self.txt_info.insert("end", f"class: {cls.label} (confidence {cls.confidence:.2f})\n\nfeatures:\n")
        for k, v in sorted(cls.features.items()):
            self.txt_info.insert("end", f"  {k:18s} {v:8.4f}\n")
        inv = None
        try:
            inv = dsp.inband_inversion(seg, self.sample_rate, flo + 50, fhi - 50)
            self.txt_info.insert("end", f"\nin-band inversion: "
                                 f"{'INVERTED' if inv.inverted else 'normal'} "
                                 f"(centroid {inv.score:.3f})\n")
        except ValueError:
            pass
        self.say(f"Classified: {cls.label}")

    def band_scan(self) -> None:
        if not len(self.samples):
            return
        self.say("Scanning band...")
        self._band_scan_job()

    def detect_waveforms(self) -> None:
        """Run named PHY-level waveform decoders (HFDL) on the
        selection; these identify the exact waveform (mode label) and
        decode frames, unlike the family-level classifier."""
        if not len(self.samples):
            return
        seg = self._selection()
        self.txt_info.delete("1.0", "end")
        self.txt_info.insert("end", "Waveform detection\n"
                             "==================\n")
        found_any = False
        try:
            from .. import hfdl as hfdl_mod
            from scipy import signal as _sp
            # audio -> analytic baseband at the measured subcarrier
            # (same conversion decode_file uses)
            f, P = _sp.welch(seg, self.sample_rate, nperseg=1024)
            tone = f[int(np.argmax(P))]
            if not (800 < tone < 2600):
                tone = hfdl_mod.SSB_CARRIER_OFFSET_HZ
            a = _sp.hilbert(seg)
            bb = a * np.exp(-2j * np.pi * tone
                            * np.arange(len(seg)) / self.sample_rate)
            frames = hfdl_mod.decode_frame(bb, self.sample_rate)
            frames = [frames] if frames else []
        except Exception:
            frames = []
        if frames:
            found_any = True
            for fr in frames:
                self.txt_info.insert(
                    "end", f"HFDL: {fr.label}  "
                    f"{'CRC OK' if fr.crc_ok else 'CRC bad'}  "
                    f"{fr.symbols} sym, {len(fr.bytes_out)} bytes\n")
                if fr.bytes_out:
                    head = fr.bytes_out[:120].decode("latin-1",
                                                     errors="replace")
                    self.txt_info.insert("end", f"  data: {head!r}\n")
        if not found_any:
            self.txt_info.insert(
                "end", "no named waveform matched the selection\n"
                "(family classification still applies; the decoder "
                "sweep names bitstream formats)\n")
            self.say("No named waveform matched")
        else:
            self.say(f"Detected: HFDL {frames[0].label}")

    def _band_scan_job(self) -> None:
        flo, fhi = self._fband()
        try:
            hits = det.band_scan(self._selection(), self.sample_rate,
                                 fmin=flo or None, fmax=fhi)
        except Exception as exc:
            messagebox.showerror("Scan", str(exc)); return
        for row in self.tree.get_children():
            self.tree.delete(row)
        for h in hits:
            self.tree.insert("", "end", values=(
                f"{h.center_hz:.0f}", f"{h.bandwidth_hz:.0f}", f"{h.snr_db:.1f}",
                h.classification.label, f"{h.classification.confidence:.2f}"))
        self.say(f"{len(hits)} occupied channel(s)")

    # -------------------------------------------------------- demod/decode

    def fsk_demod_selection(self) -> None:
        seg = self._selection()
        if not len(seg):
            return
        try:
            mark = float(self.var_mark.get()); space = float(self.var_space.get())
        except ValueError:
            messagebox.showerror("FSK", "mark/space must be numbers"); return
        baud = None
        if self.var_auto_baud.get():
            try:
                ev = dsp.baud_rate_stages(seg, self.sample_rate)
                baud = ev.rate_hz
                self.var_baud.set(f"{baud:.1f}")
                self.say(f"Measured baud: {baud:.1f} Hz (corr {ev.correlation:.2f})")
            except ValueError as exc:
                messagebox.showerror("FSK", f"baud measurement: {exc}"); return
        else:
            try:
                baud = float(self.var_baud.get())
            except ValueError:
                messagebox.showerror("FSK", "baud must be a number"); return
        try:
            dem = dsp.fsk_demod_sync(seg, self.sample_rate, mark, space, baud)
        except ValueError as exc:
            messagebox.showerror("FSK", str(exc)); return
        bits = dem.bits
        ones = float(bits.mean())
        self.txt_dec.delete("1.0", "end")
        self.txt_dec.insert("end",
                            f"FSK demod (synced): {len(bits)} symbols @ "
                            f"{dem.symbol_rate_hz:.2f} Bd (refined from {baud:.1f})\n"
                            f"mark {mark:.0f} / space {space:.0f} Hz\n"
                            f"inverted: {dem.inverted}   ones fraction: {ones:.3f}\n\n")
        self.txt_dec.insert("end", "bits: " + "".join(map(str, bits[:512])) + "...\n")
        self._last_bits = bits
        self.say(f"FSK demodulated {len(bits)} bits")

    def decode_sweep(self) -> None:
        if not len(self.samples):
            return
        self.say("Decoder sweep...")
        self._decode_job()

    def _decode_job(self) -> None:
        bits = getattr(self, "_last_bits", None)
        if bits is None or not len(bits):
            self.say("Run FSK demod first (sweep needs a bitstream)")
            return
        if not sd.bee_available():
            self.txt_dec.insert("end", f"\nbee decoders unavailable: {sd.bee_unavailable_reason()}\n")
            return
        outcomes = sd.run_all(bits)
        self.txt_dec.insert("end", f"\nDecoder sweep over {len(sd.available_decoders())} "
                            f"decoders, {len(outcomes)} hits:\n\n")
        for o in outcomes:
            self.txt_dec.insert("end", f"== {o.decoder}: {o.symbol_count} symbols, "
                                f"{o.error_count} errors\n{o.text[:2000]}\n\n")
        self.say(f"Sweep done: {len(outcomes)} decoder hit(s)")

    # ------------------------------------------------------------- dataset

    def validate_dataset(self) -> None:
        path = filedialog.askdirectory(title="Dataset directory (with truth.json)")
        if not path:
            return
        self.say("Validating dataset...")
        self.after(50, lambda: self._validate_job(path))

    def _validate_job(self, path: str) -> None:
        report = ds.validate_dataset(path, recheck_detection=False)
        self.txt_ds.delete("1.0", "end")
        self.txt_ds.insert("end", f"Dataset: {report.root}\n"
                            f"records checked: {report.records_checked}\n"
                            f"result: {'OK' if report.ok else 'PROBLEMS FOUND'}\n\n")
        for i in report.issues:
            self.txt_ds.insert("end", f"[{i.severity}] {i.file}: {i.message}\n")
        self.say(f"Dataset {'OK' if report.ok else 'has issues'} "
                 f"({report.records_checked} records)")

    def generate_dataset(self) -> None:
        path = filedialog.askdirectory(title="Where to write the test dataset")
        if not path:
            return
        self.say("Generating dataset...")
        self.after(50, lambda: self._generate_job(path))

    def _generate_job(self, path: str) -> None:
        specs = [
            ds.SignalSpec("FSK", 16000, 1.0, snr_db=30, seed=1),
            ds.SignalSpec("FSK", 16000, 1.0, snr_db=8, seed=2),
            ds.SignalSpec("PSK", 16000, 1.0, snr_db=30, seed=3),
            ds.SignalSpec("OOK", 16000, 1.0, snr_db=25, seed=4),
            ds.SignalSpec("OFDM", 16000, 1.0, snr_db=30, seed=5),
            ds.SignalSpec("tone", 16000, 1.0, freq_hz=1800, snr_db=35, seed=6),
            ds.SignalSpec("noise", 16000, 1.0, snr_db=90, seed=7),
        ]
        root = ds.write_dataset(path, specs)
        self._validate_job(str(root))

    # ------------------------------------------------------------- overlays


def main() -> None:
    app = Workbench()
    app.mainloop()


if __name__ == "__main__":
    main()
