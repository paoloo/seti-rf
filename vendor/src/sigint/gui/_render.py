"""Canvas renderers: waterfall, spectrum line, constellation, phase plane.

All draw onto a tk.Canvas with PhotoImage postscript-free pixel writes via
an offscreen numpy -> PPM -> PhotoImage pipeline (fast, no PIL dependency).
"""
from __future__ import annotations

import tkinter as tk

import numpy as np

# turbo-like colormap: dark blue -> cyan -> green -> yellow -> red -> white
_COLORMAP_STOPS = np.array([
    [0, 0, 60], [0, 30, 120], [10, 90, 180], [30, 150, 200],
    [60, 200, 160], [120, 220, 80], [200, 220, 40], [250, 170, 30],
    [240, 90, 20], [230, 30, 30], [255, 210, 180],
], dtype=np.uint8)


def _colormap(n: int = 256) -> np.ndarray:
    xs = np.linspace(0, 1, len(_COLORMAP_STOPS))
    xi = np.linspace(0, 1, n)
    out = np.empty((n, 3), dtype=np.uint8)
    for c in range(3):
        out[:, c] = np.interp(xi, xs, _COLORMAP_STOPS[:, c])
    return out


_CMAP = _colormap()


def db_to_rgb(db: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """[frames,bins] dB image -> [frames,bins,3] uint8 RGB."""
    frac = np.clip((db - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    idx = (frac * (len(_CMAP) - 1)).astype(np.uint8)
    return _CMAP[idx]


def render_ppm(rgb: np.ndarray) -> bytes:
    """[h,w,3] uint8 -> binary PPM bytes (tk PhotoImage-readable)."""
    h, w, _ = rgb.shape
    header = f"P6 {w} {h} 255 ".encode()
    return header + rgb.tobytes()


class WaterfallImage:
    """Holds the current waterfall PhotoImage; sized to the canvas."""

    def __init__(self, canvas: tk.Canvas):
        self.canvas = canvas
        self._photo: tk.PhotoImage | None = None

    def update(self, db: np.ndarray, vmin: float, vmax: float,
               time_decimate: int = 1) -> None:
        """db: [frames,bins]; newest frame at the bottom."""
        rgb = db_to_rgb(db, vmin, vmax)[::time_decimate][::-1]
        h, w, _ = rgb.shape
        self._photo = tk.PhotoImage(data=render_ppm(rgb), width=w, height=h,
                                    master=self.canvas)
        self.canvas.delete("waterfall")
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw",
                                 tags="waterfall")


class SpectrumLine:
    """Instantaneous spectrum: a polyline on the canvas with dB axis."""

    def __init__(self, canvas: tk.Canvas, width: int, height: int):
        self.canvas = canvas
        self.width = width
        self.height = height

    def update(self, db: np.ndarray, vmin: float, vmax: float) -> None:
        c = self.canvas
        c.delete("spectrum")
        n = len(db)
        if n < 2:
            return
        xs = np.linspace(0, self.width, n)
        ys = self.height - np.clip((db - vmin) / (vmax - vmin), 0, 1) * self.height
        pts = np.empty(2 * n)
        pts[0::2] = xs
        pts[1::2] = ys
        c.create_line(*pts.tolist(), fill="#30d060", width=1, tags="spectrum")
        # gridlines every 10 dB
        c.delete("grid")
        level = int(np.ceil(vmin / 10.0) * 10)
        while level < vmax:
            y = self.height - (level - vmin) / (vmax - vmin) * self.height
            c.create_line(0, y, self.width, y, fill="#204020", dash=(2, 4),
                          tags="grid")
            c.create_text(4, y - 6, text=f"{level}", fill="#407040",
                          anchor="w", font=("TkDefaultFont", 7), tags="grid")
            level += 10


class PointsView:
    """Constellation / phase-plane scatter (I/Q)."""

    def __init__(self, canvas: tk.Canvas, size: int = 200):
        self.canvas = canvas
        self.size = size

    def update(self, i: np.ndarray, q: np.ndarray, max_points: int = 4000,
               tag: str = "points") -> None:
        c = self.canvas
        c.delete(tag)
        if len(i) == 0:
            return
        step = max(1, len(i) // max_points)
        i, q = i[::step], q[::step]
        scale = max(np.abs(np.concatenate([i, q])).max(), 1e-9)
        half = self.size / 2 - 4
        cx = cy = self.size / 2
        pts = np.stack([cx + i / scale * half, cy - q / scale * half], axis=1)
        c.create_oval(cx - 2, cy - 2, cx + 2, cy + 2, fill="#604020", outline="",
                      tags=tag)
        c.create_line(0, cy, self.size, cy, fill="#204020", tags=tag)
        c.create_line(cx, 0, cx, self.size, fill="#204020", tags=tag)
        for x, y in pts:
            c.create_rectangle(x - 1, y - 1, x + 1, y + 1, fill="#40c0e0",
                               outline="", tags=tag)
