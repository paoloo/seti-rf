"""Build and analyze a mixed ATA power / GPS IQ CyberEther replay."""

import argparse
from collections import Counter
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import struct
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d

from seti_rf.signal import rfi
from seti_rf.triage import SigprocFilterbank

BATCH = 8192
HEADER = struct.Struct("<8sQ")
MAGIC = b"SETIRF01"
GPS_RATE = 4_000_000
GPS_CENTER_HZ = 1_575_420_000
TIME_AVERAGE = 16
FRAME_HEIGHT = 560
FRAME_WIDTH = 960


def prepare_ata(replay_path, background_path, output):
    """Restore original ATA power where the older GPS power profile was added."""
    with SigprocFilterbank(replay_path) as replay, SigprocFilterbank(background_path) as background:
        assert all(replay.attrs[key] == background.attrs[key] for key in ("nchans", "fch1", "foff", "tsamp"))
        rows, _, channels = replay.shape
        assert rows == 156250
        raw = np.memmap(replay_path, dtype=replay.dtype, mode="r",
                        offset=replay._offset, shape=(rows, channels))
        original = np.memmap(background_path, dtype=background.dtype, mode="r",
                             offset=background._offset, shape=(background.shape[0], channels))
        complete_rows = rows // TIME_AVERAGE * TIME_AVERAGE
        restored = raw[:complete_rows].copy()
        restored[:7812] = original[:7812]
        power = np.rint(restored.reshape(-1, TIME_AVERAGE, channels).mean(axis=1)).astype("<u2")
        rf = replay.attrs["fch1"] + np.arange(channels) * replay.attrs["foff"]
        np.savez_compressed(output, power=power, rf=rf, dt=replay.attrs["tsamp"] * TIME_AVERAGE)


def pack(ata_path, gps_path, output):
    raw = np.fromfile(gps_path, dtype="<i2", count=GPS_RATE // 100 * 2)
    if raw.size != GPS_RATE // 100 * 2:
        raise ValueError("GPS input must contain at least 10 ms of ci16 IQ")
    with np.load(ata_path, allow_pickle=False) as ata:
        payload = io.BytesIO()
        np.savez_compressed(payload, **ata, gps=raw)
    data = payload.getvalue()
    with output.open("wb") as file:
        file.write(HEADER.pack(MAGIC, len(data)))
        file.write(data)
        file.write(bytes(-(HEADER.size + len(data)) % (8 * BATCH)))


def detect_gps(raw):
    iq = raw.astype("f4").view("c8")
    epoch = GPS_RATE // 1000
    hits = {}
    for start in range(0, len(iq), epoch):
        for hit in rfi.gps.acquire(iq[start:start + epoch], GPS_RATE, doppler_span_hz=6000):
            hits.setdefault(hit.prn, []).append(hit)
    satellites = []
    for prn, candidates in sorted(hits.items()):
        bins = Counter(hit.doppler_hz for hit in candidates)
        doppler, count = bins.most_common(1)[0]
        if count >= 3:
            best = max((hit for hit in candidates if hit.doppler_hz == doppler),
                       key=lambda hit: hit.peak_to_secondary)
            satellites.append({**asdict(best), "matching_epochs": count})
    if satellites:
        return {"kind": "gps_l1_ca", "classification": "human_made",
                "time_s": 0.0, "center_hz": GPS_CENTER_HZ, "satellites": satellites}


def detect_carrier(power, rf, dt):
    baseline = np.median(power, axis=1)
    channels = power.argmax(axis=1)
    peak = power[np.arange(len(power)), channels]
    rows = np.flatnonzero(peak >= baseline * 10**(8 / 10))
    if len(rows) >= 20 and np.unique(channels[rows]).size <= 8:
        frequencies = rf[channels[rows]]
        return {"kind": "pulsed_drifting_carrier", "classification": "human_made",
                "time_s": float(rows[0] * dt), "last_time_s": float(rows[-1] * dt),
                "first_rf_mhz": float(frequencies[0]), "last_rf_mhz": float(frequencies[-1]),
                "flagged_integrations": int(rows.size)}


def detect_pulse(power, rf, dt, dm):
    delays = np.rint(4148.808 * dm * (rf**-2 - rf.max()**-2) / dt).astype(int)
    common = len(power) - delays.max()
    width = max(1, round(0.003 / dt))
    if common < width:
        raise ValueError("ATA span is shorter than the dispersion delay and pulse width")
    baseline = np.median(power, axis=0)
    good = baseline > 0
    residual = power[:, good] / baseline[good] - 1
    residual[residual >= 3] = 0
    residual -= np.median(residual, axis=1, keepdims=True)
    window = max(3, round(0.042 / dt) | 1)
    residual -= uniform_filter1d(residual, window, axis=0)
    profile = np.zeros(common)
    for channel, delay in enumerate(delays[good]):
        profile += residual[delay:delay + common, channel]
    sums = np.convolve(profile, np.ones(width), mode="valid")
    median = np.median(sums)
    scale = 1.4826 * np.median(np.abs(sums - median))
    peak = int(sums.argmax())
    if scale > 0 and (sums[peak] - median) / scale >= 6:
        return {"kind": "dispersed_pulse", "classification": "unknown",
                "time_s": peak * dt, "dm_pc_cm3": dm,
                "robust_snr": float((sums[peak] - median) / scale)}


def render(power, rf, dt, events, path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    figure = Figure(figsize=(FRAME_WIDTH / 100, FRAME_HEIGHT / 100), dpi=100)
    canvas = FigureCanvasAgg(figure)
    layout = figure.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=.58, wspace=.29)
    overview = figure.add_subplot(layout[0, :])
    zoom = figure.add_subplot(layout[1, 0])
    gps = figure.add_subplot(layout[1, 1])
    figure.subplots_adjust(left=.09, right=.98, bottom=.11, top=.89)
    figure.suptitle("ATA + GPS replay | Detected events", fontsize=17, fontweight="bold")
    baseline = np.maximum(np.median(power, axis=0), 1)
    normalized = 10 * np.log10(np.maximum(power, 1) / baseline)

    def waterfall(axis, start, stop, low, high, vmax):
        rows = slice(round(start / dt), round(stop / dt))
        band = (rf >= low) & (rf <= high)
        axis.imshow(normalized[rows, :][:, band].T, aspect="auto", cmap="magma",
                    origin="upper", interpolation="nearest", vmin=-.6, vmax=vmax,
                    extent=(start, stop, rf[band].min(), rf[band].max()))
        axis.set(xlabel="Replay time (s)", ylabel="RF (MHz)")

    waterfall(overview, 0, len(power) * dt, rf.min(), rf.max(), 3)
    by_kind = {event["kind"]: event for event in events}
    pulse = by_kind.get("dispersed_pulse")
    if pulse:
        start = pulse["time_s"]
        delay = 4148.808 * pulse["dm_pc_cm3"] * (rf**-2 - rf.max()**-2)
        stop = start + delay.max()
        overview.add_patch(Rectangle((start - .04, rf.min() + 2), stop - start + .08,
                                     np.ptp(rf) - 4, fill=False, color="cyan", linewidth=1.5))
        overview.text(start - .08, rf.max() - 35, "FRB candidate", color="cyan",
                      ha="right", fontsize=9, fontweight="bold")
        waterfall(zoom, start - .05, stop + .05, rf.min(), rf.max(), 1.2)
        zoom.plot(start + delay, rf, color="cyan", linestyle="--", linewidth=1.25)
        zoom.set_title(f"Dispersed pulse | DM {pulse['dm_pc_cm3']:.2f} | unknown")
    else:
        zoom.set_title("No dispersed pulse detected")
    carrier = by_kind.get("pulsed_drifting_carrier")
    if carrier:
        start = carrier["time_s"]
        stop = carrier["last_time_s"]
        rf1 = min(carrier["first_rf_mhz"], carrier["last_rf_mhz"])
        rf2 = max(carrier["first_rf_mhz"], carrier["last_rf_mhz"])
        overview.add_patch(Rectangle((start - .05, rf1 - 12), stop - start + .1,
                                     rf2 - rf1 + 24, fill=False, color="lime", linewidth=1.5))
        overview.annotate("Drifting carrier", xy=(start, rf2 + 13), xytext=(start - .18, rf2 + 80),
                          color="lime", ha="right", fontsize=9, fontweight="bold",
                          arrowprops={"arrowstyle": "->", "color": "lime"})
    acquired = by_kind.get("gps_l1_ca")
    gps.set_title("GPS L1 C/A | recorded IQ at replay start")
    gps.set(xlabel="Satellite PRN", ylabel="Peak / secondary")
    if acquired:
        satellites = acquired["satellites"]
        names = [str(item["prn"]) for item in satellites]
        gps.bar(names, [item["peak_to_secondary"] for item in satellites], color="#2868ae")
        gps.axhline(4, color="#404854", linewidth=1, linestyle="--", label="Acquisition threshold")
        gps.legend(loc="upper right", fontsize=8)
    canvas.draw()
    pixels = np.ascontiguousarray(np.asarray(canvas.buffer_rgba())[:, :, :3], dtype="<f4")
    pixels /= 255
    pixels.tofile(path)


def analyze(stream, dm, frame=None):
    magic, size = HEADER.unpack(stream.read(HEADER.size))
    if magic != MAGIC or not 0 < size <= 100_000_000:
        raise ValueError("invalid replay frame")
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("incomplete replay frame")
    with np.load(io.BytesIO(data), allow_pickle=False) as replay:
        power, rf, dt = replay["power"], replay["rf"], float(replay["dt"])
        events = [detect_gps(replay["gps"]), detect_carrier(power, rf, dt),
                  detect_pulse(power, rf, dt, dm)]
    events = sorted((event for event in events if event is not None), key=lambda event: event["time_s"])
    if frame is not None:
        render(power, rf, dt, events, frame)
    return events


def graph(replay_path, result_path, python, dm):
    frame_path = result_path.resolve().with_suffix(".rgb")
    worker = [python, str(Path(__file__).resolve()), "detect", "--dm", str(dm),
              "--frame", str(frame_path)]
    pythonpath = os.environ.get("PYTHONPATH")
    return f"""version: 2
title: ATA and GPS replay
graph:
  - name: replay
    module: file_reader
    device: cpu
    runtime: native
    provider: generic
    config:
      filepath: {json.dumps(str(replay_path.resolve()))}
      fileFormat: raw
      dataType: CF32
      batchSize: {BATCH}
      loop: false
      playing: true
  - name: detect
    module: python
    device: cpu
    runtime: python
    provider: generic
    config:
      code: |-
        import json
        import os
        import subprocess
        from pathlib import Path

        result = Path({str(result_path.resolve())!r})
        frame = Path({str(frame_path)!r})
        remaining = {replay_path.stat().st_size // (8 * BATCH)}
        worker = None

        def compute(ctx):
            global worker, remaining
            data = ctx.inputs[0]
            if remaining == 0:
                return SKIP
            if worker is None:
                env = os.environ.copy()
                env["MPLCONFIGDIR"] = str(result.parent / ".matplotlib")
                pythonpath = {pythonpath!r}
                if pythonpath:
                    env["PYTHONPATH"] = pythonpath
                worker = subprocess.Popen({worker!r}, stdin=subprocess.PIPE,
                                          stdout=subprocess.PIPE, env=env)
            # The final transport batch may include padding after the frame.
            try:
                worker.stdin.write(memoryview(data).cast("B"))
            except BrokenPipeError:
                if remaining != 1:
                    raise
            remaining -= 1
            if remaining == 0:
                worker.stdin.close()
                report = worker.stdout.read()
                if worker.wait() != 0:
                    raise RuntimeError("SETI replay worker failed")
                result.write_bytes(report)
                pixels = frame.read_bytes()
                memoryview(ctx.outputs[0]).cast("B")[:] = pixels
                events = json.loads(report)
                ctx.env["replay"] = {{"count": len(events), "events": events}}
                print(report.decode(), flush=True)
                return
            return SKIP

        def cleanup():
            if worker is not None and worker.poll() is None:
                worker.terminate()
      inputCount: 1
      outputCount: 1
      outputTensorSpecs:
        - shape: '[{FRAME_HEIGHT}, {FRAME_WIDTH}, 3]'
          axes: '[B, S, C]'
          dtype: F32
          device: cpu
      throttled: false
    input:
      input0: '${{graph.replay.output.signal}}'
    meta:
      node:
        x: 390
        y: 70
        width: 310
  - name: detections
    module: frame
    device: cpu
    runtime: native
    provider: generic
    config:
      autoRange: false
    input:
      frame: '${{graph.detect.output.output0}}'
    meta:
      node:
        x: 740
        y: 70
        width: 960
      surface_default:
        attachedHeight: 560
        detachedWidth: 960
        detachedHeight: 560
  - name: report
    module: note
    device: cpu
    runtime: native
    provider: generic
    config:
      content: |-
        ## Python detection report
        Events: **${{env.replay.count}}**

        | Detection | Classification | Replay time (s) |
        |---|---|---:|
        | ${{env.replay.events[*].kind}} | ${{env.replay.events[*].classification}} | ${{env.replay.events[*].time_s:.3f}} |
    meta:
      node:
        x: 740
        y: 740
        width: 660
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, paths in {"prepare-ata": ("replay", "background", "output"),
                        "pack": ("ata", "gps", "output"),
                        "graph": ("replay", "result", "output")}.items():
        command = commands.add_parser(name)
        for path in paths:
            command.add_argument(path, type=Path)
        if name == "graph":
            command.add_argument("--python", default=sys.executable)
            command.add_argument("--dm", type=float, default=219.81)
    detector = commands.add_parser("detect")
    detector.add_argument("--dm", type=float, default=219.81)
    detector.add_argument("--frame", type=Path)
    args = parser.parse_args()
    if args.command == "prepare-ata":
        prepare_ata(args.replay, args.background, args.output)
    elif args.command == "pack":
        pack(args.ata, args.gps, args.output)
    elif args.command == "graph":
        args.output.write_text(graph(args.replay, args.result, args.python, args.dm))
    else:
        print(json.dumps(analyze(sys.stdin.buffer, args.dm, args.frame), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
