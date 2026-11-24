"""Replay recorded GPS IQ and ATA power through a paced CyberEther detector."""
import argparse
import inspect
import io
import json
from pathlib import Path
import struct
from textwrap import indent

import numpy as np
from scipy.ndimage import uniform_filter1d

from cyberether_replay import GPS_RATE, HEADER, MAGIC, detect_gps

ROWS = 100
CHANNELS = 1344
HEIGHT = 200
WIDTH = CHANNELS // 2
GPS_BYTES = GPS_RATE // 100 * 4
PACKET = struct.Struct("<IIdd")
PACKET_BYTES = PACKET.size + ROWS * CHANNELS * 2 + GPS_BYTES
BATCH = PACKET_BYTES // 8
DM = 219.81


def build(source, output):
    with source.open("rb") as file:
        magic, size = HEADER.unpack(file.read(HEADER.size))
        if magic != MAGIC:
            raise ValueError("expected a SETI replay frame")
        payload = file.read(size)
    with np.load(io.BytesIO(payload), allow_pickle=False) as replay, output.open("wb") as file:
        power, rf, dt = replay["power"], replay["rf"], float(replay["dt"])
        if power.shape[1] != CHANNELS or len(replay["gps"].tobytes()) != GPS_BYTES:
            raise ValueError("unexpected ATA channel count or GPS extent")
        for start in range(0, len(power), ROWS):
            current = power[start:start + ROWS]
            file.write(PACKET.pack(start, len(current), dt, rf[0]))
            file.write(current.astype("<u2", copy=False).tobytes())
            file.write(bytes((ROWS - len(current)) * CHANNELS * 2))
            file.write(replay["gps"].tobytes() if start == 0 else bytes(GPS_BYTES))


class Detector:
    def __init__(self, pool):
        self.pool = pool
        self.gps = None
        self.image = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        self.events = []
        self.raw = np.empty((0, CHANNELS), dtype="<u2")
        self.history = np.empty((0, CHANNELS), dtype=np.float32)
        self.scores = np.empty(0)
        self.profile_start = 0
        self.baseline = None

    def feed(self, packet):
        start, count, dt, first_rf = PACKET.unpack_from(packet)
        rf = first_rf - .5 * np.arange(CHANNELS)
        data = np.frombuffer(packet, dtype="<u2", count=count * CHANNELS,
                             offset=PACKET.size).reshape(count, CHANNELS)
        if start == 0:
            iq = np.frombuffer(packet, dtype="<i2", offset=PACKET.size + ROWS * CHANNELS * 2)
            self.gps = self.pool.submit(detect_gps, iq.copy())
        if self.gps is not None and self.gps.done():
            event = self.gps.result()
            if event:
                self.events.append(event)
            self.gps = None

        # One waterfall line per 10.24 ms of input, with newest lines at the bottom.
        rows = len(data) // 10
        if rows:
            power = data[:rows * 10].reshape(rows, 10, WIDTH, 2).max(axis=(1, 3))
            reference = np.maximum(np.median(power, axis=1, keepdims=True), 1)
            db = 10 * np.log10(np.maximum(power, 1) / reference)
            self.image[:-rows] = self.image[rows:]
            self.image[-rows:] = np.clip((db + .8) / 4, 0, 1)[:, ::-1]

        medians = np.maximum(np.median(data, axis=1), 1)
        peaks = data.argmax(axis=1)
        active = np.flatnonzero(data[np.arange(count), peaks] >= medians * 10**.8)
        if len(active) >= 20 and len(np.unique(peaks[active])) <= 8:
            if not any(event["kind"] == "pulsed_drifting_carrier" for event in self.events):
                self.events.append({"kind": "pulsed_drifting_carrier", "classification": "human_made",
                                    "time_s": (start + int(active[0])) * dt,
                                    "rf_mhz": float(rf[peaks[active[0]]])})

        if self.baseline is None:
            self.raw = np.concatenate((self.raw, data))
            if len(self.raw) < 1000:
                return self.output((start + count) * dt)
            self.baseline = np.maximum(np.median(self.raw, axis=0), 1)
            data = self.raw
            self.raw = np.empty((0, CHANNELS), dtype="<u2")
        residual = (data / self.baseline - 1).astype(np.float32)
        residual[residual >= 3] = 0
        residual -= np.median(residual, axis=1, keepdims=True)
        self.history = np.concatenate((self.history, residual))
        delays = np.rint(4148.808 * DM * (rf**-2 - rf.max()**-2) / dt).astype(int)
        half = 20
        width = 3
        first = half
        stop = len(self.history) - int(delays.max()) - half - width + 1
        if stop > first:
            end = stop + width - 1
            filtered = self.history - uniform_filter1d(self.history, 2 * half + 1, axis=0)
            profile = sum(filtered[first + delay:end + delay, i] for i, delay in enumerate(delays))
            scores = np.convolve(profile, np.ones(width), mode="valid")
            if len(self.scores) >= 1500 and not any(event["kind"] == "dispersed_pulse" for event in self.events):
                reference = self.scores[-4000:]
                median = np.median(reference)
                scale = 1.4826 * np.median(np.abs(reference - median))
                peak = int(scores.argmax())
                if scale > 0 and (scores[peak] - median) / scale >= 6:
                    self.events.append({"kind": "dispersed_pulse", "classification": "unknown",
                                        "time_s": (self.profile_start + first + peak) * dt,
                                        "dm_pc_cm3": DM})
            self.scores = np.concatenate((self.scores, scores))[-4000:]
            trim = stop - half
            self.history = self.history[trim:]
            self.profile_start += trim
        return self.output((start + count) * dt)

    def output(self, seconds):
        value = self.image
        rgb = np.stack((np.clip(value * 1.7 - .25, 0, 1),
                        np.clip(value * 1.6 - .65, 0, 1),
                        np.clip(value * 2.4, 0, .95)), axis=-1)
        return {"time_s": seconds, "events": self.events}, rgb


def compute(ctx):
    global due, detector, published
    now = time.monotonic()
    if due is not None and now < due:
        return YIELD(due - now)
    data = memoryview(ctx.inputs[0]).cast("B")
    start, count, dt, _ = PACKET.unpack_from(data)
    if start == 0:
        detector = Detector(pool)
        published = 0
    due = now + count * dt
    if count < ROWS:
        due += 4
    report, rgb = detector.feed(data)
    ctx.outputs[0][...] = rgb
    events = report["events"]
    ctx.env["live"] = {"time_s": report["time_s"], "count": len(events), "events": events}
    if len(events) != published:
        published = len(events)
        path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)


def cleanup():
    pool.shutdown(wait=False, cancel_futures=True)


def graph(data, result):
    prelude = f"""import json
import struct
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from seti_rf.signal import rfi

ROWS = {ROWS}
CHANNELS = {CHANNELS}
HEIGHT = {HEIGHT}
WIDTH = {WIDTH}
GPS_RATE = {GPS_RATE}
GPS_CENTER_HZ = 1_575_420_000
DM = {DM}
PACKET = struct.Struct({PACKET.format!r})

pool = ThreadPoolExecutor(max_workers=1)
detector = None
due = None
published = 0
path = Path({str(result.resolve())!r})
"""
    code = "\n\n".join([prelude, *(inspect.getsource(item) for item in
                                   (detect_gps, Detector, compute, cleanup))])
    code = indent(code.rstrip(), " " * 8)
    return f"""version: 2
title: ATA and GPS | real-time replay
graph:
  - name: replay
    module: file_reader
    device: cpu
    runtime: native
    provider: generic
    config:
      filepath: {json.dumps(str(data.resolve()))}
      fileFormat: raw
      dataType: CF32
      batchSize: {BATCH}
      loop: true
      playing: true
    meta:
      node:
        x: 20
        y: 80
        width: 250
  - name: detector
    module: python
    device: cpu
    runtime: python
    provider: generic
    config:
      code: |-
{code}
      inputCount: 1
      outputCount: 1
      outputTensorSpecs:
        - shape: '[{HEIGHT}, {WIDTH}, 3]'
          axes: '[B, S, C]'
          dtype: F32
          device: cpu
      throttled: false
    input:
      input0: '${{graph.replay.output.signal}}'
    meta:
      node:
        x: 300
        y: 80
        width: 275
  - name: waterfall
    module: frame
    device: cpu
    runtime: native
    provider: generic
    config:
      autoRange: false
    input:
      frame: '${{graph.detector.output.output0}}'
    meta:
      node:
        x: 610
        y: 80
        width: 850
      surface_default:
        attachedHeight: 350
        detachedWidth: 1000
        detachedHeight: 420
  - name: report
    module: note
    device: cpu
    runtime: native
    provider: generic
    config:
      content: |-
        ## Real-time detection
        Replay time **${{env.live.time_s:.2f}} s** · Events **${{env.live.count}}**

        ATA waterfall · 1064–1735.5 MHz · newest data at bottom

        | Detection | Classification | Time (s) |
        |---|---|---:|
        | ${{env.live.events[*].kind}} | ${{env.live.events[*].classification}} | ${{env.live.events[*].time_s:.3f}} |
    meta:
      node:
        x: 610
        y: 580
        width: 800
        height: 230
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    builder = sub.add_parser("build")
    builder.add_argument("source", type=Path)
    builder.add_argument("output", type=Path)
    generator = sub.add_parser("graph")
    for arg in ("data", "result", "output"):
        generator.add_argument(arg, type=Path)
    args = parser.parse_args()
    if args.command == "build":
        build(args.source, args.output)
    else:
        args.output.write_text(graph(args.data, args.result))


if __name__ == "__main__":
    main()
