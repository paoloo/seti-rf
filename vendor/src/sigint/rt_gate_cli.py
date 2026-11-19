#!/usr/bin/env python3
"""rt-gate CLI: stream a recording (or pipe) through the recognition gate.

Simulates real-time by feeding samples in blocks at (optionally) the real
sample rate, printing recognition events as they fire.

Sources:
    python3 rt_gate_cli.py capture.wav                 # wav/npy/npz/raw
    cat capture.wav | python3 rt_gate_cli.py -        # stdin (wav/raw)
    python3 rt_gate_cli.py tests/fixtures/bee_demo/Signals/ASCII.bin --bits --sweep

Options:
    --realtime        sleep to match the sample rate (live simulation)
    --block-ms 100    feed block size
    --demod           enable FSK demodulation while active
    --sweep           decode rolling bits with the BEE decoders on SIGNAL_END
    --json            events as JSON lines
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_SIGINT_PARENT = str(Path(__file__).resolve().parent.parent)
if _SIGINT_PARENT not in sys.path:
    sys.path.insert(0, _SIGINT_PARENT)

from sigint import decoders as sd  # noqa: E402
from sigint import realtime as rt  # noqa: E402
from sigint.gui._io import load_recording  # noqa: E402


def run_stream(x: np.ndarray, sr: float, args: argparse.Namespace) -> None:
    cfg = rt.GateConfig(sample_rate=sr,
                        classify_after_s=args.classify_after,
                        demod=args.demod,
                        mark_hz=args.mark, space_hz=args.space,
                        baud_hz=args.baud)
    gate = rt.Gate(cfg)
    block = max(1, int(sr * args.block_ms / 1000))
    delay = args.block_ms / 1000 if args.realtime else 0.0

    print(f"# {len(x)} samples @ {sr:.0f} Hz, block {block} samples "
          f"({args.block_ms} ms)"
          f"{', realtime pacing' if args.realtime else ''}", flush=True)
    t0 = time.monotonic()
    for i in range(0, len(x), block):
        chunk = x[i:i + block]
        for ev in gate.process(chunk):
            if args.json:
                print(json.dumps(ev.as_dict()), flush=True)
            else:
                print(f"[{ev.t_s:8.3f}] {ev.kind:<18}"
                      + (f" {ev.label}" if ev.label else "")
                      + (f" conf={ev.confidence:.2f}" if ev.confidence is not None else "")
                      + (f" snr={ev.snr_db:.1f}dB" if ev.snr_db is not None else "")
                      + (f" baud={ev.baud_hz:.1f}Hz" if ev.baud_hz is not None else "")
                      + (f" {ev.detail}" if ev.detail else ""), flush=True)
        if delay:
            target = t0 + (i + block) / sr
            pause = target - time.monotonic()
            if pause > 0:
                time.sleep(pause)

    # flush any final events: end_after_s of silence so the VAD closes
    # the burst (SIGNAL_END + the final waveform probe on its audio)
    flush = np.zeros(max(1, int(gate.cfg.sample_rate
                                * gate.cfg.end_after_s) + 1))
    for ev in gate.process(flush):
        if args.json:
            print(json.dumps(ev.as_dict()), flush=True)
        else:
            print(f"[{ev.t_s:8.3f}] {ev.kind}", flush=True)

    if args.sweep and len(gate.bits) >= 64:
        bits = np.asarray(gate.bits)
        print(f"# rolling bit buffer: {len(bits)} bits", flush=True)
        if args.show_bits:
            print("# bits: " + "".join(map(str, bits[:args.show_bits])),
                  flush=True)
        outcomes = sd.run_all(bits, top=args.top_decoders)
        for o in outcomes:
            print(f"# sweep {o.decoder:28s} {o.symbol_count:5d} sym "
                  f"{o.error_count:4d} err  {o.text[:60]}", flush=True)


def run_bitstream(path: str, args: argparse.Namespace) -> None:
    data = Path(path).read_bytes()
    bits = sd.bits_from_bytes(data, bit_order="little")
    print(f"# bitstream mode: {len(data)} bytes = {len(bits)} bits", flush=True)
    outcomes = sd.run_all(bits, top=args.top_decoders)
    for o in outcomes:
        print(f"# sweep {o.decoder:28s} {o.symbol_count:5d} sym "
              f"{o.error_count:4d} err  {o.text[:60]}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="rt-gate", description="Stream a recording/pipe through the "
                                    "real-time recognition gate.")
    ap.add_argument("file", help="wav/npy/npz/raw, or '-' for stdin")
    ap.add_argument("--realtime", action="store_true",
                    help="pace the feed at the sample rate (live sim)")
    ap.add_argument("--block-ms", type=float, default=100.0)
    ap.add_argument("--classify-after", type=float, default=0.5,
                    help="seconds of activity before classification")
    ap.add_argument("--demod", action="store_true")
    ap.add_argument("--mark", type=float, default=2125.0)
    ap.add_argument("--space", type=float, default=1275.0)
    ap.add_argument("--baud", type=float, default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="decode the rolling bit buffer at stream end")
    ap.add_argument("--show-bits", type=int, default=128)
    ap.add_argument("--top-decoders", type=int, default=5)
    ap.add_argument("--bits", action="store_true",
                    help="file is a packed bitstream (BEE .bin)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.bits:
        run_bitstream(args.file, args)
        return
    if args.file == "-":
        raw = sys.stdin.buffer.read()
        # stdin: try wav header, else raw 16-bit at 16 kHz
        import io
        import wave
        try:
            with wave.open(io.BytesIO(raw)) as w:
                sr = float(w.getframerate())
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
                x = pcm.astype(np.float64) / 32768.0
        except Exception:
            x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
            sr = 16000.0
    else:
        x, sr = load_recording(args.file)
    run_stream(x, sr, args)


if __name__ == "__main__":
    main()
