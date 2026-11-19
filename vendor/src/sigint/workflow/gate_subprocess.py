#!/usr/bin/env python3
"""sigint gate worker subprocess.

Runs the recognition gate in a real Python process, isolated from the
CyberEther embedded interpreter (whose C-recursion dispatch budget
makes deep numpy ufunc chains throw RecursionError nondeterministically).

Protocol (stdin/stdout, length-prefixed little-endian u32):
  -> request:  {"cmd": "feed", "data": [floats]}   samples
  <- response: {"events": [...], "env": {...}, "audio": [floats]}

The audio floats mirror the input for downstream display blocks.
"""
from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path

# src/ on sys.path (this file lives at <repo>/src/sigint/workflow/), so
# `import sigint` resolves even when this subprocess isn't pip-installed
_SRC_DIR = str(Path(__file__).resolve().parent.parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np  # noqa: E402

from sigint.realtime import Gate, GateConfig  # noqa: E402

SAMPLE_RATE = 8000.0
BLOCK = 800
# debug/standalone fallback only (see _wav_stream): real flowgraph use
# always feeds actual samples via the 'feed' command's data field.
WAV = Path(_SRC_DIR) / "sigint" / "fixtures" / "hfdl_1200bps.cf32"

gate = Gate(GateConfig(sample_rate=SAMPLE_RATE, demod=False))
_named_decoders: list[str] = []
# rolling capture of the stream for the offline decode path (last 30 s)
_audio_buf: np.ndarray = np.zeros(0)
_SAMPLE_RATE = SAMPLE_RATE  # module-level default; config may override
_AUDIO_BUF_MAX = int(30 * _SAMPLE_RATE)
# decode-time parameters set via the config command
cfg_demod = False
cfg_rate = SAMPLE_RATE
cfg_mark: float | None = None
cfg_space: float | None = None
cfg_baud: float | None = None
_last_decode_t = -1e9


def _wav_stream():
    """Yield BLOCK-sample blocks looping the packaged HFDL reference capture
    (real off-air capture, 8 kHz, headerless complex float32 -- see hfdl.decode_file)."""
    samples = np.frombuffer(WAV.read_bytes(), dtype=np.complex64).real.astype(np.float64)
    i = 0
    while True:
        block = samples[i:i + BLOCK]
        i += BLOCK
        if len(block) < BLOCK:
            i = BLOCK - len(block)
            block = np.concatenate([block, samples[:i]])
        yield block


_source = _wav_stream()


def env_snapshot(events) -> dict:
    env = {
        "state": ("locked" if gate.active and gate._active and
                  gate._active.locked_baud else
                  "active" if gate.active else "idle"),
        "stream_t": round(gate.stream_time, 3),
    }
    if gate.active and gate._active:
        env["label"] = gate._active.label or ""
    bits = np.asarray(gate.bits)
    env["bits"] = int(len(bits))
    env["bit_head"] = "".join(map(str, bits[:64]))
    if getattr(gate, "hfdl_frames", None):
        fr = gate.hfdl_frames[0]
        env["waveform"] = f"HFDL {fr.label}"
        env["waveform_crc_ok"] = bool(fr.crc_ok)
        env["waveform_bytes"] = len(fr.bytes_out)
        env["waveform_text"] = fr.bytes_out[:256].decode(
            "latin-1", errors="replace")
    if events:
        env["events"] = [e.as_dict() for e in events]
        env["last_event"] = events[-1].as_dict()
    return env


def _read_msg():
    header = sys.stdin.buffer.read(4)
    if len(header) < 4:
        return None
    (n,) = struct.unpack("<I", header)
    return json.loads(sys.stdin.buffer.read(n))


def _write_msg(obj: dict) -> None:
    data = json.dumps(obj).encode()
    sys.stdout.buffer.write(struct.pack("<I", len(data)))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def main() -> None:
    while True:
        msg = _read_msg()
        if msg is None:
            break
        cmd = msg.get("cmd")
        if cmd == "config":
            # reconfigure the gate before the first feed (demod, FSK
            # tones, sweep). Also selects explicit named bitstream
            # decoders whose text lands in env["message"].
            global gate, _named_decoders
            global cfg_demod, cfg_rate, cfg_mark, cfg_space, cfg_baud
            cfg = msg
            cfg_demod = bool(cfg.get("demod", False))
            cfg_rate = cfg.get("sample_rate", SAMPLE_RATE)
            cfg_mark = cfg.get("mark_hz")
            cfg_space = cfg.get("space_hz")
            cfg_baud = cfg.get("baud_hz")
            gc = GateConfig(
                sample_rate=cfg_rate,
                demod=cfg_demod,
                mark_hz=cfg_mark,
                space_hz=cfg_space,
                baud_hz=cfg_baud,
                sweep=cfg.get("sweep", True))
            gate = Gate(gc)
            _named_decoders = cfg.get("named_decoders", [])
            _write_msg({"ok": True})
        elif cmd == "feed":
            global _audio_buf, _last_decode_t
            t_now = time.monotonic()
            x = np.asarray(msg.get("data", []), dtype=np.float64)
            if len(x) == 0:            # no external source: use the wav
                x = next(_source)
            events = gate.process(x)
            env = env_snapshot(events)
            _audio_buf = np.concatenate([_audio_buf, x])[-_AUDIO_BUF_MAX:]
            # explicit named bitstream decoders: run the validated
            # offline sync demod on the buffered audio (the streaming
            # demod's bits can drift on discontinuous-phase signals)
            # and surface the best message text. The buffer is trimmed
            # to the LAST activity segment first: the demod anchors its
            # symbol phase at the buffer start, so untrimmed (silence
            # head / partial burst) buffers rotate the 5-bit character
            # framing and garble the decode.
            # Throttled to once per 2 s: fsk_demod_sync costs ~0.4 s
            # and running it every cycle stalls the flowgraph. A burst
            # end always decodes (the complete audio is in the buffer).
            _burst_ended = any(e["kind"] == "SIGNAL_END"
                               for e in env.get("events", []))
            if _named_decoders and cfg_demod and len(_audio_buf) >= 32000 \
                    and (_burst_ended
                         or t_now - _last_decode_t >= 2.0):
                _last_decode_t = t_now
                from sigint import decoders as sd
                from sigint import dsp as _dsp
                from sigint import detect as _det
                arr = _audio_buf
                try:
                    segs = _det.detect_activity(arr, cfg_rate)
                except ValueError:
                    segs = []
                if segs:
                    s = segs[-1]
                    pre = int(0.2 * cfg_rate)   # small pre-roll
                    lo = max(0, int(s.start_s * cfg_rate) - pre)
                    hi = min(len(arr), int(s.end_s * cfg_rate)
                             + int(0.3 * cfg_rate))
                    arr = arr[lo:hi]
                try:
                    dem = _dsp.fsk_demod_sync(
                        arr, cfg_rate, cfg_mark, cfg_space,
                        cfg_baud)
                except ValueError:
                    dem = None   # buffer too short yet; retry next cycle
                if dem is not None:
                    best = None
                    for name in _named_decoders:
                        r = sd.run_decoder(name, dem.bits.astype(np.uint8))
                        if r.symbol_count >= 8 and (best is None or
                                                    r.error_count < best[1]):
                            best = (r, r.error_count)
                    if best and (not env.get("waveform") or
                                 best[0].error_count == 0):
                        env["waveform"] = best[0].decoder
                        env["waveform_crc_ok"] = True
                        env["waveform_bytes"] = best[0].symbol_count
                        env["waveform_text"] = best[0].text
            _write_msg({
                "env": env,
                "audio": [round(float(v), 6) for v in x],
            })
        elif cmd == "stop":
            break


if __name__ == "__main__":
    main()
