"""CyberEther Python block: sigint recognition gate inside a flowgraph.

This is the reference copy of the block code shipped inline inside
``hfdl_flowgraph.yml`` and ``baudot_demo.yml`` (CyberEther's Python-block
editor needs literal, self-contained code -- ``__file__`` is undefined in
the exec'd block and importing a sibling file from it is not reliable, so
the two shipped flowgraphs paste this logic directly rather than loading
this module). Use this file as the copy-paste source for a new flowgraph,
or as documentation of what the block does; edit ``_REPO``/``_PY`` below
for your machine, paste ``compute``/``cleanup`` (and the setup above them)
into a CyberEther Python block's code editor, one input, one output.

Architecture (load-bearing, don't simplify back): the recognition gate
runs in a SEPARATE python subprocess (`gate_subprocess.py`, a real
python3 with numpy + scipy). CyberEther's embedded interpreter cannot run
numpy reliably (RecursionError / multiarray failures inside its dispatch
layers), so no numpy import happens in this file itself -- only in the
worker it spawns.

Published to the flowgraph Environment every cycle under `sigint.*`:

    sigint.state     "idle" | "active" | "locked"
    sigint.label     modulation label ("FSK", "tone", ...)
    sigint.baud      locked symbol rate (Hz), or 0
    sigint.snr       current SNR estimate (dB), or 0
    sigint.events    latest RecognitionEvent dict (kind/t_s/label/...)
    sigint.bits      number of demodulated bits in the rolling buffer
    sigint.bit_head  the first 64 bits as a string
    sigint.waveform  named waveform: "HFDL 1200 bps 1.8 sec" after a
                     PHY decode, else the bit-sweep's best decoder;
                     absent when nothing matched
    sigint.waveform_crc_ok / waveform_bytes / waveform_text
                     HFDL frame decode details (waveform_text = first
                     256 payload bytes, latin-1)

The Environment window shows these live; downstream Python blocks read
them via `ctx.env.get("sigint", {})` and can gate their own logic on
`sigint.state`.
"""
from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

# --------------------------------------------------------------------------
# configuration (edit these for your machine and signal)
# EDIT THIS: the one machine-specific path -- where this repo lives on
# YOUR machine (used to reach the gate subprocess and the sigint package
# for the subprocess's PYTHONPATH).
_REPO = Path("/Users/paolo/Workspace/SRC/Python/BEE/sigint")
_WORKER = _REPO / "src/sigint/workflow/gate_subprocess.py"
_PY = "/Users/paolo/Workspace/CONDA/envs/mcp/bin/python3"
SAMPLE_RATE = 16000.0
DEMOD = True
MARK_HZ = 2125.0
SPACE_HZ = 1275.0
BAUD_HZ = None            # None = auto-measure, or e.g. 50.0
NAMED_DECODERS: list[str] = []   # e.g. ["Baudot/ITA2"] to force a decoder
# --------------------------------------------------------------------------

_proc = subprocess.Popen(
    [_PY, str(_WORKER)],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    stderr=open("/tmp/sigint_worker.err", "ab"))


def _call(payload: dict) -> dict:
    data = json.dumps(payload).encode()
    _proc.stdin.write(struct.pack("<I", len(data)))
    _proc.stdin.write(data)
    _proc.stdin.flush()
    (n,) = struct.unpack("<I", _proc.stdout.read(4))
    return json.loads(_proc.stdout.read(n))


_call({"cmd": "config", "sample_rate": SAMPLE_RATE, "demod": DEMOD,
       "mark_hz": MARK_HZ, "space_hz": SPACE_HZ, "baud_hz": BAUD_HZ,
       "named_decoders": NAMED_DECODERS})

_printed_classify = [False]   # survives across cycles (globals reset
                              # only on code reload)


def compute(ctx):
    # input: a real or complex batch from the File Reader, as if live
    x = ctx.inputs[0]
    resp = _call({"cmd": "feed", "data": [float(v.real) for v in x]})
    env = resp["env"]

    # top-level env values must be mappings
    ctx.env["sigint"] = env

    # report named waveforms in the block console (there is no text
    # block in CyberEther; the console is the readout)
    for ev in env.get("events", []):
        if ev["kind"] == "WAVEFORM":
            print(f"### WAVEFORM: {env.get('waveform', ev.get('label', '?'))}"
                  f"  CRC {'OK' if env.get('waveform_crc_ok') else 'BAD'}"
                  f"  {env.get('waveform_bytes', '?')} bytes")
            text = env.get("waveform_text", "")
            printable = "".join(c if 32 <= ord(c) < 127 else "."
                                for c in text)
            print(f"### payload: {printable[:120]}")
        elif ev["kind"] == "SIGNAL_CLASSIFIED" and not _printed_classify[0]:
            _printed_classify[0] = True
            print(f"### signal: {ev.get('label', '?')} "
                  f"({ev.get('snr_db', 0):.0f} dB)")
        elif ev["kind"] == "SIGNAL_START":
            _printed_classify[0] = False   # new burst -> print again
            print("### burst detected")

    # mirror audio downstream for the plots
    out = ctx.outputs[0]
    out[...] = x
    ctx.output_attrs[0]["sampleRate"] = SAMPLE_RATE


def cleanup():
    try:
        _call({"cmd": "stop"})
    except Exception:
        pass
    _proc.terminate()
