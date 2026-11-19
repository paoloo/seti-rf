"""CyberEther / GNU Radio adapters for the sigint recognition gate.

Two integration shapes for the same Gate:

1. **CyberEther Python block** — ``workflow/cyberether_block.py`` is the
   reference copy of the block code; paste it (or use the shipped
   ``hfdl_flowgraph.yml`` / ``baudot_demo.yml``, which embed the same
   logic inline, since CyberEther's block editor needs literal code) into
   a CyberEther Python block, connect an audio-rate signal, and the
   gate's recognition state is published to the flowgraph Environment
   under ``sigint.*`` where any other block (or the Environment window)
   can read it live. The gate itself runs in a subprocess
   (``gate_subprocess.py``): CyberEther's embedded interpreter can't run
   numpy reliably.

2. **GNU Radio** — a standard ``gr.basic_block`` (``SigintGateBB`` for a
   real float stream, ``SigintGateCF`` for complex IQ that is tuned to an
   audio-band channel first). Events become PMT messages on the ``rec``
   output message port; the samples pass through untouched, so the gate
   sits in the flowgraph as a passive tap.

Both adapters are thin: all recognition logic lives in
``sigint.realtime.Gate``.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# GNU Radio adapter (importable only where gnuradio is installed)
# --------------------------------------------------------------------------

__all__ = ["SigintGateBB", "SigintGateCF", "make_gate"]


def make_gate(sample_rate: float, demod: bool = False,
              mark_hz: float | None = None, space_hz: float | None = None,
              baud_hz: float | None = None):
    """Construct the underlying gate with pipeline-friendly defaults."""
    from sigint.realtime import Gate, GateConfig
    cfg = GateConfig(sample_rate=sample_rate, demod=demod,
                     mark_hz=mark_hz, space_hz=space_hz, baud_hz=baud_hz)
    return Gate(cfg)


try:  # gnuradio is optional
    from gnuradio import gr  # type: ignore
    import pmt  # type: ignore

    _HAVE_GR = True
except ImportError:  # pragma: no cover - exercised only without gnuradio
    _HAVE_GR = False


if _HAVE_GR:

    class _GateBase(gr.basic_block):
        """Shared implementation: samples in, samples out, events out."""

        def __init__(self, name: str, itemsize: int, sample_rate: float,
                     demod: bool = False, mark_hz: float | None = None,
                     space_hz: float | None = None,
                     baud_hz: float | None = None):
            gr.basic_block.__init__(
                self, name,
                gr.io_signature(1, 1, itemsize),
                gr.io_signature(1, 1, itemsize))
            self.message_port_register_out(gr.pmt.intern("rec"))
            self.gate = make_gate(sample_rate, demod=demod,
                                  mark_hz=mark_hz, space_hz=space_hz,
                                  baud_hz=baud_hz)
            # feed the gate in sub-block chunks so events carry a
            # fine-grained timestamp
            self._chunk = max(1, int(sample_rate * 0.05))  # 50 ms

        def _emit(self, events) -> None:
            for ev in events:
                d = ev.as_dict()
                d["label"] = d.get("label") or ""
                self.message_port_publish(
                    gr.pmt.intern("rec"),
                    pmt.to_pmt(d))

        def general_work(self, input_items, output_items):
            _in = input_items[0]
            _out = output_items[0]
            n = len(_in)
            produced = 0
            i = 0
            while i < n:
                j = min(i + self._chunk, n)
                self._emit(self.gate.process(_in[i:j]))
                i = j
            produced = n
            _out[:produced] = _in[:produced]
            self.consume(0, produced)
            return produced

    class SigintGateBB(_GateBase):
        """Real float (audio) input. itemsize = float32."""

        def __init__(self, sample_rate: float, demod: bool = False,
                     mark_hz: float | None = None,
                     space_hz: float | None = None,
                     baud_hz: float | None = None):
            import numpy as np
            gr.basic_block.__init__.__self__  # noqa: B018 - touch for clarity
            super().__init__("sigint_gate_bb", gr.sizeof_float, sample_rate,
                             demod=demod, mark_hz=mark_hz,
                             space_hz=space_hz, baud_hz=baud_hz)

        def general_work(self, input_items, output_items):
            import numpy as np
            _in = input_items[0]
            _out = output_items[0]
            n = len(_in)
            x = np.asarray(_in, dtype=np.float64)
            i = 0
            while i < n:
                j = min(i + self._chunk, n)
                self._emit(self.gate.process(x[i:j]))
                i = j
            _out[:n] = _in[:n]
            self.consume(0, n)
            return n

    class SigintGateCF(_GateBase):
        """Complex IQ input tuned to an audio-band channel: the gate sees
        the magnitude (envelope) of the complex stream."""

        def __init__(self, sample_rate: float, demod: bool = False,
                     mark_hz: float | None = None,
                     space_hz: float | None = None,
                     baud_hz: float | None = None):
            super().__init__("sigint_gate_cf", gr.sizeof_gr_complex,
                             sample_rate, demod=demod, mark_hz=mark_hz,
                             space_hz=space_hz, baud_hz=baud_hz)

        def general_work(self, input_items, output_items):
            import numpy as np
            _in = input_items[0]
            _out = output_items[0]
            n = len(_in)
            mag = np.abs(np.asarray(_in, dtype=np.complex128))
            i = 0
            while i < n:
                j = min(i + self._chunk, n)
                self._emit(self.gate.process(mag[i:j]))
                i = j
            _out[:n] = _in[:n]
            self.consume(0, n)
            return n
