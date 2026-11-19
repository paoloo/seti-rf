"""Bitstream decoder bridge: BEE's 76 signal-format decoders.

Thin adapter over ``bee.decoders`` (when importable): the registry there
holds bitstream decoders for Baudot, SITOR, ARQ family, ALE, MIL-STD-188-
110A, STANAG-4285/5066, packet formats, and more. This module:

- discovers availability once (``bee`` is an optional dependency);
- offers :func:`run_decoder` to apply one decoder to a bit array;
- offers :func:`run_all` to sweep every registered decoder and rank results
  by symbol yield vs error rate -- the "what is this signal" sweep an
  analyst runs on a demodulated bitstream;
- exposes :func:`available_decoders` for GUI menus.

Bit convention: LSB-first packed bytes for real captures (the project's
documented packing), i.e. ``np.unpackbits(data, bitorder='little')``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecodeOutcome:
    decoder: str
    text: str
    symbol_count: int
    error_count: int
    score: float            # symbols / (1 + errors), for ranking
    notes: str = ""
    available: bool = True  # False when bee is not importable


_BEE_STATE: dict[str, object] = {"registry": None, "error": None}


def _bee_registry():
    """Import the decoder registry: prefer the full ``bee`` package
    when importable, fall back to the vendored copy in
    ``sigint._bee`` (same modules, numpy-only)."""
    if _BEE_STATE["registry"] is None and _BEE_STATE["error"] is None:
        try:
            import bee.decoders.registry as reg  # type: ignore
            import bee.decoders  # noqa: F401 -- populates the registry
            _BEE_STATE["registry"] = reg
        except ImportError:
            try:
                from sigint import _bee  # vendored copy
                import sigint._bee.decoders.registry as reg  # type: ignore
                import sigint._bee.decoders  # noqa: F401
                _BEE_STATE["registry"] = reg
            except ImportError as exc:
                _BEE_STATE["error"] = str(exc)
    return _BEE_STATE["registry"]


def bee_available() -> bool:
    return _bee_registry() is not None


def bee_unavailable_reason() -> str:
    return str(_BEE_STATE["error"] or "")


def available_decoders() -> list[str]:
    reg = _bee_registry()
    return sorted(reg.available()) if reg else []


def run_decoder(name: str, bits: np.ndarray) -> DecodeOutcome:
    """Apply one registered BEE decoder to a 0/1 bit array."""
    reg = _bee_registry()
    if reg is None:
        return DecodeOutcome(decoder=name, text="", symbol_count=0, error_count=0,
                             score=0.0, notes=f"bee not importable: {bee_unavailable_reason()}",
                             available=False)
    decoder = reg.get(name)
    if decoder is None:
        return DecodeOutcome(decoder=name, text="", symbol_count=0, error_count=0,
                             score=0.0, notes="unknown decoder", available=False)
    try:
        result = decoder.decode(np.asarray(bits, dtype=np.uint8))
    except Exception as exc:  # a crashing decoder must not kill the sweep
        return DecodeOutcome(decoder=name, text="", symbol_count=0, error_count=0,
                             score=0.0, notes=f"decoder raised {type(exc).__name__}: {exc}")
    score = result.symbol_count / (1 + result.error_count)
    return DecodeOutcome(decoder=name, text=result.text,
                         symbol_count=result.symbol_count,
                         error_count=result.error_count, score=score,
                         notes=result.notes)


def run_all(bits: np.ndarray, min_symbols: int = 4,
            top: int | None = None) -> list[DecodeOutcome]:
    """Sweep every registered decoder; rank by score, best first.

    Score: (printable − control-char penalty − hex-dump penalty) / (1 +
    errors). Framing-parameter decoders (LPC10/MELP 'sync=0 params=0x...'
    fire on any input) are discounted by the hex-dump term: their output
    is mostly 0x-hex tokens, which real decodes don't emit. Control
    characters (NULs etc.) also count against a decoder. Decoders
    producing fewer than `min_symbols` usable symbols are dropped.
    """
    outcomes = [run_decoder(name, bits) for name in available_decoders()]
    hits = []
    for o in outcomes:
        if not o.available or o.symbol_count < min_symbols:
            continue
        n = max(1, len(o.text))
        printable = sum(1 for c in o.text
                        if c.isprintable() or c in "\n\r\t") / n
        control = sum(1 for c in o.text
                      if ord(c) < 32 and c not in "\n\r\t") / n
        hex_ratio = len(re.findall(r"0x[0-9a-fA-F]{4,}", o.text)) * 12 / n
        score = (printable - 2.0 * control - hex_ratio) / (1 + o.error_count)
        hits.append((score, o.symbol_count, o))
    hits.sort(key=lambda t: (-t[0], -t[1]))
    ranked = [o for _, _, o in hits]
    return ranked[:top] if top is not None else ranked


def bits_from_bytes(data: bytes, bit_order: str = "little") -> np.ndarray:
    """Raw capture bytes -> 0/1 bit array (LSB-first per project convention)."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder=bit_order)
