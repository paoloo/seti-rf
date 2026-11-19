"""Name -> Decoder registry.

Each decoder module registers itself on import (see ``bee/decoders/__init__.py``
for the list of built-in modules loaded at startup). This keeps each signal's
decoder in its own file, addressable by name, without a central switch
statement that grows unbounded as more of the 90 signal types are added.
"""
from __future__ import annotations

from sigint._bee.decoders.base import Decoder

_REGISTRY: dict[str, Decoder] = {}


def register(decoder: Decoder) -> None:
    _REGISTRY[decoder.name] = decoder


def get(name: str) -> Decoder | None:
    return _REGISTRY.get(name)


def available() -> list[str]:
    return sorted(_REGISTRY)


def inventory() -> list[dict[str, str]]:
    """Return deterministic, serializable metadata for registered decoders."""
    return [_REGISTRY[name].metadata() for name in sorted(_REGISTRY)]
