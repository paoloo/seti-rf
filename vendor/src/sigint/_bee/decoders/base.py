"""Decoder plugin interface.

A decoder turns a run of bits (one ``Block``'s worth, per the ``.blk``
session schema — see ``bee.core.project``) into human-readable symbols. This
mirrors the original tool's per-signal decoder plugins (``Lib/*.dll``), but
as small, independently testable Python modules registered by name instead
of compiled DLLs loaded by numeric index.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
from dataclasses import dataclass, field

import numpy as np

from sigint._bee.core.project import Block


@dataclass(frozen=True)
class DecodeArtifact:
    """A binary/text product reconstructed by a decoder.

    Artifact bytes stay out of JSON reports; :meth:`as_dict` records enough
    information to audit a separately exported file offline.
    """

    name: str
    data: bytes
    media_type: str = "application/octet-stream"
    confidence: str = "structural"
    notes: str = ""

    def as_dict(self) -> dict[str, str | int]:
        return {
            "name": self.name,
            "bytes": len(self.data),
            "sha256": hashlib.sha256(self.data).hexdigest(),
            "media_type": self.media_type,
            "confidence": self.confidence,
            "notes": self.notes,
        }


@dataclass
class DecodeResult:
    text: str
    symbol_count: int
    error_count: int = 0
    notes: str = ""
    artifacts: tuple[DecodeArtifact, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {"text": self.text, "symbol_count": self.symbol_count,
                "error_count": self.error_count, "notes": self.notes,
                "artifacts": [artifact.as_dict() for artifact in self.artifacts]}


class Decoder(ABC):
    """Base class for a signal-specific decoder."""

    name: str = "unnamed"
    description: str = ""
    status: str = "partial"
    confidence: str = "experimental"
    layer: str = "unknown"
    provenance: str = "implementation documentation"

    def metadata(self) -> dict[str, str]:
        return {key: getattr(self, key) for key in
                ("name", "description", "status", "confidence", "layer", "provenance")}

    @abstractmethod
    def decode(self, bits: np.ndarray, block: Block | None = None) -> DecodeResult:
        """Decode a 0/1 bit array into text.

        ``block`` carries the field parameters (size, sync, order, mask,
        conform) if a ``.blk`` session provided one; decoders that don't need
        that context (fixed-width codes) may ignore it.
        """
        raise NotImplementedError
