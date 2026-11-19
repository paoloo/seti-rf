"""DMR / P25 / NXDN: frame-sync detection and characterization for the
three common digital trunked-radio families -- deliberately NOT a voice
decoder. All three encode voice with a patented vocoder (AMBE+2/IMBE),
which this package cannot legally reimplement; what it can do, and does
here, is the same kind of work as every other waveform-level probe in
this package: recognize which protocol (and which burst sub-type) is on
the air from its frame sync pattern, and read the cheaply-available
cleartext metadata that sits right next to that sync (P25's NAC/DUID) --
"there's a P25 control channel here, NAC 0x293" rather than the traffic
it carries.

All three share a 4-level FSK/C4FM physical layer heritage (2 bits per
symbol, the same `dibit -> nominal frequency` mapping this module uses
for all of them), which is what makes one shared demod primitive
reasonable here instead of three separate ones.

Sync patterns (P25's 48-bit frame sync, DMR's four 48-bit BS/MS voice/
data patterns, NXDN's 20-bit FSW) are transcribed from public protocol
documentation (P25's is TIA-102, DMR's ETSI TS 102 361-1, NXDN's the
NXDN Forum's own Technical Specification Part 1) -- these are widely
published, unlike an FEC matrix there's little room for a subtle
transcription error to hide.

Self-rejecting: no candidate sync pattern correlates above the match
threshold anywhere in the buffer returns ``None``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# Sync patterns, as plain 0/1 bit sequences (MSB-first, matching how each
# spec prints its sync word in hex).
# --------------------------------------------------------------------------

def _hex_to_bits(h: str, n_bits: int) -> list[int]:
    v = int(h, 16)
    return [(v >> (n_bits - 1 - i)) & 1 for i in range(n_bits)]


P25_SYNC = _hex_to_bits("5575F5FF77FF", 48)

DMR_SYNC = {
    "BS Voice": _hex_to_bits("755FD7DF75F7", 48),
    "BS Data": _hex_to_bits("DFF57D75DF5D", 48),
    "MS Voice": _hex_to_bits("7F7D5DD57DFD", 48),
    "MS Data": _hex_to_bits("D5D7F77FD757", 48),
}

NXDN_FSW = _hex_to_bits("CDF59", 20)

_SYNC_PATTERNS: dict[str, list[int]] = {"P25": P25_SYNC, "NXDN": NXDN_FSW}
_SYNC_PATTERNS.update({f"DMR {k}": v for k, v in DMR_SYNC.items()})

# nominal C4FM/4FSK dibit -> deviation-level mapping shared by all three
# (01=+3, 00=+1, 10=-1, 11=-3 -- the standard convention).
_DIBIT_LEVEL = {(0, 1): 3, (0, 0): 1, (1, 0): -1, (1, 1): -3}
_LEVEL_DIBIT = {v: k for k, v in _DIBIT_LEVEL.items()}


def bits_to_symbols(bits: list[int]) -> list[int]:
    """A 0/1 bit sequence (even length) -> its dibit deviation levels."""
    return [_DIBIT_LEVEL[(bits[i], bits[i + 1])] for i in range(0, len(bits), 2)]


# --------------------------------------------------------------------------
# Synthesis and detection
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TrunkedSync:
    protocol: str        # 'P25', 'NXDN', or 'DMR <subtype>'
    confidence: float    # fraction of sync symbols that matched
    nac: int | None = None    # P25 only: raw NAC (best-effort, no BCH correction)
    duid: int | None = None   # P25 only: raw DUID


def generate_sync_burst(protocol: str, sample_rate: float, symbol_rate_hz: float = 4800.0,
                        base_hz: float = 20000.0, deviation_step_hz: float = 2000.0,
                        payload_bits: int = 64) -> np.ndarray:
    """Synthesize `protocol`'s frame sync pattern (`'P25'`, `'NXDN'`, or
    one of the `DMR_SYNC` keys prefixed `'DMR '`) followed by random
    payload bits, as 4-level FSK. For testing sync detection, not a
    realistic full burst/superframe (which needs far more protocol-
    specific structure than framing alone)."""
    if protocol not in _SYNC_PATTERNS:
        raise ValueError(f"unknown protocol/pattern {protocol!r}; "
                         f"choose from {sorted(_SYNC_PATTERNS)}")
    sync_bits = _SYNC_PATTERNS[protocol]
    rng = np.random.default_rng(0)
    payload = rng.integers(0, 2, payload_bits).tolist()
    levels = bits_to_symbols(sync_bits + payload)

    sps = max(1, int(round(sample_rate / symbol_rate_hz)))
    phase = 0.0
    chunks = []
    t = np.arange(sps) / sample_rate
    for level in levels:
        f = base_hz + level * deviation_step_hz
        chunks.append(np.cos(2 * np.pi * f * t + phase))
        phase = (phase + 2 * np.pi * f * sps / sample_rate) % (2 * np.pi)
    return np.concatenate(chunks)


def detect(samples: np.ndarray, sample_rate: float, symbol_rate_hz: float = 4800.0,
          base_hz: float = 20000.0, deviation_step_hz: float = 2000.0,
          phase_steps: int = 8, min_confidence: float = 0.95) -> TrunkedSync | None:
    """Demodulate 4-level symbols and correlate against every known sync
    pattern at every candidate bit-timing offset; report the best match
    that clears `min_confidence`. For a P25 match, also reads the raw
    NAC/DUID from the 64 bits immediately following sync (best-effort --
    no BCH(63,16,11) error correction, so a bit error there corrupts the
    reported value; this is framing/characterization, not a robust
    control-channel receiver).

    NXDN's own 20-bit FSW is short enough that a single-offset match
    against it isn't a reliable "is this really NXDN" signal on its own
    -- this is a real, publicly-documented weakness of that pattern (DSD's
    authors extend it with the adjacent LICH field for exactly this
    reason), not specific to this implementation -- so it additionally
    requires an *exact* match here rather than just clearing
    `min_confidence`, trading some sensitivity to real bit errors for not
    false-triggering on noise.
    """
    x = np.asarray(samples, dtype=np.float64).ravel()
    sps = max(1, int(round(sample_rate / symbol_rate_hz)))
    t = np.arange(sps) / sample_rate
    levels = sorted(_LEVEL_DIBIT)
    refs = {lv: np.exp(-2j * np.pi * (base_hz + lv * deviation_step_hz) * t) for lv in levels}

    best: TrunkedSync | None = None
    for off in np.linspace(0, sps - 1, phase_steps).astype(int):
        n_sym = (len(x) - off) // sps
        if n_sym < 20:
            continue
        sym_levels = np.empty(n_sym, dtype=np.int64)
        for k in range(n_sym):
            seg = x[off + k * sps: off + (k + 1) * sps]
            corr = {lv: abs(np.dot(seg, ref)) for lv, ref in refs.items()}
            sym_levels[k] = max(corr, key=corr.get)
        bits: list[int] = []
        for lv in sym_levels:
            bits.extend(_LEVEL_DIBIT[int(lv)])
        bitstr = "".join(str(b) for b in bits)

        for name, pattern in _SYNC_PATTERNS.items():
            pat_str = "".join(str(b) for b in pattern)
            n = len(pat_str)
            for i in range(len(bitstr) - n):
                window = bitstr[i:i + n]
                matches = sum(1 for a, b in zip(window, pat_str) if a == b)
                confidence = matches / n
                required = 1.0 if n < 24 else min_confidence
                if confidence >= required and (best is None or confidence > best.confidence):
                    nac = duid = None
                    if name == "P25" and i + n + 16 <= len(bitstr):
                        nac = int(bitstr[i + n: i + n + 12], 2)
                        duid = int(bitstr[i + n + 12: i + n + 16], 2)
                    best = TrunkedSync(protocol=name, confidence=confidence, nac=nac, duid=duid)
    return best
