"""Authoritative original-BEE decoder order and DLL mapping."""
from __future__ import annotations

FIRST_BLOCK = (
    "Stanag-4285", "AUTOSPEC", "ALIS", "MEROD-USA", "12s12p", "ITA-3-1",
    "ITA-3-2", "SITOR-B (7x22)", "HNG-FEC", "RUM-FEC 1", "RUM-FEC 2",
    "DUP-ARQ 2", "1s1", "ROU-MOI-FEC", "2b 2s1 (5s5)", "2s2",
    "PICCOLO MK-6 (006a)", "PICCOLO MK-6 (006b)", "PICCOLO MK-6 (006c)",
    "COQUELET MK8", "2s2 (2x10)", "DUP-FEC-2", "SI-ARQ 70", "ARQ-E3",
    "POL-ARQ", "ARQ-E", "FEC-A", "SI-ARQ 84", "DUP-ARQ 1", "8s8k10",
    "8s8k5", "ARQ-6/90", "ALE", "PACKET AX.25", "PACTOR FEC",
    "PACTOR LONG", "PACTOR-I", "SITOR-A", "SWED ARQ", "ARQ (7x22) 1",
    "ARQ (7x22) 2", "ARQ1a-1", "ARQ1a-2", "SITOR-FEC", "SITOR-B (7x29)",
    "BULG-ASCII", "ASCII", "BAUDOT",
)

SECOND_BLOCK = (
    ("192 (Period)", 70), ("420 (Period)", 71),
    ("FED-STD-1052 (Harris Modems)", 72),
    ("MIL-STD-188-110A (1060-period)", 73),
    ("RSX.25 (Rohde&Schwarz Modems)", 74), ("Stanag-5066", 75),
    ("ZModem", 76), ("KRE-PSK", 77), ("30s30", 78), ("MEROD-FIN", 79),
    ("MEROD-IND", 80), ("MIL-STD-188-110A (2760-period)", 81),
    ("Ital HDLC", 82), ("DatronLINK", 83), ("ALE-3G", 84), ("LPC10", 85),
    ("HDLC", 86), ("ARM-DA", 87), ("Hamming", 88), ("Q15X25", 89),
    ("MELP", 90),
)

ALL_DECODER_NAMES = FIRST_BLOCK + tuple(name for name, _ in SECOND_BLOCK)
BUILTIN_DECODERS = frozenset({"Stanag-4285"})


def original_dll_mapping() -> dict[str, str | None]:
    """Map captions to original DLL names; ``None`` means built into Bee.exe."""
    result: dict[str, str | None] = {"Stanag-4285": None}
    for index, name in enumerate(FIRST_BLOCK[1:]):
        result[name] = f"{index:03d}.dll"
    result.update({name: f"{index:03d}.dll" for name, index in SECOND_BLOCK})
    return result
