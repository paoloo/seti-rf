"""CCIR 476 7-unit constant-ratio code (Tier B: public ITU-R standard).

Used as the character alphabet by the SITOR/AMTOR/PACTOR-I family of HF ARQ
modes: each character is 7 bits with exactly 4 marks (1) and 3 spaces (0) --
a constant-ratio code that lets a receiver detect (and, in ARQ mode, request
retransmission of) a corrupted character. Two shift states (LTRS/FIGS) select
between the letter and figure/punctuation meaning of the same 7-bit values,
the same way Baudot/ITA2 does at 5 bits.

Table verified against a working reference implementation (AI5GW/CCIR476,
GPL-3.0, an Arduino library for NAVTEX/SITOR-B signal generation):
https://github.com/AI5GW/CCIR476

Codeword alignment (this module's addition): rather than assume codewords
start at bit 0 of whatever range the caller passed in, :func:`find_codeword_alignment`
searches all 7 bit-phase offsets, both byte-bit orientations (the given order
and its per-byte-bit-reversed form), and both polarities, scoring each by the
fraction of 7-bit groups that satisfy the 4-mark/3-space rule. That is the same
"search, then report a measured confidence" approach used by
``vocoder_frame.find_frame_alignment`` and ``ale.find_word_alignment``. The
random baseline for arbitrary 7-bit groups is C(7,4)/2**7 = 0.2734, so a rate
well above that is real evidence of alignment.

Measured against the bundled real captures (`tests/fixtures/bee_demo/Signals/`), searching
recovers alignments an earlier fixed-orientation search had missed. Every
confirmed alignment below is in the **byte-bit-reversed** orientation:

- ``Sitor-B (7x22).bin``: 0.9985 at offset 2, not inverted.
- ``POL-ARQ.bin``: 0.9869 at offset 3, inverted.
- ``ARQ-E3.bin``: 0.8693 at offset 4, inverted.
- ``ARQ-E.bin``: 0.5202 at offset 3, inverted (weak).
- ``SwedARQ.BIN``: 0.4893 at offset 2, not inverted (weak).
- ``Sitor-A.BIN`` 0.3889, ``FEC-A.BIN`` 0.3651, ``SI-ARQ 84.bin`` 0.3441 (weak).
- ``ARQ-6_90.bin`` 0.2631, ``DUP-ARQ-1.BIN`` 0.2157, ``SI-ARQ 70.bin`` 0.1914,
  ``DUP-ARQ-2.bin`` 0.0922 and all three PACTOR captures (0.2317-0.2495) sit at
  or below the random baseline: no CCIR 476 codeword grid is present in those
  files under any offset/orientation/polarity, so this alphabet is not
  validated for those names.

Two of the confirmed alignments produce genuinely readable plaintext, which is
the strongest available proof that both the alignment search and the alphabet
table are correct: ``POL-ARQ.bin`` decodes directly to Polish prose, and
``Sitor-B (7x22).bin`` decodes to a Polish news bulletin once its time
diversity is undone (below).

Time diversity (SITOR-B / NAVTEX "mode B" FEC): :func:`find_time_diversity`
measures, for each candidate spacing, how often two valid codewords that many
positions apart are equal, and whether those matches concentrate on one parity
of position. ``Sitor-B (7x22).bin`` shows a standout spacing of 5 (match rate
0.457 overall, 0.877 on even positions versus 0.033 on odd) -- i.e. each
character is transmitted at positions 2k and 2k+5, the documented DX/RX
character-diversity pattern. Combining the two copies (use the DX copy unless
it fails the constant-ratio check, else the RX copy) raises that capture's
valid-codeword rate from 0.9982 to **1.000** and yields clean text. The check
is deliberately strict enough not to fire on ``POL-ARQ.bin``, whose weaker
spacing-5 match rate (0.272) comes from an ARQ block-overlap retransmission
(four characters repeated around a REP symbol) rather than character diversity.

ARQ block overlap (POL-ARQ): :func:`find_rep_overlap` measures whether each REP
(0x66) control symbol is straddled by an identical run of characters.
``POL-ARQ.bin`` carries 339 REP symbols spaced 17 characters apart (181 times)
or 16 (83 times), and for **0.94** of them the four characters before the REP
are repeated exactly after it -- its ARQ retransmission overlap. No other
bundled capture in this family exceeds 0.11 at any overlap length from 1 to 8,
so the measurement discriminates cleanly. Dropping each REP plus its repeated
run turns POL-ARQ's output into continuous Polish prose.

This module still decodes the *character alphabet* plus the alignment and, where
detected, the SITOR-B time diversity and POL-ARQ REP overlap. It does not attempt the ARQ
handshake/retransmission framing, which differs per named mode (PACTOR,
SITOR-A, ARQ-E, ARQ-E3, ARQ-M2, FEC-A, SI-ARQ, POL-ARQ, SWED-ARQ, DUP-ARQ,
ARQ-6/90, ARQ1a, ...). All of those modes are registered against this same
alphabet decoder since that's their common, well-documented character set;
the mode-specific packet/retransmission structure is out of scope for a
static bitstream viewer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sigint._bee.decoders.base import DecodeResult, Decoder
from sigint._bee.decoders.registry import register

LTRS_SHIFT = 0x5A
FIGS_SHIFT = 0x36
ALPHA = 0x0F  # SITOR-B phasing signal 1
BETA = 0x33  # SITOR-B phasing signal 2
REP = 0x66  # AMTOR/PACTOR ARQ "repeat request" control signal

_LETTERS = {
    0x47: "A", 0x72: "B", 0x1D: "C", 0x53: "D", 0x56: "E", 0x1B: "F", 0x35: "G",
    0x69: "H", 0x4D: "I", 0x17: "J", 0x1E: "K", 0x65: "L", 0x39: "M", 0x59: "N",
    0x71: "O", 0x2D: "P", 0x2E: "Q", 0x55: "R", 0x4B: "S", 0x74: "T", 0x4E: "U",
    0x3C: "V", 0x27: "W", 0x3A: "X", 0x2B: "Y", 0x63: "Z",
    0x78: "\r", 0x6C: "\n", 0x5C: " ",
}
_FIGURES = {
    0x2D: "0", 0x2E: "1", 0x27: "2", 0x56: "3", 0x55: "4", 0x74: "5", 0x2B: "6",
    0x4E: "7", 0x4D: "8", 0x71: "9", 0x17: "'", 0x1B: "!", 0x1D: ":", 0x1E: "(",
    0x35: "&", 0x39: ".", 0x3A: "/", 0x3C: "=", 0x47: "-", 0x53: "$", 0x59: ",",
    0x63: "+", 0x65: ")", 0x69: "#", 0x72: "?",
    0x78: "\r", 0x6C: "\n", 0x5C: " ",
}
_SPECIAL = {ALPHA: "\x01", BETA: "\x02", REP: "\x03"}


CODEWORD_BITS = 7
MARKS_PER_CODEWORD = 4
#: Fraction of arbitrary 7-bit groups that happen to carry exactly four marks.
RANDOM_RATIO = 35 / 128
_CONFIDENT_RATIO = 0.60
_MIN_CODEWORDS = 50

_MIN_DIVERSITY_CODEWORDS = 200
_MAX_DIVERSITY_SPACING = 16
_CONFIDENT_DIVERSITY_MATCH = 0.40
_DIVERSITY_MARGIN = 2.0

_MAX_REP_OVERLAP = 8
_CONFIDENT_REP_MATCH = 0.50
_MIN_REPS = 20

_WEIGHTS = (1 << np.arange(CODEWORD_BITS)).astype(np.int64)  # LSB first
_MARKS = np.array([bin(v).count("1") for v in range(1 << CODEWORD_BITS)], dtype=np.int8)


def _mark_count(value: int) -> int:
    return bin(value).count("1")


def _reverse_each_byte(bits: np.ndarray) -> np.ndarray:
    complete = len(bits) // 8 * 8
    result = bits.copy()
    if complete:
        result[:complete] = bits[:complete].reshape(-1, 8)[:, ::-1].reshape(-1)
    return result


def _codewords(bits: np.ndarray, offset: int) -> np.ndarray:
    """The 7-bit codeword values on the stride-7 grid starting at ``offset``."""
    count = (bits.size - offset) // CODEWORD_BITS
    if count <= 0:
        return np.zeros(0, dtype=np.int64)
    grid = bits[offset: offset + count * CODEWORD_BITS].reshape(count, CODEWORD_BITS)
    return grid.astype(np.int64) @ _WEIGHTS


@dataclass(frozen=True)
class CodewordAlignment:
    """Where the 7-unit codeword grid actually starts, and how well it holds."""

    offset: int
    orientation: str  # "selected" or "byte-reversed"
    inverted: bool
    ratio: float  # fraction of groups carrying exactly four marks
    codeword_count: int

    @property
    def confident(self) -> bool:
        return self.ratio >= _CONFIDENT_RATIO and self.codeword_count >= _MIN_CODEWORDS

    def apply(self, bits: np.ndarray) -> np.ndarray:
        """The codeword values this alignment selects out of ``bits``."""
        oriented = bits if self.orientation == "selected" else _reverse_each_byte(bits)
        if self.inverted:
            oriented = 1 - oriented
        return _codewords(oriented, self.offset)


def find_codeword_alignment(bits: np.ndarray) -> CodewordAlignment:
    """Search bit phase, byte-bit orientation, and polarity for the codeword grid.

    Scores each candidate by the fraction of 7-bit groups carrying exactly four
    marks. :data:`RANDOM_RATIO` (0.2734) is what an unaligned or non-CCIR-476
    stream scores, so only a clearly higher rate is evidence. Ties keep the
    first candidate tried, which is offset 0 of the caller's own bit order.
    """
    best: CodewordAlignment | None = None
    for orientation in ("selected", "byte-reversed"):
        oriented = bits if orientation == "selected" else _reverse_each_byte(bits)
        for inverted in (False, True):
            polarised = 1 - oriented if inverted else oriented
            for offset in range(CODEWORD_BITS):
                values = _codewords(polarised, offset)
                if values.size == 0:
                    continue
                ratio = float((_MARKS[values] == MARKS_PER_CODEWORD).mean())
                candidate = CodewordAlignment(
                    offset, orientation, inverted, ratio, int(values.size)
                )
                if best is None or candidate.ratio > best.ratio:
                    best = candidate
    if best is None:
        return CodewordAlignment(0, "selected", False, 0.0, 0)
    return best


@dataclass(frozen=True)
class TimeDiversity:
    """A repeated-character spacing, as used by SITOR-B/NAVTEX mode-B FEC."""

    spacing: int
    match_ratio: float  # over all positions
    runner_up_ratio: float  # best match ratio at any other spacing
    parity: int  # position parity the matches concentrate on
    parity_ratio: float
    other_parity_ratio: float

    @property
    def confident(self) -> bool:
        return (
            self.match_ratio >= _CONFIDENT_DIVERSITY_MATCH
            and self.match_ratio >= _DIVERSITY_MARGIN * self.runner_up_ratio
            and self.parity_ratio >= _DIVERSITY_MARGIN * self.other_parity_ratio
        )


def find_time_diversity(values: np.ndarray) -> TimeDiversity | None:
    """Measure whether valid codewords repeat at a standout spacing.

    SITOR-B transmits every character twice, at positions ``2k`` and ``2k+d``,
    so matches at the true spacing land on one position parity only. Returns
    ``None`` when the stream is too short to measure.
    """
    valid = _MARKS[values] == MARKS_PER_CODEWORD
    if int(valid.sum()) < _MIN_DIVERSITY_CODEWORDS:
        return None
    ratios: dict[int, float] = {}
    matches: dict[int, np.ndarray] = {}
    for spacing in range(1, _MAX_DIVERSITY_SPACING + 1):
        both = valid[:-spacing] & valid[spacing:]
        denominator = int(both.sum())
        if denominator < _MIN_DIVERSITY_CODEWORDS:
            continue
        hit = (values[:-spacing] == values[spacing:]) & both
        ratios[spacing] = float(hit.sum()) / denominator
        matches[spacing] = hit
    if not ratios:
        return None
    spacing = max(ratios, key=lambda s: ratios[s])
    runner_up = max((r for s, r in ratios.items() if s != spacing), default=0.0)
    hit = matches[spacing]
    even, odd = float(hit[0::2].mean()), float(hit[1::2].mean())
    parity = 0 if even >= odd else 1
    return TimeDiversity(
        spacing=spacing,
        match_ratio=ratios[spacing],
        runner_up_ratio=runner_up,
        parity=parity,
        parity_ratio=max(even, odd),
        other_parity_ratio=min(even, odd),
    )


def combine_time_diversity(values: np.ndarray, diversity: TimeDiversity) -> np.ndarray:
    """Fold a DX/RX diversity pair into one character stream.

    Keeps the first (DX) copy unless it fails the constant-ratio check, in
    which case the second (RX) copy is substituted -- the actual error
    correction SITOR-B mode B provides.
    """
    start, spacing = diversity.parity, diversity.spacing
    count = (values.size - start - spacing + 1) // 2
    if count <= 0:
        return values
    dx = values[start: start + 2 * count: 2]
    rx = values[start + spacing: start + spacing + 2 * count: 2]
    return np.where(_MARKS[dx] == MARKS_PER_CODEWORD, dx, rx)


@dataclass(frozen=True)
class RepOverlap:
    """An ARQ block overlap: N characters repeated across each REP symbol."""

    length: int
    rep_count: int
    match_ratio: float

    @property
    def confident(self) -> bool:
        return self.match_ratio >= _CONFIDENT_REP_MATCH and self.rep_count >= _MIN_REPS


def find_rep_overlap(values: np.ndarray) -> RepOverlap | None:
    """Measure whether each REP symbol is straddled by a repeated character run.

    ``POL-ARQ.bin`` sends a REP (0x66) control symbol every 16-17 characters and
    repeats the four characters preceding it immediately afterwards -- the
    retransmission overlap of its ARQ block. This measures the overlap length
    directly instead of assuming one: only the true length matches, because the
    whole run has to be equal.
    """
    positions = np.flatnonzero(values == REP)
    best: RepOverlap | None = None
    for length in range(1, _MAX_REP_OVERLAP + 1):
        usable = positions[(positions >= length) & (positions + length < values.size)]
        if usable.size < _MIN_REPS:
            continue
        hits = sum(
            1
            for p in usable
            if np.array_equal(values[p - length: p], values[p + 1: p + 1 + length])
        )
        candidate = RepOverlap(length, int(usable.size), hits / usable.size)
        if best is None or candidate.match_ratio > best.match_ratio:
            best = candidate
    return best


def strip_rep_overlap(values: np.ndarray, overlap: RepOverlap) -> np.ndarray:
    """Drop each REP symbol and the repeated run that follows it."""
    keep = np.ones(values.size, dtype=bool)
    length = overlap.length
    for p in np.flatnonzero(values == REP):
        if p < length or p + 1 + length > values.size:
            continue
        if np.array_equal(values[p - length: p], values[p + 1: p + 1 + length]):
            keep[p: p + 1 + length] = False
    return values[keep]


class Ccir476Decoder(Decoder):
    """Base decoder; subclassed/registered under each mode's display name."""

    name = "CCIR476"
    description = "7-unit constant-ratio code (SITOR/AMTOR/PACTOR-I alphabet)"

    def decode(self, bits: np.ndarray, block=None) -> DecodeResult:
        alignment = find_codeword_alignment(bits)
        values = alignment.apply(bits)
        diversity = find_time_diversity(values)
        if diversity is not None and diversity.confident:
            values = combine_time_diversity(values, diversity)
        else:
            diversity = None
        overlap = find_rep_overlap(values)
        if overlap is not None and overlap.confident:
            values = strip_rep_overlap(values, overlap)
        else:
            overlap = None
        n_symbols = int(values.size)

        chars = []
        figs_mode = False
        errors = 0
        for v in values:
            v = int(v)
            if _mark_count(v) != 4:
                chars.append("#")  # ratio violation: detected transmission error
                errors += 1
                continue
            if v == LTRS_SHIFT:
                figs_mode = False
                continue
            if v == FIGS_SHIFT:
                figs_mode = True
                continue
            if v in _SPECIAL:
                continue  # phasing/ARQ control signal, not a character
            table = _FIGURES if figs_mode else _LETTERS
            char = table.get(v)
            if char is None:
                chars.append("?")
                errors += 1
            else:
                chars.append(char)
        confidence_word = "confident" if alignment.confident else "weak/inconclusive"
        notes = (
            f"alignment offset={alignment.offset} orientation={alignment.orientation} "
            f"polarity={'inverted' if alignment.inverted else 'normal'} "
            f"4-mark rate={alignment.ratio:.3f} over {alignment.codeword_count} "
            f"codewords vs {RANDOM_RATIO:.3f} random baseline ({confidence_word})"
        )
        if diversity is not None:
            notes += (
                f"; SITOR-B-style time diversity applied: spacing={diversity.spacing} "
                f"match rate={diversity.match_ratio:.3f} (next best spacing "
                f"{diversity.runner_up_ratio:.3f}, parity {diversity.parity} "
                f"{diversity.parity_ratio:.3f} vs {diversity.other_parity_ratio:.3f})"
            )
        if overlap is not None:
            notes += (
                f"; ARQ REP block overlap removed: {overlap.length} characters "
                f"repeated across {overlap.match_ratio:.3f} of {overlap.rep_count} "
                "REP symbols"
            )
        notes += "; alphabet layer only, remaining mode-specific ARQ framing not decoded"
        return DecodeResult(
            text="".join(chars),
            symbol_count=n_symbols,
            error_count=errors,
            notes=notes,
        )


# Every named mode below shares the CCIR 476 alphabet (see module docstring).
_MODE_NAMES = [
    "PACTOR-I", "PACTOR FEC", "PACTOR LONG",
    "SITOR-A", "SITOR-FEC", "SITOR-B (7x22)", "SITOR-B (7x29)",
    "ARQ-E", "ARQ-E3", "ARQ-M2-342/242", "FEC-A",
    "SI-ARQ 70", "SI-ARQ 84", "SI ARQ",
    "POL-ARQ", "SWED ARQ", "DUP-ARQ 1", "DUP-ARQ 2", "DUP-ARQ",
    "ARQ-6/90", "ARQ (7x22) 1", "ARQ (7x22) 2", "ARQ1a-1", "ARQ1a-2",
    # ITA-3-1/ITA-3-2: Signals/ARQ-M2-342_242 (ITA-3-1).bin and (ITA-3-2).bin
    # confirm these are alternate names for the ARQ-M2-342/242 mode above.
    "ITA-3-1", "ITA-3-2",
]

# Names whose alignment is confirmed against their own bundled capture. Only the
# two that decode to readable plaintext are "high"; ARQ-E3's 0.869 constant-ratio
# rate is a confirmed codeword grid but its text is not yet readable.
_CONFIRMED = {
    "SITOR-B (7x22)": (
        "high",
        "character alignment and text",
        "ITU-R M.476 alphabet confirmed against bundled Sitor-B (7x22).bin: searched "
        "alignment scores 0.9985 4-mark rate, spacing-5 DX/RX diversity lifts it to "
        "1.000 and yields readable Polish text",
    ),
    "POL-ARQ": (
        "high",
        "character alignment and text",
        "ITU-R M.476 alphabet confirmed against bundled POL-ARQ.bin: searched "
        "alignment scores 0.9869 4-mark rate and, after removing the 4-character "
        "REP block overlap measured on 0.94 of 339 REP symbols, yields readable "
        "Polish text",
    ),
    "ARQ-E3": (
        "medium",
        "character alignment",
        "ITU-R M.476 alphabet aligned against bundled ARQ-E3.bin at a 0.8693 4-mark "
        "rate (0.9947 on alternate characters); text not yet readable",
    ),
}

for _name in _MODE_NAMES:
    _decoder = Ccir476Decoder()
    _decoder.name = _name
    _decoder.description = f"{_name}: CCIR 476 alphabet (see Ccir476Decoder docstring)"
    if _name in _CONFIRMED:
        _decoder.confidence, _decoder.layer, _decoder.provenance = _CONFIRMED[_name]
    register(_decoder)

register(Ccir476Decoder())  # also register under the generic "CCIR476" name
