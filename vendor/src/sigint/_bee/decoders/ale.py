"""MIL-STD-188-141A Automatic Link Establishment (ALE) word decoder (Tier B/C).

Each 24-bit ALE word (3-bit preamble + 21-bit data) is split into two 12-bit
halves, each independently protected by extended Golay(24,12) (see
``bee.core.golay``, corrects up to 3 bit errors per half), producing two
24-bit codewords A and B.

Bit interleaving and the B-codeword inversion are now taken from the primary
public standard rather than assumed. MIL-STD-188-141B Appendix A, section
A.5.2.2.3 and Figure A-10 ("Word bit coding and interleaving") state: "The
basic word bits W1 (MSB) through W24 (LSB), and resultant Golay FEC bits G1
through G24 (with G13 through G24 inverted), shall be interleaved... and they
shall be transmitted A1, B1, A2, B2... A24, B24, S49" -- i.e. the two 24-bit
Golay codewords are interleaved bit-by-bit (not sent as two back-to-back
24-bit blocks), the 49th bit is a fixed stuff bit (value 0, ignored on
receive), and codeword B's own check bits (the second half's Golay parity,
labeled G13-G24 in the standard's continuous numbering) are transmitted
inverted. Source: public copy of MIL-STD-188-141B, retrieved 2026-08-14 from
http://hflink.com/standards/ALE_standard_188_141B.pdf (linked from
https://www.sigidwiki.com/wiki/Automatic_Link_Establishment_(2G_ALE) on the
Signal Identification Guide). MIL-STD-188-141A itself is ITAR/export
controlled and wasn't consulted; 141B's word coding/interleaving section is
believed unchanged from 141A for this base (non-3G/4G) word format, but that
continuity is not independently confirmed.

Applying this interleaving to the bundled ``ALE.BIN`` capture raises the
best both-halves-Golay-decodable rate from ~0.39 (previous back-to-back
model, effectively at the ~0.32-0.39 random-chance baseline for two
independent 24-bit values landing within Golay's distance-3 correction ball)
to 0.518 at a standout offset under the byte-reversed orientation --
meaningfully above neighboring offsets and above the random baseline, but not
a clean 100%-style confirmation. The standard additionally specifies that
each 49-bit word is sent 3x redundantly with a 2/3 bit-level majority vote
before Golay decoding (A.5.2.2.4); this decoder does not yet implement that
majority-vote step, which real captures need for full recovery -- so this
remains a real, standard-backed improvement over the previous
back-to-back-blocks assumption, not a verified decode.

The 3-bit preamble is commonly documented (ham-radio ALE references) as
selecting one of 8 word types (DATA, THRU, TO, TWAS, TIS, CMD, REP, ORDER),
which MIL-STD-188-141B Table A-VIII confirms; "Type N" is still reported
numerically here rather than asserting names pending a dedicated pass over
that table.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sigint._bee.core.golay import decode as golay_decode
from sigint._bee.decoders.base import DecodeResult, Decoder
from sigint._bee.decoders.registry import register

WORD_BITS = 49  # 2x24 interleaved Golay bits + 1 stuff bit (MIL-STD-188-141B A.5.2.2.3)


def _reverse_each_byte(bits: np.ndarray) -> np.ndarray:
    complete = len(bits) // 8 * 8
    result = bits.copy()
    if complete:
        result[:complete] = bits[:complete].reshape(-1, 8)[:, ::-1].reshape(-1)
    return result


def _bits_to_int(bits) -> int:
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def _decode_word(word49: np.ndarray) -> tuple[int | None, bool, bool]:
    """Deinterleave/decode one 49-bit word; returns (word24, half1_fec, half2_fec)."""
    interleaved = word49[:48]
    a_bits = interleaved[0::2]  # A1..A24
    b_bits = interleaved[1::2]  # B1..B24
    a_value = _bits_to_int(a_bits)
    b_data = b_bits[:12]
    b_check = b_bits[12:] ^ 1  # G13-G24 transmitted inverted (Figure A-10)
    b_value = _bits_to_int(np.concatenate([b_data, b_check]))
    r1, r2 = golay_decode(a_value), golay_decode(b_value)
    if r1.data is None or r2.data is None:
        return None, r1.corrected, r2.corrected
    word24 = (r1.data << 12) | r2.data
    return word24, r1.corrected, r2.corrected


@dataclass(frozen=True)
class WordAlignment:
    offset: int
    orientation: str
    success_rate: float
    word_count: int


def find_word_alignment(bits: np.ndarray) -> WordAlignment:
    """Search phase offset and byte-bit orientation for the best Golay-decode rate."""
    best: WordAlignment | None = None
    for oriented, orientation in ((bits, "selected"), (_reverse_each_byte(bits), "byte-reversed")):
        for offset in range(WORD_BITS):
            n_words = (len(oriented) - offset) // WORD_BITS
            if n_words < 1:
                continue
            ok = 0
            for i in range(n_words):
                word = oriented[offset + i * WORD_BITS: offset + (i + 1) * WORD_BITS]
                word24, _, _ = _decode_word(word)
                if word24 is not None:
                    ok += 1
            rate = ok / n_words
            candidate = WordAlignment(offset, orientation, rate, n_words)
            if best is None or candidate.success_rate > best.success_rate:
                best = candidate
    assert best is not None
    return best


class AleDecoder(Decoder):
    name = "ALE"
    description = "MIL-STD-188-141B word: interleaved 2x Golay(24,12), searched alignment"

    def decode(self, bits: np.ndarray, block=None) -> DecodeResult:
        alignment = find_word_alignment(bits)
        oriented = bits if alignment.orientation == "selected" else _reverse_each_byte(bits)
        offset = alignment.offset
        n_words = (len(oriented) - offset) // WORD_BITS
        lines = []
        errors = 0
        for i in range(n_words):
            word = oriented[offset + i * WORD_BITS: offset + (i + 1) * WORD_BITS]
            word24, c1, c2 = _decode_word(word)
            if word24 is None:
                lines.append("[BAD] uncorrectable Golay error")
                errors += 1
                continue
            preamble = (word24 >> 21) & 0b111
            data21 = word24 & 0x1FFFFF
            chars = [chr((data21 >> shift) & 0x7F) for shift in (14, 7, 0)]
            text = "".join(c if 32 <= ord(c) < 127 else "." for c in chars)
            flag = "OK " if not (c1 or c2) else "FEC"
            lines.append(f"[{flag}] type={preamble} data=0x{data21:06X} ascii={text!r}")
        return DecodeResult(
            text="\n".join(lines),
            symbol_count=n_words,
            error_count=errors,
            notes=(f"alignment offset={offset} orientation={alignment.orientation} "
                   f"Golay-decodable={alignment.success_rate:.3f}; interleaving/inversion per "
                   "MIL-STD-188-141B A.5.2.2.3 (see module docstring); 3x redundant-word "
                   "majority voting is not yet implemented"),
        )


for _name in ("ALE", "ALE-3G"):
    _decoder = AleDecoder()
    _decoder.name = _name
    register(_decoder)
