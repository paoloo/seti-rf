"""Extended binary Golay(24,12) code: encode/decode with up to 3-bit correction.

Ported from a public, tested reference implementation (MIT-licensed):
https://github.com/notatuta/golay (golay.h) -- chosen over deriving the
parity matrix from scratch, since a transcription error in a hand-derived
matrix would silently corrupt every decode. The algorithm: a 24-bit
codeword is [12 data bits | 12 checksum bits]; on decode, the checksum is
recomputed from the received data bits and XORed against the received
checksum to get a syndrome, then a standard weight-based search (per
Golay's structure, any pattern of <=3 bit errors has a unique nearest
codeword) locates and corrects up to 3 flipped bits.
"""
from __future__ import annotations

from dataclasses import dataclass

_B_MATRIX = [
    0x9F1, 0x4FA, 0x27D, 0x93E, 0xC9D, 0xE4E,
    0xF25, 0xF92, 0x7C9, 0x3E6, 0x557, 0xAAB,
]


def _popcount(x: int) -> int:
    return bin(x).count("1")


def _parity(x: int) -> int:
    return _popcount(x) & 1


def _checksum_bits(x: int) -> int:
    y = 0
    for row in _B_MATRIX:
        y = (y << 1) | _parity(x & row)
    return y


def encode(data12: int) -> int:
    """12 data bits -> 24-bit codeword [data(12) | checksum(12)]."""
    data12 &= 0xFFF
    return (data12 << 12) | _checksum_bits(data12)


@dataclass
class GolayResult:
    data: int | None  # 12-bit decoded value, or None if uncorrectable
    corrected: bool


def decode(codeword24: int) -> GolayResult:
    """24-bit received codeword -> corrected 12 data bits, or None if uncorrectable."""
    received_data = (codeword24 >> 12) & 0xFFF
    received_checksum = codeword24 & 0xFFF
    expected_checksum = _checksum_bits(received_data)
    syndrome = expected_checksum ^ received_checksum
    weight = _popcount(syndrome)

    if weight <= 3:
        return GolayResult(data=received_data, corrected=weight > 0)

    for i in range(12):
        error_mask = 1 << (11 - i)
        coding_error = _B_MATRIX[i]
        if _popcount(syndrome ^ coding_error) <= 2:
            return GolayResult(data=received_data ^ error_mask, corrected=True)

    inverted_syndrome = _checksum_bits(syndrome)
    if _popcount(inverted_syndrome) <= 3:
        return GolayResult(data=received_data ^ inverted_syndrome, corrected=True)

    for i in range(12):
        coding_error = _B_MATRIX[i]
        if _popcount(inverted_syndrome ^ coding_error) <= 2:
            return GolayResult(data=received_data ^ inverted_syndrome ^ coding_error, corrected=True)

    return GolayResult(data=None, corrected=False)
