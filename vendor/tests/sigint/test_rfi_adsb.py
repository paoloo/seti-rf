"""ADS-B / Mode S: CRC verified against a canonical real message, decode
verified end-to-end against a synthetic PPM-encoded burst built from it."""
import numpy as np

from sigint.rfi import adsb

# A widely-cited real DF17 (extended squitter) example message: DF=17,
# CA=5, ICAO 4840D6, ME 202CC371C32CE0, PI 576098. Used across ADS-B
# literature/tooling (e.g. Junzi Sun's pyModeS test fixtures) as a known-
# good decode target.
CANONICAL_HEX = "8D4840D6202CC371C32CE0576098"


def _bits_of(hex_msg: str) -> np.ndarray:
    raw = bytes.fromhex(hex_msg)
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8))


def test_crc24_is_zero_over_a_genuinely_valid_df17_message():
    """DF17's parity is a pure CRC of the other 88 bits (no address XOR):
    CRC-ing the whole 112-bit message (data + its own parity) must give a
    zero remainder for a message that was valid to begin with."""
    assert adsb.crc24(_bits_of(CANONICAL_HEX)) == 0


def test_crc24_flags_a_corrupted_message():
    bits = _bits_of(CANONICAL_HEX)
    bits[40] ^= 1   # flip one data bit
    assert adsb.crc24(bits) != 0


def _synthesize_burst(hex_msg: str, sps: int, lead_in: int = 20,
                      amplitude: float = 1.0) -> np.ndarray:
    """PPM-encode `hex_msg` behind the standard 8 us preamble, at `sps`
    samples per microsecond, as an AM envelope (0/1 pulses)."""
    bits = _bits_of(hex_msg)
    half = max(1, sps // 2)
    preamble = np.zeros(int(round(adsb.PREAMBLE_US * sps)))
    for start_us in (0.0, 1.0, 3.5, 4.5):
        i = int(round(start_us * sps))
        preamble[i:i + half] = amplitude
    data = np.zeros(len(bits) * sps)
    for i, b in enumerate(bits):
        base = i * sps
        if b:
            data[base:base + half] = amplitude
        else:
            data[base + half:base + sps] = amplitude
    return np.concatenate([np.zeros(lead_in), preamble, data,
                           np.zeros(lead_in)])


def test_decode_recovers_the_canonical_message_from_a_synthetic_burst():
    sps = 4
    sr = adsb.BIT_RATE_HZ * sps
    x = _synthesize_burst(CANONICAL_HEX, sps)
    rng = np.random.default_rng(0)
    x = x + rng.normal(0.0, 0.05, len(x))   # a little receiver noise
    msgs = adsb.decode(x, sr)
    assert len(msgs) == 1
    m = msgs[0]
    assert m.df == 17
    assert m.crc_ok is True
    assert m.bytes_out.hex().upper() == CANONICAL_HEX[:22]  # 88 data bits


def test_decode_finds_nothing_in_plain_noise():
    rng = np.random.default_rng(2)
    x = rng.normal(0.3, 0.1, 20000)
    msgs = adsb.decode(x, adsb.BIT_RATE_HZ * 4)
    assert msgs == ()


def test_decode_accepts_a_zero_baseline_burst():
    x = _synthesize_burst(CANONICAL_HEX, 4)
    messages = adsb.decode(x, 4_000_000)
    assert len(messages) == 1
    assert messages[0].bytes_out.hex().upper() == CANONICAL_HEX[:22]
