"""HFDL (HF Data Link, ARINC 635) detection and decoding.

Realtime-friendly port of the HFDL PHY, parameterized against
dumphfdl (szpajder/dumphfdl, GPL) and validated end-to-end on a
synthetic burst (see tests/test_hfdl.py). Ground-station system table
parsing mirrors PC-HFDL's configs/pchfdl.dat (version 51 layout,
byte-verified).

PHY facts (all verified from dumphfdl src/hfdl.c / src/hfdl.h):
- Symbol rate: 1800 baud in EVERY mode. The four bit rates come from
  PSK order + FEC rate, not baud:
      300 bps = BPSK, code_rate 1/4
      600 bps = BPSK, code_rate 1/2
     1200 bps = QPSK, code_rate 1/2
     1800 bps = 8PSK, code_rate 1/2
- SSB subcarrier offset: +1440 Hz above the tuned HF carrier.
- Burst layout (symbols):
      PREKEY(448) + A(127) + A(127) + M1(127) + M2(15) + 9*T(15)
      + 72 (single slot) or 168 (double) x [30 data + 15 T]
- A = 127-bit m-sequence sent twice; M1 = circular shift of a 127-bit
  base sequence identifying one of 8 modes; T = 15-symbol BPSK
  training, sent 9x in the preamble and after every 30 data symbols.
- Coding: 15-bit LFSR scrambler (phase flip per symbol), 40-row block
  interleaver, V27 (K=7, r=1/2) convolutional code, CRC-16-CCITT FCS.

Only numpy + scipy are required.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pathlib import Path
from scipy import signal as sp

# ------------------------------------------------------------------
# PHY constants (dumphfdl src/hfdl.h, src/hfdl.c)
# ------------------------------------------------------------------
SYMBOL_RATE = 1800          # baud, all modes
SSB_CARRIER_OFFSET_HZ = 1440
PREKEY_LEN = 448
A_LEN = 127
M1_LEN = 127
M2_LEN = 15
T_LEN = 15
PREAMBLE_LEN = 2 * A_LEN + M1_LEN + M2_LEN + 9 * T_LEN  # 531
DATA_FRAME_LEN = 30         # data symbols per segment
DATA_FRAME_CNT_SINGLE = 72  # 1.8 s interleaver
DATA_FRAME_CNT_DOUBLE = 168 # 4.2 s interleaver
DEINT_ROWS = 40
DEINT_POP_ROW_SHIFT = 9
DEINT_PUSH_SHIFT_SINGLE = 17
DEINT_PUSH_SHIFT_DOUBLE = 23

# hfdl_frame_params[mode]: (scheme_arity, segments, code_rate, push_shift)
# modes 0..7 = 300S 600S 1200S 1800S 300D 600D 1200D 1800D
FRAME_PARAMS = {
    0: (1, DATA_FRAME_CNT_SINGLE, 4, 17),
    1: (1, DATA_FRAME_CNT_SINGLE, 2, 17),
    2: (2, DATA_FRAME_CNT_SINGLE, 2, 17),
    3: (3, DATA_FRAME_CNT_SINGLE, 2, 17),
    4: (1, DATA_FRAME_CNT_DOUBLE, 4, 23),
    5: (1, DATA_FRAME_CNT_DOUBLE, 2, 23),
    6: (2, DATA_FRAME_CNT_DOUBLE, 2, 23),
    7: (3, DATA_FRAME_CNT_DOUBLE, 2, 23),
}
MODE_LABEL = {
    0: "300 bps 1.8 sec", 1: "600 bps 1.8 sec", 2: "1200 bps 1.8 sec",
    3: "1800 bps 1.8 sec", 4: "300 bps 4.2 sec", 5: "600 bps 4.2 sec",
    6: "1200 bps 4.2 sec", 7: "1800 bps 4.2 sec",
}

# A sequence: 127-bit m-sequence (16 octets, first 127 bits), MSB first
A_OCTETS = [0b01011011, 0b10111100, 0b01110100, 0b01010111, 0b00000011,
            0b11011001, 0b10001001, 0b00111001, 0b11110010, 0b00001000,
            0b11010101, 0b00110110, 0b10010100, 0b00101100, 0b00110010,
            0b11111110]
A_BITS = []
for _o in A_OCTETS:
    for _k in range(7, -1, -1):
        A_BITS.append((_o >> _k) & 1)
A_BITS = A_BITS[:A_LEN]

# M1 base sequence (127 bits)
M1_BASE = [0,1,1,1,0,1,1,0,1,1,1,1,0,1,0,0,0,1,0,1,1,0,0,
           1,0,1,1,1,1,1,0,0,0,1,0,0,0,0,0,0,1,1,0,0,1,1,0,1,1,
           0,0,0,1,1,1,0,0,1,1,1,0,1,0,1,1,1,0,0,0,0,1,0,0,1,1,
           0,0,0,0,0,1,0,1,0,1,0,1,1,0,1,0,0,1,0,0,1,0,1,0,0,1,
           1,1,1,0,0,1,0,0,0,1,1,0,1,0,1,0,0,0,0,1,1,1,1,1,1,1]
M_SHIFTS = {0: 72, 1: 82, 2: 113, 3: 123, 4: 61, 5: 103, 6: 93, 7: 9}

# T training sequence (BPSK, used at all data rates)
T_SEQ = np.array([1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1],
                 dtype=float)

# dumphfdl matched filter taps (19 = SPS*3*2+1 at SPS=3)
MF_TAPS = np.array([
    -0.0170974647427123, 0.01148231492068473, 0.03138375667422348,
    0.009454398851680437, -0.04161644170893816, -0.06451564801420356,
    -0.005495792933327306, 0.1316404671361545, 0.2759693160697777,
    0.3375901874933208, 0.2759693160697777, 0.1316404671361545,
    -0.005495792933327306, -0.06451564801420356, -0.04161644170893816,
    0.009454398851680437, 0.03138375667422348, 0.01148231492068473,
    -0.0170974647427123])

# ------------------------------------------------------------------
# System table (PC-HFDL configs/pchfdl.dat, version >= 51)
# ------------------------------------------------------------------
_STATION_NAMES_51 = [
    "UNKNOWN", "SAN FRANCISCO - CALIFORNIA", "MOLOKAI - HAWAII",
    "REYKJAVIK - ICELAND", "RIVERHEAD - NEW YORK", "AUCKLAND - NZ",
    "HAT YAI - THAILAND", "SHANNON - IRELAND",
    "JOHANNESBURG - SOUTH AFRICA", "BARROW - ALASKA", "MUAN - SOUTH KOREA",
    "UNKNOWN12", "SANTA CRUZ - BOLIVIA", "KRASNOYARSK - RUSSIA",
    "AL MUHARRAQ - BAHRAIN", "AGANA - GUAM", "CANARIAS - SPAIN"]


@dataclass
class Station:
    gs_id: int
    name: str
    frequencies: list = field(default_factory=list)  # [(hz, slot), ...]


def parse_system_table_dat(data: bytes) -> dict:
    """Parse PC-HFDL's configs/pchfdl.dat (version 51 record layout:
    16-byte header, then fixed 0xb8-byte records: (gs_id, 0, lon, lat,
    nf, 0) + up to 20 (freq_hz, slot) pairs)."""
    import struct
    if len(data) < 0x10:
        raise ValueError("system table too short")
    hdr = struct.unpack_from("<4i", data, 0)
    version = hdr[2]
    stride = 0xB8
    out = {}
    for k in range(hdr[3]):
        off = 0x10 + k * stride
        if off + 24 > len(data):
            break
        gs_id, _, lon, lat, nf, _ = struct.unpack_from("<6i", data, off)
        if gs_id == 0 and nf == 0:
            continue
        raw = struct.unpack_from("<40i", data, off + 24)
        pairs = [(raw[2 * i], raw[2 * i + 1])
                 for i in range(nf) if raw[2 * i]]
        name = (_STATION_NAMES_51[gs_id]
                if gs_id < len(_STATION_NAMES_51) else f"GS{gs_id}")
        out[gs_id] = Station(gs_id, name, pairs)
    return out


# ------------------------------------------------------------------
# Scrambler / interleaver / FEC
# ------------------------------------------------------------------
def scrambler_bits(n: int, poly: int = 0x4001, seed: int = 0x4D4B) -> list:
    """15-bit LFSR, liquid-dsp msequence convention (poly 0x4001 /
    seed 0x4D4B in current liquid; equals 0x8002/0x6959 bit-reversed).
    Returns the phase-flip bit per data symbol (seq_len 120 reset)."""
    v = seed
    out = []
    for _ in range(n):
        if len(out) % 120 == 0 and len(out):
            v = seed
        b = bin(v & poly).count("1") & 1
        v = ((v << 1) & 0x7FFF) | b
        out.append(b)
    return out


def deinterleave(bits, mode: int):
    """Exact dumphfdl deinterleaver (push/pop pattern)."""
    arity, segs, rate, push = FRAME_PARAMS[mode]
    cols = segs * DATA_FRAME_LEN * arity // DEINT_ROWS
    table = np.zeros((DEINT_ROWS, cols), dtype=np.int8)
    row = col = 0
    for v in bits:
        table[row][col] = v
        row += 1
        if row == DEINT_ROWS:
            row = 0
            col += 1
        col -= push
        if col < 0:
            col += cols
    out = []
    row = col = 0
    for _ in range(DEINT_ROWS * cols):
        out.append(table[row][col])
        row = (row + DEINT_POP_ROW_SHIFT) % DEINT_ROWS
        if row == 0:
            col += 1
    return np.array(out)


_PARTAB = []
for _i in range(256):
    _c = 0
    _t = _i
    for _ in range(8):
        _c += _t & 1
        _t >>= 1
    _PARTAB.append(_c & 1)

BRANCHTAB = [
    [255 if _PARTAB[(2 * s) & 0x6D] else 0 for s in range(32)],
    [255 if _PARTAB[(2 * s) & 0x4F] else 0 for s in range(32)],
]


def viterbi27(syms, nbits: int, neq: int = 6) -> bytes:
    """V27 (K=7, r=1/2, polys 0x6d/0x4f) Viterbi decoder matching
    libfec's viterbi27_port.c bit-for-bit (including the +6 decision
    lookahead and MSB-first byte packing)."""
    old = [63] * 64
    old[0] = 0
    decisions = []
    for n in range(nbits):
        s0, s1 = syms[2 * n], syms[2 * n + 1]
        new = [0] * 64
        w = [0, 0]
        for i in range(32):
            metric = (BRANCHTAB[0][i] ^ s0) + (BRANCHTAB[1][i] ^ s1)
            m0 = old[i] + metric
            m1 = old[i + 32] + (510 - metric)
            d0 = (m0 - m1) > 0
            new[2 * i] = m1 if d0 else m0
            w[i // 16] |= (1 if d0 else 0) << ((2 * i) % 32)
            m0 -= (metric + metric - 510)
            m1 += (metric + metric - 510)
            d1 = (m0 - m1) > 0
            new[2 * i + 1] = m1 if d1 else m0
            w[i // 16] |= (1 if d1 else 0) << ((2 * i + 1) % 32)
        decisions.append(w)
        old = list(new)
    es = 0
    out = bytearray((nbits + 7) // 8)
    for n in range(nbits - 1, -1, -1):
        blk = decisions[n + neq] if n + neq < len(decisions) else (0, 0)
        idx = es >> 2
        k = (blk[idx // 32] >> (idx % 32)) & 1
        es = (es >> 1) | (k << 7)
        out[n >> 3] = es & 0xFF
    return bytes(out)


_CRC_TABLE = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = (_c >> 1) ^ 0x8408 if _c & 1 else _c >> 1
    _CRC_TABLE.append(_c)


def crc16_ccitt(data: bytes) -> int:
    """CRC-16-CCITT as implemented by dumphfdl (src/crc.c): REFLECTED
    table (poly 0x8408), init 0xFFFF, final XOR 0xFFFF (src/pdu.c
    hfdl_pdu_fcs_check). FCS stored little-endian after the data."""
    crc = 0xFFFF
    for byte in data:
        crc = (crc >> 8) ^ _CRC_TABLE[(crc ^ byte) & 0xFF]
    return crc ^ 0xFFFF


# ------------------------------------------------------------------
# Demodulation
# ------------------------------------------------------------------
@dataclass
class HFDLFrame:
    mode: int                 # 0..7
    label: str                # human-readable mode string
    symbols: int              # data symbol count
    bytes_out: bytes          # viterbi output (before PDU parse)
    crc_ok: bool
    fcs_offset: int = 14      # where the FCS was checked (best effort)
    payload: bytes = b""


def _liquid_psk4(u: complex) -> int:
    """liquid-dsp PSK4 hard symbol: 0=0deg, 1=90, 2=270, 3=180.
    Bits = binary of the symbol index (verified against liquid's
    modem_demodulate_soft)."""
    a = np.angle(u) * 180 / np.pi
    if -45 <= a < 45:
        return 0
    if 45 <= a < 135:
        return 1
    if a >= 135 or a < -135:
        return 3
    return 2


def _liquid_psk8(u: complex) -> int:
    """liquid-dsp PSK8 hard symbol (gray-coded 8PSK, symbol 0 at 0
    degrees)."""
    a = (np.angle(u) + 2 * np.pi) % (2 * np.pi)
    k = int(round(a / (np.pi / 4))) % 8
    return k ^ (k >> 1)


def _demod_symbol(u: complex, arity: int) -> list:
    if arity == 1:
        return [1 if u.real > 0 else 0]
    if arity == 2:
        s = _liquid_psk4(u)
        return [(s >> 1) & 1, s & 1]
    s = _liquid_psk8(u)
    return [(s >> 2) & 1, (s >> 1) & 1, s & 1]


def find_frame_symbols(samples, sample_rate: float):
    """Locate an HFDL burst in complex baseband and return
    (symbol_stream, mode) or None. samples: complex numpy array
    (analytic, subcarrier near DC)."""
    # resample to 3 SPS
    target = SYMBOL_RATE * 3
    if abs(sample_rate - target) > 1:
        g = np.gcd(int(sample_rate), int(target))
        bb = sp.resample_poly(samples, int(target // g),
                              int(sample_rate // g))
    else:
        bb = samples
    mf = np.convolve(bb, MF_TAPS, "same")
    idx = np.arange(0, len(mf) - 1, 3).astype(int)
    sym = mf[idx]
    norm = sym / (np.abs(sym) + 1e-9)

    A = np.array([1.0 if b else -1.0 for b in A_BITS])
    dA = A[:-1] * A[1:]
    dn = norm[:-1] * np.conj(norm[1:])
    dn = dn / (np.abs(dn) + 1e-9)

    # find the two A sequences via differential correlation (robust to
    # static phase offset)
    hits = []
    for st in range(0, len(dn) - A_LEN):
        v = np.mean(dA * dn[st:st + A_LEN - 1])
        if abs(v) > 0.6:
            if hits and st - hits[-1][1] < 50:
                continue
            hits.append((float(abs(v)), st))
    if not hits:
        return None
    # A2 = the second hit (should be A1 + 127)
    a2 = None
    for _, st in hits:
        if a2 is not None and st - a2 == A_LEN:
            a2 = st
            break
        if a2 is None:
            a2 = st
    if a2 is None:
        return None
    m1_start = a2 + A_LEN
    if m1_start + M1_LEN + M2_LEN + 9 * T_LEN > len(norm):
        return None

    # M1 mode detection
    m1_seg = norm[m1_start:m1_start + M1_LEN]
    rot = np.mean(A * norm[a2:a2 + A_LEN])
    m1_seg = m1_seg * np.exp(-1j * np.angle(rot))
    best_mode, best_corr = None, 0.0
    for mode, sh in M_SHIFTS.items():
        ref = np.array([1.0 if M1_BASE[(sh + j) % 127] else -1.0
                        for j in range(M1_LEN)])
        v = abs(np.mean(ref * m1_seg))
        if v > best_corr:
            best_corr, best_mode = v, mode
    if best_mode is None or best_corr < 0.5:
        return None

    data_start = m1_start + M1_LEN + M2_LEN + 9 * T_LEN
    arity, segs, rate, push = FRAME_PARAMS[best_mode]
    need = segs * (DATA_FRAME_LEN + T_LEN)
    if data_start + need > len(norm):
        return None

    d = norm * np.exp(-1j * np.angle(rot))
    out_syms = []
    pos = data_start
    prev_ph = 0.0
    for _ in range(segs):
        tseg = d[pos + DATA_FRAME_LEN:pos + DATA_FRAME_LEN + T_LEN]
        tt = tseg / (np.abs(tseg) + 1e-9)
        phT = np.angle(np.mean(T_SEQ * tt))
        seg = d[pos:pos + DATA_FRAME_LEN] * np.exp(-1j * phT)
        for u in seg:
            out_syms.append(u / (np.abs(u) + 1e-9))
        pos += DATA_FRAME_LEN + T_LEN
    return out_syms, best_mode


def decode_frame(samples, sample_rate: float) -> HFDLFrame | None:
    """Find and decode one HFDL frame from complex baseband samples.
    Returns None if no frame is found."""
    found = find_frame_symbols(samples, sample_rate)
    if found is None:
        return None
    syms, mode = found
    arity, segs, rate, push = FRAME_PARAMS[mode]
    scr = scrambler_bits(len(syms))
    bits = []
    for i, u in enumerate(syms):
        uu = -u if scr[i] else u
        bits.extend(_demod_symbol(uu, arity))
    deint = deinterleave(bits, mode)
    if rate == 4:
        # 1/4 rate: chips sent twice; average pairs
        merged = [min((int(deint[2 * i]) + int(deint[2 * i + 1]) + 1) // 2, 1)
                  for i in range(len(deint) // 2)]
        vsoft = [255 if b else 0 for b in merged]
    else:
        vsoft = [255 if b else 0 for b in deint]
    # NOTE: no flush append - dumphfdl reads zero decision blocks for
    # the last K-1=6 steps, so the final 6 decoded bits are the
    # convolutional tail, not user data (harmless for PDU parsing).
    vout = viterbi27(vsoft, len(vsoft) // 2)
    # REVERSE_BYTE (dumphfdl applies it before PDU dispatch)
    rev = bytes(int(f"{b:08b}"[::-1], 2) for b in vout)
    crc_ok = crc16_ccitt(rev[:64]) == \
        int.from_bytes(rev[64:66], "little") if len(rev) >= 66 else False
    return HFDLFrame(mode=mode, label=MODE_LABEL[mode],
                     symbols=len(syms), bytes_out=rev, crc_ok=crc_ok)


def decode_file(path: str, sample_rate: float | None = None) -> list:
    """Decode all HFDL frames in an audio file. Supports WAV via the
    wave module; .cf32 (interleaved float32 complex, I first) and
    .f32 (real float32) are read headerless at `sample_rate`
    (default 8000 Hz — the HFDL audio convention)."""
    suffix = Path(path).suffix.lower()
    if suffix == ".cf32":
        x = np.fromfile(path, dtype=np.complex64).real.astype(np.float64)
        sr = sample_rate or 8000.0
    elif suffix == ".f32":
        x = np.fromfile(path, dtype="<f4").astype(np.float64)
        sr = sample_rate or 8000.0
    else:
        import wave
        w = wave.open(path)
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()),
                          dtype="<i2") / 32768.0
        w.close()
        if sample_rate is not None:
            sr = sample_rate
    # measure the actual subcarrier from the strongest 0.2 s window
    f, P = sp.welch(x, sr, nperseg=1024)
    tone = f[int(np.argmax(P))]
    if not (800 < tone < 2600):
        tone = SSB_CARRIER_OFFSET_HZ
    a = sp.hilbert(x)
    bb = a * np.exp(-2j * np.pi * tone * np.arange(len(x)) / sr)
    frame = decode_frame(bb, sr)
    return [frame] if frame else []
