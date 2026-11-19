"""CW (Morse code): keying-timing detection and decode.

International Morse code, per ITU-R M.1677-1 (the current standard: a dash
is 3 units, intra-character gap 1 unit, inter-character gap 3 units,
inter-word gap 7 units -- verified against the ITU recommendation's own
table, not trusted from memory alone). Unlike every other waveform-level
probe in this package, CW carries no fixed symbol clock in the signal
itself -- a human keys it, so timing is inherently variable (operator
"fist", Farnsworth spacing that keys elements at one speed and spaces
words at another). :func:`decode` handles this by clustering the
observed on/off durations of the actual recording rather than assuming a
fixed unit length -- the same "measure it, don't assume it" discipline
:mod:`sigint.dsp`'s baud estimators use for keyed digital signals.

Self-rejecting: a recording with no on/off keying structure (too few
distinct activity segments) returns ``None`` rather than a low-confidence
guess.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..dsp import _analytic

# --------------------------------------------------------------------------
# The code table (ITU-R M.1677-1, cross-checked against the recommendation's
# own PDF; a handful of widely-used procedural prosigns are included too).
# --------------------------------------------------------------------------
MORSE_TABLE: dict[str, str] = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".",
    "F": "..-.", "G": "--.", "H": "....", "I": "..", "J": ".---",
    "K": "-.-", "L": ".-..", "M": "--", "N": "-.", "O": "---",
    "P": ".--.", "Q": "--.-", "R": ".-.", "S": "...", "T": "-",
    "U": "..-", "V": "...-", "W": ".--", "X": "-..-", "Y": "-.--",
    "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
    ".": ".-.-.-", ",": "--..--", ":": "---...", "?": "..--..",
    "'": ".----.", "-": "-....-", "/": "-..-.", "(": "-.--.",
    ")": "-.--.-", '"': ".-..-.", "=": "-...-", "+": ".-.-.",
    "@": ".--.-.",
    "<AR>": ".-.-.",    # end of message
    "<SK>": "...-.-",   # end of contact
    "<BT>": "-...-",    # new paragraph / break (same pattern as "=")
    "<KA>": "-.-.-",    # start of message / attention
    "<UNDERSTOOD>": "...-.",
    "<ERROR>": "........",
}
_REVERSE_TABLE: dict[str, str] = {v: k for k, v in MORSE_TABLE.items()}

DEFAULT_WPM = 20.0
DEFAULT_TONE_HZ = 700.0


# --------------------------------------------------------------------------
# Synthesis (ground truth for testing, mirrors sigint.dataset's discipline)
# --------------------------------------------------------------------------

def text_to_elements(text: str) -> str:
    """Uppercase text -> a string of '.'/'-'/' '/'/' (element/letter/word
    separators), skipping any character with no table entry."""
    words = text.upper().split(" ")
    out_words = []
    for word in words:
        letters = [MORSE_TABLE[c] for c in word if c in MORSE_TABLE]
        out_words.append(" ".join(letters))
    return " / ".join(out_words)


def generate(text: str, sample_rate: float, wpm: float = DEFAULT_WPM,
            tone_hz: float = DEFAULT_TONE_HZ, farnsworth_wpm: float | None = None,
            rise_ms: float = 4.0) -> np.ndarray:
    """Synthesize a CW recording of `text` at `wpm` (PARIS standard: one
    word = 50 units, so unit_s = 1.2 / wpm), OOK-keyed `tone_hz` tone.

    `farnsworth_wpm` (<= wpm), if given, keys elements at the full `wpm`
    speed but stretches inter-character/inter-word gaps to whatever a
    slower `farnsworth_wpm` would use -- the standard technique for
    teaching/copying practice (faster-sounding characters, easier-to-catch
    spacing). `rise_ms` ramps each element's edges (a hard on/off keyed
    tone splatters broadband key clicks; real keyers shape this).
    """
    unit_s = 1.2 / wpm
    gap_unit_s = 1.2 / farnsworth_wpm if farnsworth_wpm else unit_s
    elements = text_to_elements(text)

    chunks: list[np.ndarray] = []

    def tone(duration_s: float) -> np.ndarray:
        n = max(1, int(round(duration_s * sample_rate)))
        t = np.arange(n) / sample_rate
        x = np.sin(2 * np.pi * tone_hz * t)
        ramp_n = min(n // 2, max(1, int(round(rise_ms / 1000 * sample_rate))))
        if ramp_n > 0:
            ramp = np.linspace(0, 1, ramp_n)
            x[:ramp_n] *= ramp
            x[-ramp_n:] *= ramp[::-1]
        return x

    def silence(duration_s: float) -> np.ndarray:
        return np.zeros(max(1, int(round(duration_s * sample_rate))))

    for wi, word in enumerate(elements.split(" / ")):
        if wi > 0:
            chunks.append(silence(7 * gap_unit_s))
        for li, letter in enumerate(word.split(" ")):
            if li > 0:
                chunks.append(silence(3 * gap_unit_s))
            for ei, el in enumerate(letter):
                if ei > 0:
                    chunks.append(silence(unit_s))
                chunks.append(tone(unit_s if el == "." else 3 * unit_s))
    return np.concatenate(chunks) if chunks else np.zeros(0)


# --------------------------------------------------------------------------
# Detection + decode
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MorseMessage:
    text: str
    elements: str            # the raw '.'/'-'/' '/'/' string, for debugging
    wpm: float                # estimated element speed
    unrecognized: int = 0     # letter groups that didn't match the table


def _cluster_n(durations: np.ndarray, n: int,
              min_log_jump: float = np.log(1.8)) -> np.ndarray:
    """Split positive durations into up to `n` groups (0=shortest, ...) by
    the `n - 1` biggest ratio jumps in the sorted, log-scaled values --
    e.g. dot/dash on-durations (n=2) or intra-/inter-character/inter-word
    gaps (n=3). A candidate jump only counts as a real category boundary
    if it clears `min_log_jump` (default log(1.8), an ~80% relative step
    -- comfortably above realistic keying jitter, comfortably below the
    >=2.3x ratio between any two of Morse's four real timing categories:
    1/3/7 unit gaps, 1/3 unit elements); smaller jumps are just noise
    inside one population and must not fabricate an extra category. With
    fewer than `n` real categories present, the unused top group(s) are
    simply empty -- every value still gets a group."""
    n = max(1, min(n, len(durations)))
    logs = np.log(durations + 1e-12)
    order = np.argsort(logs)
    sorted_logs = logs[order]
    jumps = np.diff(sorted_logs)
    candidates = np.argsort(jumps)[::-1]
    split_positions = sorted(int(i) for i in candidates[:n - 1]
                             if jumps[i] >= min_log_jump)
    splits = [float(np.exp((sorted_logs[i] + sorted_logs[i + 1]) / 2))
             for i in split_positions]
    groups = np.zeros(len(durations), dtype=int)
    for s in splits:
        groups += (durations > s)
    return groups


def _cluster_two(durations: np.ndarray) -> tuple[np.ndarray, float]:
    """Two-group convenience wrapper over :func:`_cluster_n`; returns
    (is_long, split) with split = 2x the max duration when everything
    falls in one group (nothing to split)."""
    groups = _cluster_n(durations, 2)
    is_long = groups > 0
    split = float(durations[~is_long].max()) if (~is_long).any() else float(durations.max() * 2)
    return is_long, split


@dataclass(frozen=True)
class _Seg:
    start_s: float
    end_s: float


def _otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's method: the threshold that maximizes between-class variance
    of a bimodal histogram. Unlike :func:`_cluster_n`'s biggest-jump split
    (which needs a genuinely empty gap between populations), this handles
    a real, noisy, continuous energy distribution -- the on/off keying
    populations overlap somewhat once there's receiver noise, so there is
    no empty gap to find, only two humps to separate."""
    hist, edges = np.histogram(values, bins=bins)
    hist = hist.astype(np.float64)
    total = hist.sum()
    bin_centers = (edges[:-1] + edges[1:]) / 2
    sum_all = float(np.sum(hist * bin_centers))
    w_b = np.cumsum(hist)
    sum_b = np.cumsum(hist * bin_centers)
    w_f = total - w_b
    valid = (w_b > 0) & (w_f > 0)
    m_b = np.where(valid, sum_b / np.maximum(w_b, 1), 0.0)
    m_f = np.where(valid, (sum_all - sum_b) / np.maximum(w_f, 1), 0.0)
    var_between = np.where(valid, w_b * w_f * (m_b - m_f) ** 2, -1.0)
    return float(bin_centers[int(np.argmax(var_between))])


def _keying_segments(env: np.ndarray, sample_rate: float,
                     block_ms: float, min_contrast_db: float = 6.0) -> list[_Seg]:
    """On/off run-length segmentation of a keyed envelope.

    Deliberately different from :func:`sigint.detect.detect_activity` in
    two ways. First, no merging: that function's 1-block-lookahead
    hysteresis merge exists to keep a single burst from fragmenting on
    energy noise, which is the wrong behavior here -- in CW the OFF
    durations between elements/characters/words are exactly as meaningful
    as the ON durations, so a real short gap must never get bridged over.
    Second, the on/off threshold itself: CW's mark/space ratio runs much
    higher than a sparse-burst-in-silence VAD assumes (a string of dots is
    on half the time even at the element level), so a fixed-percentile
    noise floor can sit inside the "on" population instead of describing
    silence. Otsu's method on the block energies (:func:`_otsu_threshold`)
    finds the true on/off boundary regardless of what fraction of the
    record is keyed, and unlike a biggest-jump split it stays correct once
    there's real receiver noise blurring the two populations together.
    """
    block = max(1, int(sample_rate * block_ms / 1000))
    n = len(env) // block
    if n < 4:
        return []
    e = np.array([np.sqrt(np.mean(env[i * block:(i + 1) * block] ** 2))
                 for i in range(n)])
    log_e = np.log(e + 1e-12)
    on = log_e > _otsu_threshold(log_e)
    if not on.any() or on.all():
        return []
    # Otsu always finds *some* split, even in pure noise -- only trust it
    # if the two groups are actually far enough apart to be real keying,
    # not an arbitrary halfway point through one noise population (the
    # same 6 dB bar sigint.detect/.realtime use elsewhere for "is this a
    # real signal, not noise").
    contrast_db = 20 * (np.log10(e[on].mean() + 1e-12)
                        - np.log10(e[~on].mean() + 1e-12))
    if contrast_db < min_contrast_db:
        return []
    segs: list[_Seg] = []
    i = 0
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            segs.append(_Seg(start_s=i * block / sample_rate,
                             end_s=j * block / sample_rate))
            i = j
        else:
            i += 1
    return segs


def decode(samples: np.ndarray, sample_rate: float,
          block_ms: float = 2.0) -> MorseMessage | None:
    """Detect CW keying and decode it to text.

    Envelope-detects, finds on/off keying segments, then clusters the
    observed on-durations into dot/dash and the observed off-durations
    into intra-character/inter-character/inter-word gaps -- by the actual
    timing seen in this recording, not an assumed WPM. Returns ``None`` if
    there isn't enough on/off structure to call this CW at all
    (self-rejecting, like every other probe in this package).
    """
    x = np.asarray(samples, dtype=np.float64).ravel()
    env = np.abs(_analytic(x))
    segs = _keying_segments(env, sample_rate, block_ms)
    if len(segs) < 2:
        return None

    on_durs = np.array([s.end_s - s.start_s for s in segs])
    is_dash, _ = _cluster_two(on_durs)
    elements_per_seg = np.where(is_dash, "-", ".")

    gaps = np.array([segs[i].start_s - segs[i - 1].end_s
                     for i in range(1, len(segs))])
    # 0=intra-character 1=inter-character 2=inter-word, split by the two
    # biggest ratio jumps across ALL observed gaps at once -- splitting
    # intra-vs-rest first (as a smaller pass) is wrong whenever the
    # intra-to-inter-character jump is numerically bigger in log-space
    # than the inter-character-to-word jump (very common: 1u->3u is a 3x
    # ratio, 3u->7u only ~2.3x), which would merge inter-character gaps
    # into the "word" bucket instead of splitting them out correctly.
    gap_kind = _cluster_n(gaps, 3) if len(gaps) else np.array([], dtype=int)

    # unit estimate: dot on-durations, else the shortest observed on-duration
    dot_durs = on_durs[~is_dash]
    unit_s = float(np.median(dot_durs)) if len(dot_durs) else float(on_durs.min())
    wpm = 1.2 / unit_s if unit_s > 0 else 0.0

    letters: list[str] = []
    words: list[list[str]] = []
    cur = [elements_per_seg[0]]
    for i, kind in enumerate(gap_kind):
        if kind == 0:
            cur.append(elements_per_seg[i + 1])
        elif kind == 1:
            letters.append("".join(cur))
            cur = [elements_per_seg[i + 1]]
        else:  # word boundary
            letters.append("".join(cur))
            words.append(letters)
            letters = []
            cur = [elements_per_seg[i + 1]]
    letters.append("".join(cur))
    words.append(letters)

    text_words = []
    unrecognized = 0
    elem_words = []
    for word in words:
        chars = []
        elem_words.append(" ".join(word))
        for code in word:
            c = _REVERSE_TABLE.get(code)
            if c is None:
                unrecognized += 1
                chars.append("#")
            else:
                chars.append(c)      # single letter, or a prosign like <AR>
        text_words.append("".join(chars))

    return MorseMessage(text=" ".join(text_words),
                        elements=" / ".join(elem_words),
                        wpm=wpm, unrecognized=unrecognized)
