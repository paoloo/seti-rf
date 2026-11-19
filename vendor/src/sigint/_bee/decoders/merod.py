"""MEROD-USA/-FIN/-IND message framing, recovered from `002.dll`/`079.dll`/`080.dll`.

`MEROD-USA`/`-FIN`/`-IND` share the DLL codename base `127c127`
(`Sig127c127_u.dll`, `127C127Fin.dll`, `127C127Ind.dll`), but the three are not
packaged the same way. `002.dll` (`MEROD-USA`) is a Borland/Delphi VCL object
exporting only `CreateDec`/`DestroyDec`, whereas `079.dll` and `080.dll` export
the MFC `Create`/`SetData`/`LoadFile`/`Decod` quartet this project has already
traced successfully for Ital HDLC, DatronLINK and FED-STD-1052 -- so those two
were reachable by the same method. `002.dll` needed the Delphi VMT route
instead: `CreateDec` -> its constructor -> the class VMT at `0x0044e144`
(parent VMT `0x0044dbe8`, both within `002.dll`'s own code), and two of the
virtual methods hanging off it -- see "MEROD-USA recovered from the Delphi
VMT" below.

## MEROD-FIN / MEROD-IND (`079.dll` / `080.dll`)

## What the DLLs do

`Decod` (`0x10001290` in both) delegates to `fcn.10001ff0`, which emits
`"---------- START ----------"`, then loops:

```text
position = find_sync(this, 0)                 # fcn.10002170
while position:
    end = find_message_end(this, position)    # fcn.10002240
    emit(this, position, end)                 # fcn.10002340
    position = find_sync(this, end)
emit "----------  END  ----------"
```

`find_sync` slides a fixed table over the decoder's bit buffer -- which holds
**one bit per byte**, so the tables are literal bit patterns -- and counts
matching elements. Above a high threshold it reports a normal-polarity hit;
below a low threshold the window matches the table's complement and it reports
an inverted hit (`[this+0xc]` is the polarity flag). Either way it returns the
hit offset plus a fixed skip, which is where the payload starts. `find_message_end`
slides a second table forward from there and stops at the first window that
either matches that table or is (almost) perfectly alternating -- an idle
pattern -- and returns that offset, which is also where the next sync search
begins.

| | `MEROD-FIN` (`079.dll`) | `MEROD-IND` (`080.dll`) |
|---|---|---|
| sync table | `0x10026478`, 32 bits | `0x100264b8`, 128 bits |
| sync accept (normal) | > 30 of 32 | > 120 of 128 |
| sync accept (inverted) | < 2 of 32 | < 5 of 128 |
| payload skip after sync | `0x120` = 288 bits | `0x80` = 128 bits |
| end table | `0x10026498`, 32 bits | `0x10026478`, 64 bits |
| end accept | > 30 matches, or > 30 transitions | > 60 matches, or > 60 transitions |

## What the captures say

Read with the documented LSB-first packing (see `documented_period`), each
sync table matches its own bundled capture **exactly**, and nothing else:

| Sync table | `Merod (fin).BIN` | `Merod (ind).BIN` | `Merod (usa).bin` | `12s12p.bin` |
|---|---|---|---|---|
| `079.dll` 32-bit | **15 exact** (next best 27/32) | 27/32, none exact | 25/32, none exact | 26/32, none exact |
| `080.dll` 128-bit | 86/128, none exact | **3 exact** (next best 86/128) | 88/128, none exact | 87/128, none exact |

MSB-first, neither table produces a single hit in any capture. A 128-bit exact
match cannot be a coincidence, and the 32-bit one is separated from the best
non-match by five bit errors.

Walking the DLL's loop over the captures yields 15 messages in
`Merod (fin).BIN` and 3 in `Merod (ind).BIN`, all normal polarity, with
payloads that terminate on a real end-marker rather than running to the end of
the file.

## MEROD-USA recovered from the Delphi VMT (`002.dll`)

`CreateDec` (`0x0044e680`) calls a constructor chain (`0x0044e1e8` ->
`0x0044dc94`) that ends by storing the class VMT pointer `0x0044e144` in the
new object. Unlike the earlier constant-table sweep of this DLL (which only
found five short, unreferenced one-bit-per-byte runs and correctly concluded
they weren't it), the VMT's own positive-offset slots point at the class's
*actual* overridden methods, two of which are the framing logic:

- **`+0x54` (`0x0044e330`)**: creates a `TIniFile(".\\merod(u).ini")`, calls
  `ReadString("addr", "prefix", <64-bit ASCII default>)`, loads the result
  into a `TOkBits` bit-buffer, and slides it over the capture looking for an
  exact match -- then does the same again with a second, 63-bit ASCII
  default. The literal `127` (`mov ecx, 0x7f` at `0x0044e3f3`) appears in the
  modulo used to compute a phase offset from the hit position.
- **`+0x50` (`0x0044e50c`)**: the literal `78` (`imul eax, [ebx+0x84], 0x4e`
  at `0x0044e55f`) multiplies that phase by 78 while walking a payload in
  78-bit steps -- the BCH(127,78) codeword/message split (see below).

Both literals match a third-party MEROD reference
(`https://cartoonman.github.io/Shoc/merod.htm`, retrieved 2026-08-14, cited
in GUIDE.md's MEROD section): MEROD uses BCH(127,78) forward error
correction, so `127`/`78` are the codeword and message lengths, not
coincidental constants -- independent, external corroboration for a value
this session found by reading the disassembly, not the other way around.

The two ASCII bit-string defaults (`0x0044e450`, 64 bits; `0x0044e4bc`, 63
bits -- read as `TIniFile` fallback values, so the *original* application
would use a custom `merod(u).ini` if one exists, and these hardcoded strings
otherwise) match `Merod (usa).bin` **exactly**, 4 times each, LSB-first only
(0 hits MSB-first): offsets `[170, 2060, 3762, 5722]` for the 64-bit prefix
and `[1506, 3396, 5098, 7058]` for the 63-bit marker, pairing into four
identical 1,399-bit messages (`prefix(64b) + payload(1,272b) + marker(63b)`).
1,272 bits = exactly **159 bytes**, which independently matches a third VMT
slot (`+0x5c`, `0x0044e5d8`): a getter that does nothing but `mov eax, 0x9f;
ret` (159 decimal) -- a payload-size constant found by a completely separate
route confirming the same figure. Unlike `079.dll`/`080.dll`'s sync tables,
these are *default* configuration values rather than unconditional hardcoded
constants, so `tests/test_merod.py` notes that distinction explicitly.

## Scope

This recovers message framing: where each message starts, where its payload
begins, and where it ends. `fcn.10002340`/`fcn.0044e224` and friends, which
would turn the payload bits into the report text, are not implemented -- the
alphabet and the BCH(127,78) FEC/voting structure remain unrecovered for all
three names.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sigint._bee.decoders.base import DecodeResult, Decoder
from sigint._bee.decoders.registry import register


def _pattern(text: str) -> np.ndarray:
    return np.array([int(c) for c in text], dtype=np.uint8)


@dataclass(frozen=True)
class MerodSpec:
    """The constants `fcn.10002170`/`fcn.10002240` (or, for `MEROD-USA`,
    `fcn.0044e330`) are compiled with."""

    dll: str
    sync: np.ndarray
    sync_va: int
    sync_min_match: int
    sync_max_match_inverted: int
    payload_skip: int
    end: np.ndarray
    end_va: int
    end_min_match: int
    end_min_transitions: int
    #: True for MEROD-USA's `TIniFile.ReadString` fallback values, which the
    #: DLL stores as ASCII '0'/'1' characters rather than the one-bit-per-byte
    #: tables `079.dll`/`080.dll` use; False (default) for those tables.
    ascii_default: bool = False


SPECS = {
    # 079.dll: sync 0x10026478, end 0x10026498, skip 0x120, thresholds 0x1e/2 and 0x1e.
    "MEROD-FIN": MerodSpec(
        dll="079.dll",
        sync=_pattern("00100100011101010010111000110100"),
        sync_va=0x10026478,
        sync_min_match=31,
        sync_max_match_inverted=1,
        payload_skip=0x120,
        end=_pattern("10011001100110011001100110011001"),
        end_va=0x10026498,
        end_min_match=31,
        end_min_transitions=31,
    ),
    # 080.dll: sync 0x100264b8, end 0x10026478, skip 0x80, thresholds 0x78/5 and 0x39+3/0x3c.
    "MEROD-IND": MerodSpec(
        dll="080.dll",
        sync=_pattern(
            "11111000000100001100010100111101000111001001011011101100"
            "11010101000001111110111100111010110000101110001101101001"
            "0001001100101010"),
        sync_va=0x100264B8,
        sync_min_match=121,
        sync_max_match_inverted=4,
        payload_skip=0x80,
        end=_pattern("0101000010001011011111101011100011001110110000001111001001010100"),
        end_va=0x10026478,
        end_min_match=61,
        end_min_transitions=61,
    ),
    # 002.dll: TIniFile("addr","prefix") ASCII defaults at 0x44e450 (64b) and
    # 0x44e4bc (63b), read by fcn.0044e330; both match Merod (usa).bin exactly,
    # 4 times each, LSB-first only. No inverted-polarity or idle-marker
    # evidence exists for this name, so those thresholds are set unreachable
    # rather than guessed.
    "MEROD-USA": MerodSpec(
        dll="002.dll",
        sync=_pattern("0011101100111010000111011101010011111111101101001001110100010001"),
        sync_va=0x0044E450,
        sync_min_match=64,
        sync_max_match_inverted=-1,
        payload_skip=64,
        end=_pattern("001000011000101001111010001110010010110111011001101010111111000"),
        end_va=0x0044E4BC,
        end_min_match=63,
        end_min_transitions=999,
        ascii_default=True,
    ),
}


@dataclass(frozen=True)
class MerodMessage:
    sync_offset: int
    payload_start: int
    payload_stop: int
    inverted: bool
    orientation: str

    @property
    def payload_bits(self) -> int:
        return self.payload_stop - self.payload_start


def _reverse_each_byte(bits: np.ndarray) -> np.ndarray:
    complete = len(bits) // 8 * 8
    out = bits.copy()
    if complete:
        out[:complete] = bits[:complete].reshape(-1, 8)[:, ::-1].reshape(-1)
    return out


def _walk(bits: np.ndarray, spec: MerodSpec, orientation: str) -> list[MerodMessage]:
    if len(bits) <= max(len(spec.sync), len(spec.end)):
        return []
    sync_match = (np.lib.stride_tricks.sliding_window_view(bits, len(spec.sync))
                  == spec.sync).sum(axis=1)
    end_window = np.lib.stride_tricks.sliding_window_view(bits, len(spec.end))
    end_match = (end_window == spec.end).sum(axis=1)
    end_transitions = (end_window[:, 1:] != end_window[:, :-1]).sum(axis=1)
    end_here = (end_match >= spec.end_min_match) | (end_transitions >= spec.end_min_transitions)

    messages: list[MerodMessage] = []
    position = 0
    while position < len(sync_match):
        normal = sync_match[position:] >= spec.sync_min_match
        inverted = sync_match[position:] <= spec.sync_max_match_inverted
        candidates = np.flatnonzero(normal | inverted)
        if not len(candidates):
            break
        hit = position + int(candidates[0])
        start = hit + spec.payload_skip
        after = np.flatnonzero(end_here[min(start, len(end_here) - 1):])
        stop = min(start, len(bits) - 1) + int(after[0]) if len(after) else len(bits) - 1
        messages.append(MerodMessage(hit, start, stop, bool(inverted[hit - position]),
                                     orientation))
        if stop <= hit:
            break
        position = stop
    return messages


def find_merod_messages(bits: np.ndarray, name: str) -> list[MerodMessage]:
    """Reproduce the DLL's sync/end message walk, searching both byte
    orientations because the caller may have unpacked MSB-first."""
    spec = SPECS[name]
    stream = np.asarray(bits, dtype=np.uint8).reshape(-1) & 1
    best: list[MerodMessage] = []
    for oriented, orientation in ((stream, "as-given"),
                                  (_reverse_each_byte(stream), "byte-reversed")):
        found = _walk(oriented, spec, orientation)
        if len(found) > len(best):
            best = found
    return best


class MerodDecoder(Decoder):
    status = "partial"
    confidence = "high"
    layer = "synchronization"
    description = "DLL sync-word message framing (payload alphabet unrecovered)"

    def decode(self, bits: np.ndarray, block=None) -> DecodeResult:
        spec = SPECS[self.name]
        messages = find_merod_messages(bits, self.name)
        kind = "TIniFile default" if spec.ascii_default else "table"
        text = "\n".join(
            f"@{item.sync_offset}b sync={len(spec.sync)}b payload={item.payload_start}"
            f"..{item.payload_stop} ({item.payload_bits}b) "
            f"polarity={'inverted' if item.inverted else 'normal'} "
            f"orientation={item.orientation}" for item in messages)
        return DecodeResult(
            text=text,
            symbol_count=len(messages),
            error_count=0 if messages else 1,
            notes=(f"{len(messages)} message(s) framed by the {len(spec.sync)}-bit sync "
                   f"{kind} at {spec.dll}:0x{spec.sync_va:08x} and the "
                   f"{len(spec.end)}-bit end {kind} at 0x{spec.end_va:08x}. Payload "
                   "alphabet and the BCH(127,78) FEC/voting structure are not "
                   "recovered -- see bee/decoders/merod.py."),
        )


for _name in SPECS:
    _decoder = MerodDecoder()
    _decoder.name = _name
    _spec = SPECS[_name]
    if _spec.ascii_default:
        _decoder.provenance = (
            f"static analysis of original Lib/{_spec.dll} (Delphi VMT slots "
            "+0x54/+0x50/+0x5c off the CreateDec-constructed object) with both "
            "TIniFile default bit-strings confirmed exactly on the bundled capture")
    else:
        _decoder.provenance = (f"static analysis of original Lib/{_spec.dll} "
                               "(fcn.10002170 sync search, fcn.10002240 end search) with "
                               "both constant tables confirmed exactly on the bundled capture")
    register(_decoder)
