"""Batch-run the sigint decoder sweep over BEE .bin fixtures.

Writes a JSON + text summary: for each fixture, the top-ranked decoders
and whether the expected format (parsed from the filename) ranks first.

Usage:
    python3 -m sigint.fixture_sweep                  # this repo's own corpus
    python3 -m sigint.fixture_sweep /dir/with/bins    # a directory of .bin files
    python3 -m sigint.fixture_sweep a.bin b.bin ...   # explicit files
"""
import json
import re
import sys
import time
from pathlib import Path

_SIGINT_PARENT = str(Path(__file__).resolve().parent.parent)
if _SIGINT_PARENT not in sys.path:
    sys.path.insert(0, _SIGINT_PARENT)

from sigint import decoders as sd  # noqa: E402

# this repo's own corpus, used when no path is given on the command line
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIGDIR = _REPO_ROOT / "tests" / "fixtures" / "bee_demo" / "Signals"

# filename -> expected decoder-name fragments (normalized matching)
EXPECT = {
    "12s12p": ["12s12p"],
    "192": ["192"],
    "1s1": ["1s1"],
    "2b 2s1": ["2b 2s1"],
    "2s2": ["2s2"],
    "30s30": ["30s30"],
    "420": ["420"],
    "8s8k10l5": ["8s8k10"],
    "8s8k5": ["8s8k5"],
    "ALE": ["ALE"],
    "Ale3G": ["ALE"],
    "ALIS": ["ALIS"],
    "ARQ-6_90": ["ARQ-6/90"],
    "ARQ-E3": ["ARQ-E3"],
    "ARQ-E": ["ARQ-E"],
    "ARQ-M2-342_242 (ITA-3-1)": ["ITA-3-1"],
    "ARQ-M2-342_242 (ITA-3-2)": ["ITA-3-2"],
    "ASCII": ["ASCII"],
    "AUTOSPEC": ["AUTOSPEC"],
    "BAUDOT": ["BAUDOT", "Baudot"],
    "BULG-ASCII": ["BULG-ASCII"],
    "COCUELET MK8 008(а)": ["COQUELET"],
    "DatronLink": ["DatronLINK"],
    "DUP-ARQ-1": ["DUP-ARQ 1"],
    "DUP-ARQ-2": ["DUP-ARQ 2"],
    "DUP-FEC-2": ["DUP-FEC-2"],
    "FEC-A": ["FEC-A"],
    "Fed-std-1052": ["FED-STD-1052"],
    "HNG-FEC": ["HNG-FEC"],
    "Hamming": ["Hamming"],
    "KRE-PSK": ["KRE-PSK"],
    "Melp": ["MELP"],
    "MELPe2400": ["MELP"],
    "Merod (fin)": ["MEROD-FIN"],
    "Merod (ind)": ["MEROD-IND"],
    "Merod (usa)": ["MEROD-USA"],
    "Mil-Std-188-110A(period-1060)1": ["1060"],
    "Mil-Std-188-110A(period-1060)2": ["1060"],
    "Mil-Std-188-110A(period-2760)1": ["2760"],
    "Mil-Std-188-110A(period-2760)2": ["2760"],
    "Packet AX25": ["PACKET AX.25"],
    "PICCOLO MK6 006(a)": ["006a"],
    "PICCOLO MK6 006(b)": ["006b"],
    "PICCOLO MK6 006(c)": ["006c"],
    "POL-ARQ": ["POL-ARQ"],
    "Pactor FEC": ["PACTOR FEC"],
    "Pactor Long": ["PACTOR LONG"],
    "Pactor": ["PACTOR-I"],
    "Q15x25": ["Q15X25"],
    "ROU-MOI-FEC": ["ROU-MOI-FEC"],
    "RUM-FEC-1": ["RUM-FEC 1"],
    "RUM-FEC-2": ["RUM-FEC 2"],
    "Rsx25": ["RSX.25"],
    "SI-ARQ 70": ["SI-ARQ 70"],
    "SI-ARQ 84": ["SI-ARQ 84"],
    "Sitor-A": ["SITOR-A"],
    "Sitor-B (7x22)": ["SITOR-B (7x22)"],
    "Stanag_5066": ["Stanag-5066"],
    "SwedARQ": ["SWED ARQ"],
    "ZModem": ["ZModem"],
    "arm-da": ["ARM-DA"],
    "lpc": ["LPC10"],
}


def expected_for(name: str) -> list[str] | None:
    base = re.sub(r"\.(bin|BIN)$", "", name)
    for key, frags in EXPECT.items():
        if base.startswith(key) or key in base:
            return frags
    return None


def _collect_files(args: list[str]) -> list[Path]:
    if not args:
        if not DEFAULT_SIGDIR.is_dir():
            sys.exit(f"no path given and the default corpus is not present: "
                     f"{DEFAULT_SIGDIR}")
        sigdir = DEFAULT_SIGDIR
    elif len(args) == 1 and Path(args[0]).is_dir():
        sigdir = Path(args[0])
    else:
        return sorted(Path(a) for a in args)
    return sorted(sigdir.glob("*.bin")) + sorted(sigdir.glob("*.BIN"))


def main() -> None:
    files = _collect_files(sys.argv[1:])
    rows = []
    t0 = time.time()
    for f in files:
        bits = sd.bits_from_bytes(f.read_bytes(), bit_order="little")
        hits = sd.run_all(bits, top=3)
        frags = expected_for(f.name)
        top_names = [o.decoder for o in hits]
        matched = None
        if frags:
            for i, n in enumerate(top_names):
                if any(frag.lower() in n.lower() for frag in frags):
                    matched = i  # rank of the expected decoder (0 = first)
                    break
        rows.append({
            "file": f.name,
            "bits": len(bits),
            "top3": top_names,
            "expected": frags,
            "expected_rank": matched,
        })
        rank = matched if matched is not None else "-"
        print(f"{f.name[:42]:44s} rank={rank}  top={top_names[:2]}")
    elapsed = time.time() - t0
    matched_count = sum(1 for r in rows if r["expected_rank"] == 0)
    known = sum(1 for r in rows if r["expected"] is not None)
    hit_any = sum(1 for r in rows if r["expected_rank"] is not None)
    print(f"\n{len(rows)} fixtures in {elapsed:.0f}s")
    print(f"  expected format ranked #1: {matched_count}/{known}")
    print(f"  expected format in top-3:  {hit_any}/{known}")
    out = Path.cwd() / "fixture_report.json"
    out.write_text(json.dumps({"rows": rows,
                               "summary": {"total": len(rows),
                                           "rank1": matched_count,
                                           "top3": hit_any,
                                           "known": known,
                                           "elapsed_s": elapsed}}, indent=2))
    print(f"report: {out}")


if __name__ == "__main__":
    main()
