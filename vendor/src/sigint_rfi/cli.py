"""Write an auditable JSON Lines report from a SIGPROC or HDF5 filterbank."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path

import h5py

from .filterbank import FilterbankScan
from .sigproc import SigprocFilterbank


def _open(path: str):
    if Path(path).suffix.lower() == ".fil":
        return SigprocFilterbank(path)
    return h5py.File(path, "r")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("on", help="on-target SIGPROC .fil or HDF5 filterbank")
    parser.add_argument("off", nargs="?", help="optional simultaneous off-target filterbank")
    parser.add_argument("--on-db", type=float, default=12.0)
    parser.add_argument("--off-db", type=float, default=12.0)
    parser.add_argument("--tolerance-channels", type=int, default=0)
    parser.add_argument("--block-channels", type=int, default=8192)
    parser.add_argument("--rows", type=int, nargs=2, metavar=("START", "STOP"),
                        help="zero-based half-open integration range")
    parser.add_argument("--channels", type=int, nargs=2, metavar=("START", "STOP"),
                        help="zero-based half-open channel range")
    args = parser.parse_args()

    try:
        with ExitStack() as stack:
            on = stack.enter_context(_open(args.on))
            off = stack.enter_context(_open(args.off)) if args.off else None
            scan = FilterbankScan(on, off)
            settings = vars(args).copy()
            settings.pop("on")
            settings.pop("off")
            print(json.dumps({"type": "observation", "schema_version": 1,
                              "sources": {"on": args.on, "off": args.off},
                              "grid": asdict(scan.grid), "settings": settings}))
            count = matches = 0
            for candidate in scan.candidates(**settings):
                print(json.dumps({"type": "candidate", **asdict(candidate),
                                  "coincident": candidate.coincident if off is not None else None}, allow_nan=False))
                count += 1
                matches += candidate.coincident
            print(json.dumps({"type": "summary", "candidates": count,
                              "coincident": matches if off is not None else None}))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
