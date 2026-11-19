#!/usr/bin/env python3
"""Provision sigint fixture files for the CyberEther demo flowgraphs.

The File Reader block requires absolute paths and validates its file
at flowgraph creation — before any Python block can run — so the demo
flowgraphs read from /tmp/sigint_fixtures/. Run this once per machine
(after cloning):

    python3 sigint/workflow/provision_fixtures.py

Idempotent: re-copies only when the source is newer.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent          # sigint/
DEST = Path("/tmp/sigint_fixtures")

FILES = [
    "baudot_50bd_16k.cf32",
    "sitorb_50bd_16k.cf32",
    "hfdl_1200bps.cf32",
]


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in FILES:
        src = PKG / "fixtures" / name
        dst = DEST / name
        if not src.exists():
            print(f"missing: {src}", file=sys.stderr)
            continue
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            shutil.copy(src, dst)
            print(f"copied {name} -> {dst}")
            n += 1
        else:
            print(f"up to date: {name}")
    return 0 if n or all((DEST / f).exists() for f in FILES) else 1


if __name__ == "__main__":
    sys.exit(main())
