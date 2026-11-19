"""Refresh the bundled source and tests from two committed Git snapshots."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "sigint": {
        "url": "https://github.com/paoloo/sigint",
        "package": "sigint",
        "tests": (
            "test_rfi_adsb.py", "test_rfi_bands.py", "test_rfi_gps.py", "test_rfi_triage.py",
        ),
    },
    "sigint-rfi": {
        "url": "https://github.com/paoloo/sigint-rfi",
        "package": "sigint_rfi",
        "tests": ("test_filterbank.py", "test_sigproc.py"),
    },
}


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=True,
    ).stdout


def _snapshot(repo: Path, revision: str, name: str, target: Path) -> dict[str, str]:
    source = SOURCES[name]
    sha = _git(repo, "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}").decode().strip()
    package_prefix = f"src/{source['package']}/"
    names = [package_prefix.rstrip("/"), *(f"tests/{test}" for test in source["tests"])]
    if name == "sigint-rfi":
        names.append("tools/frb_retention.py")
    archive = _git(repo, "archive", "--format=tar", sha, "--", *names)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        for item in tar:
            if item.isdir():
                continue
            parts = Path(item.name).parts
            if not item.isfile() or ".." in parts or Path(item.name).is_absolute():
                raise ValueError(f"unsupported archive entry {item.name!r}")
            if item.name.startswith(package_prefix):
                relative = Path(*parts)
            elif parts[0] == "tests" and len(parts) == 2 and parts[1] in source["tests"]:
                relative = Path("tests", name, parts[1])
            elif item.name == "tools/frb_retention.py" and name == "sigint-rfi":
                relative = Path("frb_retention.py")
            else:
                raise ValueError(f"unexpected archive entry {item.name!r}")
            output = target / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(item) as data:
                output.write_bytes(data.read())
    return {"repository": source["url"], "commit": sha}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigint", type=Path, default=ROOT.parent / "sigint")
    parser.add_argument("--sigint-rfi", type=Path, default=ROOT.parent / "sigint-rfi")
    parser.add_argument("--sigint-ref", required=True)
    parser.add_argument("--sigint-rfi-ref", required=True)
    args = parser.parse_args()

    destination = ROOT / "vendor"
    try:
        with TemporaryDirectory(prefix=".refresh-", dir=ROOT) as directory:
            staging, previous = (Path(directory) / name for name in ("new", "previous"))
            manifest = {
                "sigint": _snapshot(args.sigint, args.sigint_ref, "sigint", staging),
                "sigint-rfi": _snapshot(args.sigint_rfi, args.sigint_rfi_ref, "sigint-rfi", staging),
            }
            (staging / "upstream.json").write_text(json.dumps(manifest, indent=2) + "\n")
            if destination.exists():
                destination.rename(previous)
            try:
                staging.rename(destination)
            except OSError:
                if previous.exists():
                    previous.rename(destination)
                raise
    except subprocess.CalledProcessError as exc:
        parser.exit(1, exc.stderr.decode(errors="replace"))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
