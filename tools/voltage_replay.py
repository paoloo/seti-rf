"""Replay sample-indexed CF32 chunks and candidate records from a JSON manifest."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from seti_rf.signal import rfi
from seti_rf.triage import VoltageHistory, VoltageRequest, VoltageStream

GPS_CENTER = 1_575_420_000


def probe_gps(cutout):
    stream = cutout.request.stream
    center_offset = GPS_CENTER - stream.center_hz
    if abs(center_offset) + 1_023_000 > stream.sample_rate_hz / 2:
        raise ValueError("GPS L1 C/A main lobe does not fit the voltage bandwidth")
    samples = cutout.samples if stream.sideband == 1 else cutout.samples.conj()
    if center_offset:
        seconds = np.arange(samples.size) / stream.sample_rate_hz
        samples = samples * np.exp(-2j * np.pi * center_offset * seconds)
    return [asdict(hit) for hit in rfi.gps.acquire(samples, stream.sample_rate_hz)]


def replay(path):
    manifest = json.loads(path.read_text())
    stream = VoltageStream(**manifest["stream"])
    history = VoltageHistory(stream, manifest["capacity_samples"])
    probes = {"gps_l1_ca": probe_gps, None: None}
    for event in manifest["events"]:
        if event["type"] == "chunk":
            source = path.parent / event["path"]
            count = event["sample_count"]
            offset = event.get("file_sample", 0)
            if count <= 0 or offset < 0:
                raise ValueError("invalid chunk size or file offset")
            samples = np.fromfile(source, dtype="<c8", count=count, offset=offset * 8)
            if len(samples) != count:
                raise ValueError(f"incomplete chunk in {source}")
            history.append(event["first_sample"], samples)
        elif event["type"] == "candidate":
            candidate_stream = VoltageStream(**{**manifest["stream"], **event.get("stream", {})})
            request = VoltageRequest(
                event["id"], candidate_stream, event["first_sample"], event["stop_sample"],
                event["low_hz"], event["high_hz"],
            )
            evidence = history.annotate(request, probes[event.get("probe")])
            yield {"candidate": event, "lookup": evidence.status.value,
                   "probe_result": evidence.probe_result, "probe_error": evidence.probe_error}
        else:
            raise ValueError(f"unknown event type: {event['type']!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    for evidence in replay(args.manifest):
        print(json.dumps(evidence, allow_nan=False))


if __name__ == "__main__":
    main()
