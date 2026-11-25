# seti-rf

One install for RF signal analysis and radio telescope RFI triage. This
experimental bundle combines the `sigint` and `sigint-rfi` projects for
evaluation at the SETI Institute.

## Install

Use Python 3.10 or newer in a fresh environment. The bundle installs the
same `sigint` and `sigint_rfi` import paths as the standalone projects,
so use a dedicated environment for the bundle.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install 'git+ssh://git@github.com/paoloo/seti-rf.git'
```

Only the bundle repository requires GitHub access. The numerical
dependencies are installed by pip. A local checkout also supports
`python -m pip install ".[dev]"`.

## Use

```python
from seti_rf import signal, triage

with triage.SigprocFilterbank("observation.fil") as source:
    scan = triage.FilterbankScan(source)
    for hit in scan.candidates(rows=(0, 128)):
        print(hit.frequency_hz, hit.on_excess_db)
```

Use `signal.detect` for carrier and modulation analysis and
`signal.rfi` for frequency-allocation lookup and waveform probes.
Use `triage` for single-beam SIGPROC or HDF5 scans and simultaneous
paired-beam comparisons.

The bundle includes the original Python imports and the `sigint-cli`,
`sigint-gui`, and `sigint-rfi` commands. The GUI requires Tk.

```sh
sigint-rfi on.fbh5 off.fbh5 --rows 0 16 > review.jsonl
```

The triage command emits review flags and opens observations read-only.
Automatic rejection and the live CyberEther voltage bridge are planned.

## Refresh the bundle

Development happens in the upstream repositories. The bundle takes
committed copies of their source and selected self-contained tests.
[The manifest](vendor/upstream.json) records both exact source commits.

From this checkout, use sibling checkouts of both upstream repositories
or pass their paths with `--sigint` and `--sigint-rfi`:

```sh
python tools/refresh.py --sigint-ref <commit> --sigint-rfi-ref <commit>
python -m pip install -e '.[dev]'
python -m pytest
```

Review the snapshots and the upstream dependency and package-data
settings when refreshing. The bundle includes legacy code and demo
captures from the pinned sources. Confirm the applicable redistribution
terms before changing repository visibility.

## License and citation

Project code is released under the [MIT License](LICENSE). Third-party
materials remain subject to their applicable terms.

If you use this software in research, cite it using [CITATION.cff](CITATION.cff).

## Recorded voltage replay

`seti_rf.triage.VoltageHistory` retains bounded complex-voltage chunks
for one observation, beam, tuning, and polarization. Candidate lookup uses
half-open sample intervals on the same observation clock. Requests include
an RF range, which is checked against the signed tuning. A successful
lookup returns the full-band voltages over the requested time interval
and the signed baseband endpoints for the requested RF range.

Run an offline replay with an installed copy of the package:

```sh
python tools/voltage_replay.py path/to/manifest.json > evidence.jsonl
```

The manifest contains a `stream` object matching `VoltageStream`, an
integer `capacity_samples`, and an `events` array in arrival order. A
chunk event has `type: "chunk"`, `path` (little-endian CF32, relative
to the manifest), `first_sample`, `sample_count`, and optional
`file_sample` (zero-based source offset). A candidate event has
`type: "candidate"`, `id`, `first_sample`, `stop_sample`, `low_hz`,
and `high_hz`. It may override stream fields in a `stream` object
and select the `gps_l1_ca` probe with `probe`.

The output contributes one JSON record per candidate, including the
original candidate record, lookup status, and any probe result or error.
