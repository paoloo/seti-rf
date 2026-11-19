#!/usr/bin/env python3
"""sigint-cli: run the full detection pipeline on a recording, no GUI.

Usage:
    python3 sigint_cli.py <file.wav|npy|npz|raw> [options]

Pipeline: load -> PSD/carriers -> activity segments -> modulation
classification -> band scan -> optional FSK demod + decoder sweep ->
report to stdout (text or JSON).

Examples:
    python3 sigint_cli.py capture.wav
    python3 sigint_cli.py capture.wav --json
    python3 sigint_cli.py capture.wav --demod 2125 1275 --sweep
    python3 sigint_cli.py capture.wav --band 1500 2500 --save-bits out.bin
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SIGINT_PARENT = str(Path(__file__).resolve().parent.parent)
if _SIGINT_PARENT not in sys.path:
    sys.path.insert(0, _SIGINT_PARENT)

from sigint import dataset as ds  # noqa: E402
from sigint import detect as det  # noqa: E402
from sigint import decoders as sd  # noqa: E402
from sigint import dsp  # noqa: E402
from sigint.gui._io import load_recording  # noqa: E402


def run_bits(path: str, args: argparse.Namespace) -> dict:
    """Bitstream mode: the file is a packed demodulated capture (BEE's
    .bin fixtures). Sweep the decoders directly, no RF stage."""
    data = Path(path).read_bytes()
    result: dict = {
        "file": str(path), "mode": "bitstream",
        "bytes": len(data), "bits": len(data) * 8,
    }
    order = "little" if not args.big_endian else "big"
    result["bit_order"] = order
    bits = sd.bits_from_bytes(data, bit_order=order)
    if args.show_bits:
        result["bitstream_head"] = "".join(map(str, bits[:args.show_bits]))
    outcomes = sd.run_all(bits, top=args.top_decoders)
    result["decoder_sweep"] = [
        {"decoder": o.decoder, "symbols": o.symbol_count,
         "errors": o.error_count, "text": o.text[:args.max_text]}
        for o in outcomes]
    return result


def run(path: str, args: argparse.Namespace) -> dict:
    x, sr = load_recording(path, preserve_iq=getattr(args, "complex_iq", False))
    is_iq = np.iscomplexobj(x)
    if is_iq and args.demod:
        raise ValueError("--demod requires real audio; omit --complex-iq for legacy audio demos")
    if args.sample_rate is not None:
        sr = args.sample_rate
    if not np.isfinite(sr) or sr <= 0:
        raise ValueError("sample rate must be positive and finite")
    if getattr(args, "center_hz", None) is not None and not np.isfinite(args.center_hz):
        raise ValueError("center frequency must be finite")
    result: dict = {
        "file": str(path),
        "samples": len(x),
        "sample_rate": sr,
        "duration_s": round(len(x) / sr, 3),
    }
    flo = args.band[0] if args.band else None
    fhi = args.band[1] if args.band else None

    # 1. carriers
    center_hz = getattr(args, "center_hz", None)
    scan = det.find_carriers(x, sr, fmin=flo, fmax=fhi, center_hz=center_hz or 0.0)
    result["noise_floor_db"] = round(scan.noise_floor_db, 1)
    result["carriers"] = [
        {"freq_hz": round(c.freq_hz, 1),
         **({"abs_freq_hz": c.abs_freq_hz} if center_hz is not None else {}),
         "prominence_db": round(c.prominence_db, 1),
         "bandwidth_hz": round(c.bandwidth_hz, 1),
         "persistence": f"{c.frames_present}/{c.frames_total}"}
        for c in scan.carriers[:args.max_carriers]]

    # 2. activity
    segs = det.detect_activity(x, sr)
    result["activity_segments"] = [
        {"start_s": round(s.start_s, 3), "end_s": round(s.end_s, 3),
         "energy_db": round(s.energy_db, 1)} for s in segs[:10]]

    # 3. classification (whole record + per activity segment)
    cls = det.classify_modulation(x, sr, fmin=flo, fmax=fhi)
    result["classification"] = {"label": cls.label,
                                "confidence": round(cls.confidence, 2),
                                "features": {k: round(v, 4)
                                             for k, v in cls.features.items()}}

    # 4. band scan
    hits = det.band_scan(x, sr, fmin=flo, fmax=fhi, center_hz=center_hz or 0.0)
    result["band_scan"] = [
        {"center_hz": round(h.center_hz, 1),
         **({"abs_center_hz": h.abs_center_hz} if center_hz is not None else {}),
         "snr_db": round(h.snr_db, 1),
         "class": h.classification.label} for h in hits[:8]]

    # 4b. waveform-level probe: named PHY decoders that identify the
    # exact waveform (mode label) from samples, not bits. HFDL first:
    # audio -> analytic baseband at the measured subcarrier (same
    # conversion hfdl.decode_file uses, but on the LOADED samples so
    # headerless formats and --sample-rate work).
    if is_iq:
        return result

    try:
        from sigint import hfdl as hfdl_mod
        from scipy import signal as _sp
        f_peak, P = _sp.welch(x, sr, nperseg=1024)
        tone = f_peak[int(np.argmax(P))]
        if not (800 < tone < 2600):
            tone = hfdl_mod.SSB_CARRIER_OFFSET_HZ
        ana = _sp.hilbert(x)
        bb = ana * np.exp(-2j * np.pi * tone * np.arange(len(x)) / sr)
        frame = hfdl_mod.decode_frame(bb, sr)
        frames = [frame] if frame else []
        if frames:
            result["hfdl"] = [
                {"mode": f.mode, "waveform": f.label,
                 "symbols": f.symbols, "crc_ok": f.crc_ok,
                 "bytes": len(f.bytes_out),
                 "text": f.bytes_out[:args.max_text]
                         .decode("latin-1", errors="replace")}
                for f in frames]
            result["waveform"] = f"HFDL {frames[0].label}"
    except Exception:
        pass  # no frame / unreadable file: not an HFDL recording

    # 5. optional FSK demod + decoder sweep
    if args.demod:
        mark, space = args.demod
        baud = args.baud
        if baud is None:
            ev = dsp.baud_rate_stages(x, sr)
            baud = ev.rate_hz
            result["baud_estimate_hz"] = round(baud, 2)
        dem = dsp.fsk_demod_sync(x, sr, mark, space, baud)
        bits = dem.bits
        result["demod"] = {"mark_hz": mark, "space_hz": space,
                           "symbol_rate_hz": round(dem.symbol_rate_hz, 2),
                           "bits": len(bits),
                           "ones_fraction": round(float(bits.mean()), 3)}
        # named decoders: force specific bitstream formats (e.g.
        # --decoder "Baudot/ITA2") instead of relying on the sweep's
        # generic ranking, which favors framing decoders
        if args.decoder:
            forced = []
            for name in args.decoder:
                o = sd.run_decoder(name, bits)
                forced.append({"decoder": o.decoder,
                               "symbols": o.symbol_count,
                               "errors": o.error_count,
                               "text": o.text[:args.max_text]})
            result["named_decode"] = forced
            if forced:
                best = min(forced, key=lambda f: f["errors"])
                result["waveform"] = best["decoder"]
                result["message"] = best["text"]
        if args.show_bits:
            shown = bits[:args.show_bits]
            result["demod"]["bitstream_head"] = "".join(map(str, shown))
        if args.save_bits:
            packed = np.packbits(bits.astype(np.uint8), bitorder="little")
            Path(args.save_bits).write_bytes(packed.tobytes())
            result["demod"]["bits_saved_to"] = args.save_bits
        if args.sweep:
            outcomes = sd.run_all(bits, top=args.top_decoders)
            result["decoder_sweep"] = [
                {"decoder": o.decoder, "symbols": o.symbol_count,
                 "errors": o.error_count,
                 "text": o.text[:args.max_text]}
                for o in outcomes]
            result["waveform"] = outcomes[0].decoder if outcomes else None
            result["waveform_candidates"] = [o.decoder for o in outcomes]
    return result


def print_text(r: dict) -> None:
    if r.get("mode") == "bitstream":
        print(f"File:          {r['file']}  (bitstream mode)")
        print(f"Size:          {r['bytes']} bytes = {r['bits']} bits "
              f"({r['bit_order']}-first)")
        if r.get("bitstream_head"):
            bs = r["bitstream_head"]
            for i in range(0, len(bs), 64):
                print(f"  {i:5d}: {bs[i:i + 64]}")
        print(f"Decoder sweep ({len(r['decoder_sweep'])} hits):")
        for o in r["decoder_sweep"]:
            first = o["text"].strip().splitlines()[:2]
            text = " | ".join(first)[:70]
            print(f"  {o['decoder']:28s} {o['symbols']:5d} sym "
                  f"{o['errors']:4d} err  {text}")
        if not r["decoder_sweep"]:
            print("  no decoder produced >= 4 symbols")
        return
    print(f"File:          {r['file']}")
    print(f"Samples:       {r['samples']} @ {r['sample_rate']:.0f} Hz "
          f"({r['duration_s']} s)")
    print(f"Noise floor:   {r['noise_floor_db']} dB")
    print(f"Classification: {r['classification']['label']} "
          f"(confidence {r['classification']['confidence']})")
    if r["carriers"]:
        print(f"Carriers ({len(r['carriers'])}):")
        for c in r["carriers"]:
            print(f"  {c['freq_hz']:9.1f} Hz  prom {c['prominence_db']:6.1f} dB  "
                  f"bw ~{c['bandwidth_hz']:7.1f} Hz  persist {c['persistence']}")
    else:
        print("Carriers:      none")
    if r["activity_segments"]:
        print(f"Activity ({len(r['activity_segments'])}):")
        for s in r["activity_segments"]:
            print(f"  {s['start_s']:8.3f} .. {s['end_s']:8.3f} s  "
                  f"+{s['energy_db']:5.1f} dB above floor")
    if r["band_scan"]:
        print(f"Band scan hits ({len(r['band_scan'])}):")
        for h in r["band_scan"]:
            print(f"  {h['center_hz']:9.1f} Hz  SNR {h['snr_db']:6.1f} dB  "
                  f"{h['class']}")
    if "baud_estimate_hz" in r:
        print(f"Baud estimate: {r['baud_estimate_hz']} Hz (4-stage)")
    if r.get("hfdl"):
        print(f"HFDL ({len(r['hfdl'])} frame(s)):")
        for f in r["hfdl"]:
            crc = "CRC OK" if f["crc_ok"] else "CRC bad"
            print(f"  {f['symbols']} sym  {f['waveform']}  {crc}  "
                  f"{f['bytes']} bytes")
    if "demod" in r:
        d = r["demod"]
        print(f"FSK demod:     {d['bits']} bits @ {d['symbol_rate_hz']} Bd "
              f"({d['mark_hz']:.0f}/{d['space_hz']:.0f} Hz), "
              f"ones {d['ones_fraction']}")
        if r.get("message"):
            print(f"Message:       {r['message'][:200]}")
        if d.get("bitstream_head"):
            bs = d["bitstream_head"]
            width = 64
            print("Bits:")
            for i in range(0, len(bs), width):
                print(f"  {i:5d}: {bs[i:i + width]}")
        if d.get("bits_saved_to"):
            print(f"Bits saved:    {d['bits_saved_to']}")
    # waveform naming isn't gated on --demod: the HFDL probe (run
    # unconditionally above) can name a waveform on its own
    if r.get("waveform"):
        print(f"Waveform:      {r['waveform']}")
        if r.get("waveform_candidates") and \
                len(r["waveform_candidates"]) > 1:
            print(f"Candidates:    {', '.join(r['waveform_candidates'])}")
    if "decoder_sweep" in r:
        print(f"Decoder sweep ({len(r['decoder_sweep'])} hits):")
        for o in r["decoder_sweep"]:
            first = o["text"].strip().splitlines()[:2]
            text = " | ".join(first)[:70]
            print(f"  {o['decoder']:28s} {o['symbols']:5d} sym "
                  f"{o['errors']:4d} err  {text}")


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="sigint-cli",
        description="Run the sigint detection pipeline on a recording.")
    ap.add_argument("file", help="recording: .wav .npy .npz .raw .iq")
    ap.add_argument("--json", action="store_true", help="JSON output")
    ap.add_argument("--band", nargs=2, type=float, metavar=("LO", "HI"),
                    help="restrict analysis to a frequency band (Hz)")
    ap.add_argument("--max-carriers", type=int, default=8)
    ap.add_argument("--demod", nargs=2, type=float, metavar=("MARK", "SPACE"),
                    help="FSK demodulate with these tones (Hz)")
    ap.add_argument("--baud", type=float, default=None,
                    help="nominal symbol rate (default: 4-stage estimate)")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep all bitstream decoders on the demodulated bits")
    ap.add_argument("--complex-iq", action="store_true",
                    help="preserve signed IQ in .cf32 and complex NumPy recordings")
    ap.add_argument("--center-hz", type=float, help="RF frequency of baseband DC in Hz")
    ap.add_argument("--sample-rate", type=float, default=None,
                    metavar="HZ",
                    help="sample rate for headerless files "
                         "(.f32/.cf32/.raw/.iq; default 16000)")
    ap.add_argument("--decoder", action="append", metavar="NAME",
                    help="force a named bitstream decoder on the demodulated "
                         "bits (repeatable, e.g. --decoder 'Baudot/ITA2')")
    ap.add_argument("--top-decoders", type=int, default=5)
    ap.add_argument("--max-text", type=int, default=400,
                    help="max decoded-text chars per decoder in the report")
    ap.add_argument("--show-bits", nargs="?", const=256, type=int,
                    metavar="N",
                    help="print the first N demodulated bits (default 256)")
    ap.add_argument("--save-bits", metavar="PATH",
                    help="write demodulated bits (packed LSB-first)")
    ap.add_argument("--bits", action="store_true",
                    help="treat the file as a packed bitstream (.bin "
                         "fixtures): skip RF, run the decoder sweep directly")
    ap.add_argument("--big-endian", action="store_true",
                    help="with --bits: MSB-first bit order (default LSB)")
    args = ap.parse_args()

    try:
        r = run_bits(args.file, args) if args.bits else run(args.file, args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    if args.json:
        print(json.dumps(r, indent=2))
    else:
        print_text(r)


if __name__ == "__main__":
    main()
