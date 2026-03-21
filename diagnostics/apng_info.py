#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
apng_info.py — Inspect APNG frame structure.

Reports per-frame sub-image dimensions, offsets, blend/dispose modes,
and compressed data sizes by parsing fcTL and fdAT/IDAT chunks directly.
No dependencies beyond the standard library.

Usage:
    python apng_info.py FILE.apng [FILE2.apng ...]
    python apng_info.py --help
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

DISPOSE_NAMES = {0: "none", 1: "background", 2: "previous"}
BLEND_NAMES = {0: "source", 1: "over"}
COLOR_TYPE_NAMES = {0: "gray", 2: "rgb", 3: "indexed", 4: "gray+a", 6: "rgba"}


def _read_chunks(path: str):
    """Yield (chunk_type, payload) for each chunk in a PNG file."""
    with open(path, "rb") as f:
        sig = f.read(8)
        if sig[:4] != b"\x89PNG":
            raise ValueError(f"Not a PNG file: {path}")
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            length = struct.unpack(">I", header[:4])[0]
            chunk_type = header[4:8]
            payload = f.read(length)
            f.read(4)  # CRC
            yield chunk_type, payload
            if chunk_type == b"IEND":
                break


def inspect(path: str, *, verbose: bool = False) -> None:
    canvas_w = canvas_h = 0
    bit_depth = color_type = 0
    n_frames = 0
    n_plays = 0
    frame_idx = -1
    pending_fctl: dict | None = None
    data_bytes = 0  # accumulated IDAT/fdAT bytes for current frame
    total_data = 0

    print(f"File: {path}  ({Path(path).stat().st_size:,} bytes)")

    for chunk_type, payload in _read_chunks(path):
        if chunk_type == b"IHDR":
            canvas_w, canvas_h, bit_depth, color_type = struct.unpack(
                ">IIBB", payload[:10]
            )
            ct_name = COLOR_TYPE_NAMES.get(color_type, f"?{color_type}")
            print(
                f"  Canvas: {canvas_w}×{canvas_h}  "
                f"{bit_depth}-bit {ct_name}"
            )

        elif chunk_type == b"PLTE":
            n_entries = len(payload) // 3
            print(f"  Palette: {n_entries} entries")

        elif chunk_type == b"tRNS":
            n_transp = sum(1 for b in payload if b < 255)
            print(f"  tRNS: {len(payload)} entries, {n_transp} transparent")

        elif chunk_type == b"acTL":
            n_frames, n_plays = struct.unpack(">II", payload[:8])
            loop = "infinite" if n_plays == 0 else f"{n_plays}×"
            print(f"  Frames: {n_frames}  Loop: {loop}")
            print()
            if verbose:
                print(
                    f"  {'#':>3s}  {'Size':>11s}  {'Offset':>7s}  "
                    f"{'Delay':>9s}  {'Dispose':<11s}  {'Blend':<7s}  "
                    f"{'Data':>8s}"
                )
                print(f"  {'—'*3}  {'—'*11}  {'—'*7}  {'—'*9}  {'—'*11}  {'—'*7}  {'—'*8}")
            else:
                print(
                    f"  {'#':>3s}  {'Size':>11s}  {'Offset':>7s}  "
                    f"{'Dispose':<11s}  {'Blend':<7s}  {'Data':>8s}"
                )
                print(f"  {'—'*3}  {'—'*11}  {'—'*7}  {'—'*11}  {'—'*7}  {'—'*8}")

        elif chunk_type == b"fcTL":
            # Flush previous frame's data total.
            if pending_fctl is not None:
                _print_frame(pending_fctl, data_bytes, verbose=verbose)
                total_data += data_bytes

            seq, sw, sh, x, y = struct.unpack(">IIIII", payload[:20])
            delay_num, delay_den, dispose_op, blend_op = struct.unpack(
                ">HHBB", payload[20:]
            )
            frame_idx += 1
            data_bytes = 0
            pending_fctl = {
                "idx": frame_idx,
                "w": sw,
                "h": sh,
                "x": x,
                "y": y,
                "delay_num": delay_num,
                "delay_den": delay_den,
                "dispose": dispose_op,
                "blend": blend_op,
            }

        elif chunk_type in (b"IDAT", b"fdAT"):
            data_bytes += len(payload)

    # Flush last frame.
    if pending_fctl is not None:
        _print_frame(pending_fctl, data_bytes, verbose=verbose)
        total_data += data_bytes

    print()
    print(f"  Total compressed data: {total_data:,} bytes")


def _print_frame(f: dict, data_bytes: int, *, verbose: bool) -> None:
    size_str = f"{f['w']}×{f['h']}"
    offset_str = f"+{f['x']}+{f['y']}"
    dispose_str = DISPOSE_NAMES.get(f["dispose"], f"?{f['dispose']}")
    blend_str = BLEND_NAMES.get(f["blend"], f"?{f['blend']}")
    data_str = f"{data_bytes:,}"

    if verbose:
        den = f["delay_den"] or 1000
        delay_ms = f["delay_num"] * 1000 / den
        delay_str = f"{delay_ms:.0f} ms"
        print(
            f"  {f['idx']:3d}  {size_str:>11s}  {offset_str:>7s}  "
            f"{delay_str:>9s}  {dispose_str:<11s}  {blend_str:<7s}  "
            f"{data_str:>8s}"
        )
    else:
        print(
            f"  {f['idx']:3d}  {size_str:>11s}  {offset_str:>7s}  "
            f"{dispose_str:<11s}  {blend_str:<7s}  {data_str:>8s}"
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("files", nargs="+", metavar="FILE", help="APNG file(s)")
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show per-frame delay timing",
    )
    args = p.parse_args()

    for i, path in enumerate(args.files):
        if i > 0:
            print()
        try:
            inspect(path, verbose=args.verbose)
        except (ValueError, FileNotFoundError, struct.error) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
