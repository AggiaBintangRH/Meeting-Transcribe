#!/usr/bin/env python3
"""Repair recordings whose WAV header says 0 frames while holding real audio.

THE FAILURE THIS REPAIRS
------------------------
`AVAudioFile` writes the `data` chunk's size field when the object is RELEASED.
If the app goes away while recording — Force Quit, a crash, and (until
2026-08-05) an ordinary ⌘Q or a ⌃C on `swift run` — that release never happens.
The samples are all written; only the 4-byte size field is still 0. Every stage
in this project then reads the file as EMPTY: `sf.info` reports 0.0 s, the
diarizers return no speakers, and a whole meeting looks lost.

Found on the owner's machine 2026-08-05: **13 recordings, ~1.7 GB**, one of them
34.1 minutes of real speech (RMS 0.0063, peaks ±0.5). All recoverable, because
nothing was ever missing except the number.

The app side is fixed (`AppDelegate.applicationShouldTerminate` +
`installSignalHandlers` → `AudioRecorder.finalizeRecordingFiles`), but SIGKILL
cannot be caught by anyone, so this tool remains the answer for a hard kill.

WHAT IT CHANGES
---------------
Exactly two little-endian uint32 fields, in place:
  * the `data` chunk size  → the bytes actually present after it, rounded DOWN
    to a whole frame (`block_align`), so a torn final frame is dropped rather
    than fed to a decoder as a partial sample;
  * the `RIFF` size        → `filesize - 8`.
No audio byte is touched, and the file does not change length.

SAFETY
------
Dry run by DEFAULT. `--apply` is required to write, and it refuses any file
whose `data` size is already non-zero — repairing a healthy file is the one way
this tool could destroy something, so it simply will not.

Usage:
    python3 scripts/tools/repair-wav-header.py recordings/            # report
    python3 scripts/tools/repair-wav-header.py recordings/ --apply    # repair
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

# `fmt ` fields we need to size a frame. Both are read from the file itself
# rather than assumed, because this project writes 44.1 kHz float32 mono today
# and has written other shapes before.
_FMT_CHANNELS = 2
_FMT_BLOCK_ALIGN = 12


class NotRepairable(Exception):
    """The file is fine, or is not a WAV this tool understands."""


def scan(path: Path) -> dict:
    """Walk the RIFF chunks and describe what is wrong, without writing."""
    size = path.stat().st_size
    with path.open("rb") as f:
        riff = f.read(12)
        if len(riff) < 12 or riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise NotRepairable("not a RIFF/WAVE file")
        block_align = None
        while True:
            head = f.read(8)
            if len(head) < 8:
                raise NotRepairable("no data chunk")
            cid, csize = struct.unpack("<4sI", head)
            body = f.tell()
            if cid == b"fmt ":
                fmt = f.read(csize)
                if len(fmt) >= 16:
                    block_align = struct.unpack_from("<H", fmt, _FMT_BLOCK_ALIGN)[0]
                f.seek(body + csize + (csize & 1))
                continue
            if cid == b"data":
                return {
                    "path": path,
                    "file_size": size,
                    "data_offset": body,
                    "declared": csize,
                    "present": size - body,
                    "block_align": block_align or 1,
                    "size_field_at": body - 4,
                }
            f.seek(body + csize + (csize & 1))


def repair(info: dict, apply: bool) -> str:
    if info["declared"] != 0:
        raise NotRepairable(
            f"data chunk already declares {info['declared']:,} bytes — healthy")
    if info["present"] <= 0:
        raise NotRepairable("no bytes after the data chunk header")

    align = max(1, info["block_align"])
    # Round DOWN to a whole frame: a kill can land mid-sample, and a decoder
    # handed a partial frame is a second, subtler corruption.
    usable = (info["present"] // align) * align
    dropped = info["present"] - usable
    if usable == 0:
        raise NotRepairable("fewer bytes present than one frame")

    verb = "would set" if not apply else "set"
    note = (f"{verb} data={usable:,} bytes"
            + (f" (dropping {dropped} trailing byte(s) of a torn frame)"
               if dropped else ""))
    if apply:
        with info["path"].open("r+b") as f:
            f.seek(info["size_field_at"])
            f.write(struct.pack("<I", usable))
            f.seek(4)                                   # RIFF size field
            f.write(struct.pack("<I", info["file_size"] - 8))
    return note


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="a .wav file or a directory of them")
    ap.add_argument("--apply", action="store_true",
                    help="actually write (default is a dry run)")
    args = ap.parse_args()

    target = Path(args.target)
    files = (sorted(target.glob("*.wav")) if target.is_dir() else [target])
    if not files:
        print(f"no .wav files under {target}")
        return 1

    broken = healthy = failed = 0
    for path in files:
        try:
            info = scan(path)
            note = repair(info, args.apply)
        except NotRepairable as exc:
            if "healthy" in str(exc):
                healthy += 1
            else:
                failed += 1
                print(f"  SKIP   {path.name}: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR  {path.name}: {exc}")
            continue
        broken += 1
        seconds = info["present"] / info["block_align"] / 44100
        print(f"  {'REPAIRED' if args.apply else 'BROKEN  '} {path.name}: {note}"
              f"  (~{seconds / 60:.1f} min at 44.1 kHz)")

    print(f"\n{broken} repairable, {healthy} already healthy, {failed} skipped.")
    if broken and not args.apply:
        print("Dry run — nothing was written. Re-run with --apply to repair.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
