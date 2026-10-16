#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Niko1221 and the Strata contributors
"""Main-owned post-run full byte comparison; never launches native work.

Use only for a same-nq/cap cohort with successful fixture/runtime/fault receipts.
This comparator establishes frame identity, not process success or speed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def read_frame(path):
    path = Path(path)
    size = path.stat().st_size
    if size > 7 * 1024 * 1024 or size < 24:
        raise ValueError(f"frame byte budget/header: {path}")
    blob = path.read_bytes()
    magic, nq, cap, count = struct.unpack_from("<8sIIQ", blob)
    if (magic != b"QRCV1LE\0" or nq not in (128, 256) or cap != 2051
            or count != nq * 24 * 256 or len(blob) != 24 + count * 4):
        raise ValueError(f"invalid frame extent/header: {path}")
    if not all(math.isfinite(x[0]) for x in struct.iter_unpack("<f", blob[24:])):
        raise ValueError(f"nonfinite frame: {path}")
    return blob, {"path": str(path), "bytes": len(blob), "nq": nq,
                  "cap": cap, "sha256": hashlib.sha256(blob).hexdigest(),
                  "finite": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", nargs="+")
    args = parser.parse_args()
    if len(args.frames) < 2:
        parser.error("at least two frames from the same nq cohort are required")
    first, receipt = read_frame(args.frames[0])
    receipts = [receipt]
    equal = True
    for path in args.frames[1:]:
        other, receipt = read_frame(path)
        receipt["exact_reference"] = other == first
        receipts.append(receipt)
        equal = equal and other == first
    print(json.dumps({"status": "PASS" if equal else "FAIL",
                      "scope": "full_initial_frame_bytes_only",
                      "process_and_fault_validation": "EXTERNAL_REQUIRED",
                      "frames": receipts}, indent=2))
    return 0 if equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
