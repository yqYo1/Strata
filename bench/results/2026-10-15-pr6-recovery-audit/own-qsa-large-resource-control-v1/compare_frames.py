#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Niko1221 and the Strata contributors
"""Stream full finite/byte comparison of same-shape QRCV1LE initial frames.

Main must separately qualify fixture exit, ownership closure and fault receipts.
This program never launches native work and never deletes or modifies frames.
One canonical plus one candidate suffices; retire consumed candidates externally.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import stat
import struct

HEADER = struct.Struct("<8sIIQ")
MAX_BYTES = 193 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024  # Multiple of FP32 size; no complete payload is buffered.


def stable_identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def open_frame(stack, path):
    # Reject FIFOs/devices/symlinks and never block opening a FIFO by mistake.
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        stream = os.fdopen(fd, "rb", buffering=0)
    except BaseException:
        os.close(fd)
        raise
    stack.enter_context(stream)  # close errors propagate; both handles always close.
    info = os.fstat(stream.fileno())
    if not stat.S_ISREG(info.st_mode) or not HEADER.size <= info.st_size <= MAX_BYTES:
        raise ValueError(f"regular frame byte budget/header: {path}")
    header = read_exact(stream, HEADER.size, path)
    magic, nq, cap, count = HEADER.unpack(header)
    if (magic != b"QRCV1LE\0" or nq not in (128, 256, 8192) or cap != 2051
            or count != nq * 24 * 256 or info.st_size != HEADER.size + count * 4):
        raise ValueError(f"invalid frame extent/header: {path}")
    return stream, info, header, {"path": str(path), "bytes": info.st_size,
                                  "nq": nq, "cap": cap, "count": count}


def read_exact(stream, count, path):
    # Regular-file short reads are legal. Each request is bounded independently.
    result = bytearray(count)
    view = memoryview(result)
    done = 0
    while done < count:
        got = stream.readinto(view[done:])
        if not got:
            raise ValueError(f"truncated frame: {path}")
        done += got
    return result


def finite_bits(block, path):
    # All-ones IEEE binary32 exponent is exactly Inf/NaN; includes signaling NaNs.
    # Inspect every element, with no lossy conversion or fingerprint-only pass.
    if any((bits & 0x7F800000) == 0x7F800000
           for (bits,) in struct.iter_unpack("<I", block)):
        raise ValueError(f"nonfinite frame: {path}")


def compare_pair(reference, candidate):
    with ExitStack() as stack:
        a, ai, ah, ar = open_frame(stack, reference)
        b, bi, bh, br = open_frame(stack, candidate)
        if ah != bh:
            raise ValueError("same nq/cap/count cohort required")
        if (ai.st_dev, ai.st_ino) == (bi.st_dev, bi.st_ino):
            raise ValueError("reference and candidate must be distinct files")
        hashes = [hashlib.sha256(ah), hashlib.sha256(bh)]
        equal, first_mismatch = True, None
        remaining, offset = ar["count"] * 4, HEADER.size
        while remaining:
            size = min(CHUNK_BYTES, remaining)
            ab = read_exact(a, size, reference)
            bb = read_exact(b, size, candidate)
            finite_bits(ab, reference)
            finite_bits(bb, candidate)
            hashes[0].update(ab)
            hashes[1].update(bb)
            if ab != bb:
                equal = False
                if first_mismatch is None:
                    first_mismatch = offset + next(i for i, (x, y) in enumerate(zip(ab, bb)) if x != y)
            remaining -= size
            offset += size
        for stream, info, path in ((a, ai, reference), (b, bi, candidate)):
            if stream.read(1) or stable_identity(os.fstat(stream.fileno())) != stable_identity(info):
                raise ValueError(f"frame changed during comparison: {path}")
        for receipt, digest in zip((ar, br), hashes):
            receipt.update(sha256=digest.hexdigest(), finite=True)
        br.update(exact_reference=equal, first_mismatch_byte=first_mismatch)
    return ar, br  # Success only after both checked closes.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", nargs="+", type=Path)
    args = parser.parse_args()
    if len(args.frames) < 2:
        parser.error("at least two distinct frames from the same nq cohort are required")
    receipts, canonical, equal = [], None, True
    try:
        for path in args.frames[1:]:
            reference, other = compare_pair(args.frames[0], path)
            if canonical is None:
                canonical = reference
                receipts.append(reference)
            elif reference != canonical:
                raise ValueError("canonical frame changed between candidates")
            receipts.append(other)
            equal = equal and other["exact_reference"]
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error),
                          "scope": "full_initial_frame_bytes_only",
                          "process_and_fault_validation": "EXTERNAL_REQUIRED",
                          "frames_completed": receipts}, indent=2))
        return 1
    print(json.dumps({"status": "PASS" if equal else "FAIL",
                      "scope": "full_initial_frame_bytes_only",
                      "process_and_fault_validation": "EXTERNAL_REQUIRED",
                      "comparison": "streaming_all_payload_bytes_and_all_finite",
                      "frames": receipts}, indent=2))
    return 0 if equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
