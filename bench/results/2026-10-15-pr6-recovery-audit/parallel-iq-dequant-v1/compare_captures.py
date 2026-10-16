#!/usr/bin/env python3
"""Offline complete-array comparison; no native child or environment mutation.

Main must first close both owned native processes normally and attach their
exit/retirement/fault receipts. This script reports array parity only.
"""
import hashlib
import json
from pathlib import Path
import struct
import sys


def check(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    check(len(sys.argv) == 4, "usage: compare_captures.py OFF_DIR ON_DIR NEW_REPORT.json")
    off, on, report = map(Path, sys.argv[1:])
    check(not report.exists(), "report exists")
    a, b = (read_json(p / "complete.json") for p in (off, on))
    check(a["complete"] is True and b["complete"] is True, "incomplete arm")
    check(a["arm"] == "off" and b["arm"] == "on", "arms")
    check(a["mode"] == b["mode"] and a["mode"] in ("quick", "full"), "mode")
    expected_cases = 48 if a["mode"] == "full" else 42
    for receipt in (a, b):
        check(receipt["cases"] == expected_cases, "missing cases")
        check(receipt["negative_checks"] == 20, "negative checks")
        check(receipt["f32_cases"] == 6, "f32 cases")
        check(receipt["expected_product_launches_per_valid_case"] == 1, "launch definition")
        check(receipt["fixture_markers_per_half_case"] == 2, "marker definition")
    check((off / "device.txt").read_bytes() == (on / "device.txt").read_bytes(), "device")
    expected = {"complete.json", "device.txt"}
    samples, hashes = [], {"off": {}, "on": {}}
    fields = ("type", "gu", "rows", "width", "offset", "elements", "edges",
              "cpu_mismatch", "cpu_nan_sign_mismatch", "guards", "inputs_unchanged")
    formats = {18:(256,98), 20:(32,18), 21:(256,110), 22:(256,82),
               23:(256,136), 42:(64,18), 2:(32,18), 6:(32,22), 7:(32,24),
               8:(32,34), 11:(256,110), 12:(256,144), 13:(256,176),
               14:(256,210), 16:(256,66), 17:(256,74), 29:(256,56), 30:(1,2)}
    for i in range(expected_cases):
        name = str(i)
        left, right = (read_json(p / (name + ".json")) for p in (off, on))
        for field in fields:
            check(left[field] == right[field], f"case {i} metadata {field}")
        check(left["cpu_mismatch"] == 0, f"case {i} CPU oracle")
        check(left["guards"] is True and left["inputs_unchanged"] is True, "guards/input")
        n = left["rows"] * left["width"] * (2 if left["gu"] else 1)
        check(left["elements"] == n, "element count")
        width, size = formats[left["type"]]
        packed = left["rows"] * left["width"] // width * size + 64 + left["offset"]
        suffixes = [".input", ".half", ".oracle"] + ([".up"] if left["gu"] else [])
        for suffix in suffixes:
            file = name + suffix
            expected.add(file)
            x, y = ((p / file).read_bytes() for p in (off, on))
            check(x == y, f"full bytes differ {file} (NaN bits included)")
            required = (n+32)*2 if suffix == ".half" else n*2 if suffix == ".oracle" else packed
            check(len(x) == required, f"length {file}")
            if suffix == ".half":
                check(x[:32] == b"\x5a\xa5"*16 and x[-32:] == b"\x5a\xa5"*16, "output guards")
                oracle = (off / (name + ".oracle")).read_bytes()
                nan_sign = 0
                for (got,), (want,) in zip(struct.iter_unpack("<H", x[32:-32]),
                                          struct.iter_unpack("<H", oracle)):
                    if got != want:
                        check((got & 0x7fff) == (want & 0x7fff) == 0x7e00, "CPU half bits")
                        nan_sign += 1
                check(nan_sign == left["cpu_nan_sign_mismatch"], "CPU NaN accounting")
        expected.add(name + ".json")
        samples.append({"case": i, "identity": {k:left[k] for k in fields},
                        "off": {k:left[k] for k in ("host_submit_ns", "drained_wall_ns", "marker_interval_ns")},
                        "on": {k:right[k] for k in ("host_submit_ns", "drained_wall_ns", "marker_interval_ns")}})
    for type_id in (18,22,21,23,20,42):
        file = f"f32-{type_id}.bin"
        expected.add(file)
        x, y = ((p / file).read_bytes() for p in (off, on))
        check(x == y, file + " full bits")
        check(len(x) == (formats[type_id][0]*3+2)*4, file + " count")
        guard = struct.pack("<f", 123456.0)
        check(x[:4] == guard and x[-4:] == guard, file + " guards")
        input_file = f"f32-{type_id}.input"
        expected.add(input_file)
        x, y = ((p / input_file).read_bytes() for p in (off, on))
        check(x == y and len(x) == formats[type_id][1]*3+2, input_file + " full input")
    for label, root in (("off", off), ("on", on)):
        check({p.name for p in root.iterdir()} == expected, "unexpected/missing capture files")
        hashes[label] = {file: {"bytes": (root/file).stat().st_size,
                                "sha256": digest(root/file)} for file in sorted(expected)}
    result = {"array_comparison": "PASS", "mode": a["mode"], "samples": samples,
              "files": hashes,
              "scope": "Complete device OFF/ON half buffers including NaNs, guards, inputs, CPU oracles and six F32 outputs. Owner exit/retirement, fault checks, actual route, repeated speed and model gates require separate Main receipts."}
    with report.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print("PASS complete arrays; timing samples are diagnostic only")


if __name__ == "__main__":
    main()
