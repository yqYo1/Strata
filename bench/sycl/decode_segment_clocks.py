"""Summarize the target verifier's four sampled T4 windows from engine stderr."""

import argparse
import json
from pathlib import Path


def layer_kind(layer):
    return "qsa" if (layer + 1) % 4 == 0 else "gdn"


def summarize(lines, request):
    prefix = "strata decode segment: "
    fields = "request window pos qsa_interval segment status fields submit start end resolution_ns host_wait_ns pool_callback_ns".split()
    windows = []
    count = previous_end = 0
    for line in lines:
        if not line.startswith(prefix):
            continue
        parts = [field.split("=", 1) for field in line[len(prefix):].split()]
        if any(len(p) != 2 for p in parts) or [p[0] for p in parts] != fields:
            raise ValueError("malformed verifier segment row")
        row = {k: int(v) for k, v in parts}
        window, segment = divmod(count, 49)
        if (row["request"], row["window"], row["segment"]) != (request, 9 + window, segment) or window >= 4:
            raise ValueError("missing, repeated, foreign or out-of-order segment")
        start, end = row["start"], row["end"]
        if (row["status"] or row["fields"] != 7 or row["qsa_interval"] != 4
                or not 0 < row["submit"] <= start < end or start < previous_end
                or not 0 < row["resolution_ns"] <= end - start
                or min(row["host_wait_ns"], row["pool_callback_ns"], row["pos"]) < 0):
            raise ValueError("unsupported geometry or invalid segment clocks")
        previous_end = end
        if segment == 0:
            if windows and row["pos"] <= windows[-1]["position"]:
                raise ValueError("non-advancing sampled decode position")
            windows.append(dict(position=row["pos"], boundaries={}))
        current = windows[-1]
        if row["pos"] != current["position"] or (segment == 48 and row["pool_callback_ns"]):
            raise ValueError("segment position or tail callback differs")
        prev = layer_kind(segment - 1) if segment else "input"
        following = layer_kind(segment) if segment < 48 else "head"
        bucket = current["boundaries"].setdefault(f"{prev}_to_{following}", dict(
            segments=0, graph_ns=0, queue_delay_ns=0, host_wait_ns=0, pool_callback_ns=0))
        bucket["segments"] += 1
        bucket["graph_ns"] += end - start
        bucket["queue_delay_ns"] += start - row["submit"]
        for key in ("host_wait_ns", "pool_callback_ns"):
            bucket[key] += row[key]
        count += 1
    if count != 4 * 49:
        raise ValueError(f"expected 196 completed segment readings; got {count}")
    return windows


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("engine_stderr", type=Path)
    cli.add_argument("request", type=int)
    args = cli.parse_args()
    with args.engine_stderr.open() as stream:
        print(json.dumps(summarize(stream, args.request), indent=2))
