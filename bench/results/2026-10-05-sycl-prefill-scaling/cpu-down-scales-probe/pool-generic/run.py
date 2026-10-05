#!/usr/bin/env python3
"""Paired CPU-only production-pool comparison; no SYCL queue or GPU reset."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

out = Path(sys.argv[1]).resolve()
model = Path.home() / ".local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
pack = Path.home() / ".local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s"
env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")}
records = []


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def same_bytes(a, b):
    assert a.stat().st_size == b.stat().st_size
    with a.open("rb") as fa, b.open("rb") as fb:
        while True:
            x, y = fa.read(1024 * 1024), fb.read(1024 * 1024)
            assert x == y, f"output bits differ: {a} {b}"
            if not x:
                break


def status(pid):
    try:
        p = Path(f"/proc/{pid}")
        stat = (p / "stat").read_text().split(")", 1)[1].split()
        info = {"pid": pid, "state": stat[0], "cpu_ticks": int(stat[11]) + int(stat[12]), "last_cpu": int(stat[36])}
        for line in (p / "status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmSwap:")):
                k, v = line.split(":", 1)
                info[k] = v.strip()
        return info
    except FileNotFoundError:
        return {"pid": pid, "exited": True}


manifest = {"model": str(model), "pack": str(pack), "gpu_control": status(1074147),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "environment": {k: v for k, v in env.items() if k.startswith(("ONEAPI", "SYCL", "LD_LIBRARY", "OMP"))},
            "binaries": {arm: digest(out / f"pool-{arm}") for arm in ("baseline", "scales")},
            "archives": {arm: digest(out / f"production-{arm}.a") for arm in ("baseline", "scales")},
            "source": digest(out / "probe.cpp"), "validation": {}, "case_outputs": []}
for arm in ("baseline", "scales"):
    prefix = out / f"validation-{arm}"
    with prefix.with_suffix(".jsonl").open("w") as stdout, prefix.with_suffix(".stderr").open("w") as stderr:
        result = subprocess.run([str(out / f"pool-{arm}"), str(model), str(pack), "validate", str(prefix.with_suffix(".bin"))], env=env, stdout=stdout, stderr=stderr)
    assert result.returncode == 0, f"validation failed: {arm} rc={result.returncode}"
    events = [json.loads(line) for line in prefix.with_suffix(".jsonl").read_text().splitlines()]
    assert events[-1]["kind"] == "completed"
    manifest["validation"][arm] = {"summary": events[-1], "sha256": digest(prefix.with_suffix(".bin")), "bytes": prefix.with_suffix(".bin").stat().st_size}
same_bytes(out / "validation-baseline.bin", out / "validation-scales.bin")
manifest["validation"]["all_bits_equal"] = True
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"kind": "validation_passed", **manifest["validation"]}), flush=True)


class Peer:
    def __init__(self, arm, process_round):
        self.arm = arm
        self.raw = (out / f"process-{process_round}-{arm}.jsonl").open("w")
        self.err = (out / f"process-{process_round}-{arm}.stderr").open("w")
        self.proc = subprocess.Popen([str(out / f"pool-{arm}"), str(model), str(pack), "bench"], env=env,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True, bufsize=1)
        assert self.receive()["kind"] == "ready"

    def receive(self):
        line = self.proc.stdout.readline()
        assert line, f"{self.arm} exited unexpectedly: {self.proc.poll()}"
        self.raw.write(line)
        self.raw.flush()
        return json.loads(line)

    def command(self, text):
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()
        return self.receive()

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
        assert self.proc.wait() == 0, f"{self.arm} failed"
        self.raw.close()
        self.err.close()


with (out / "pairs.jsonl").open("w") as raw:
    for process_round in range(3):
        peers = {arm: Peer(arm, process_round) for arm in ("baseline", "scales")}
        try:
            for layer in (1, 2, 17, 21):
                for count in (12, 48):
                    for pattern in (1, 2, 4, -4):
                        meta = None
                        for arm, peer in peers.items():
                            ready = peer.command(f"CASE {layer} {count} {pattern}")
                            assert ready["kind"] == "case_ready"
                            if meta is None:
                                meta = ready
                            else:
                                assert ready == meta
                            assert peer.command(f"DUMP {out / (arm + '-case.bin')}")["kind"] == "dumped"
                        same_bytes(out / "baseline-case.bin", out / "scales-case.bin")
                        manifest["case_outputs"].append({**meta, "process_round": process_round, "sha256": digest(out / "baseline-case.bin"), "all_bits_equal": True})
                        repeats = 8 if count == 12 else 2
                        for pair_round in range(9):
                            first = "baseline" if (process_round + pair_round) % 2 == 0 else "scales"
                            order = (first, "scales" if first == "baseline" else "baseline")
                            timings = {}
                            observations = {arm: status(peer.proc.pid) for arm, peer in peers.items()}
                            observations["gpu_control"] = status(1074147)
                            for arm in order:
                                # Let the other production pool reach its baseline 20 ms sleep boundary.
                                time.sleep(.06)
                                timings[arm] = peers[arm].command(f"BENCH {repeats}")
                                assert timings[arm]["kind"] == "timing"
                            row = {**meta, "kind": "pair", "process_round": process_round, "pair_round": pair_round,
                                   "first": first, "status_before": observations, **timings,
                                   "speed_ratio": timings["baseline"]["wall_ms"] / timings["scales"]["wall_ms"]}
                            raw.write(json.dumps(row) + "\n")
                            raw.flush()
                            records.append(row)
                        subset = records[-9:]
                        print(json.dumps({"kind": "case_completed", "layer": layer, "experts": count,
                                          "pattern": pattern, "process_round": process_round,
                                          "median_speed_ratio": statistics.median(x["speed_ratio"] for x in subset)}), flush=True)
        finally:
            for peer in peers.values():
                peer.close()

summary = []
for layer in (1, 2, 17, 21):
    for count in (12, 48):
        for pattern in (1, 2, 4, -4):
            subset = [x for x in records if (x["layer"], x["experts"], x["pattern"]) == (layer, count, pattern)]
            row = {k: subset[0][k] for k in ("layer", "gu_type", "down_type", "experts", "pattern", "weight_bytes")}
            row.update(pairs=len(subset), speed_ratio=statistics.median(x["speed_ratio"] for x in subset),
                       process_speed_ratios=[statistics.median(x["speed_ratio"] for x in subset if x["process_round"] == r) for r in range(3)])
            for phase in ("wall_ms", "gu_ms", "quant_ms", "down_ms"):
                row[phase] = {arm: statistics.median(x[arm][phase] for x in subset) for arm in ("baseline", "scales")}
                row[phase]["paired_speed_ratio"] = statistics.median(x["baseline"][phase] / x["scales"][phase] for x in subset)
            summary.append(row)
manifest["gpu_control_end"] = status(1074147)
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({"kind": "completed", "paired_cases": len(summary), "pairs": len(records)}), flush=True)
