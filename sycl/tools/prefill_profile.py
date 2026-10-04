#!/usr/bin/env python3
"""Measure prompt-length scaling with fixed memory settings and actual DMA timings.

Run under the same oneAPI/driver environment used for the engine. Each length
uses a prefix of one token fixture, followed by one held-out prompt token. The
engine batches all but that last token. Model loading and decode are outside
the reported prefill wall time. No MTP draft layer is loaded by default.
"""

import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--tokens-file", type=Path, required=True)
    parser.add_argument("--expert-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", default="1024,2048,4007")
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--expert-cache", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--mtp", type=Path, help="Also include production draft-KV prompt processing")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--mode", choices=("profile", "transfer", "wall"), default="profile",
                        help="Full phase markers, only memcpy events, or elapsed time without instrumentation")
    parser.add_argument("--cwd", type=Path, default=Path(__file__).resolve().parents[2])
    opts = parser.parse_args()
    lengths = sorted(set(int(n) for n in opts.lengths.split(",")))
    tokens = [int(t) for t in opts.tokens_file.read_text().split()]
    if not lengths or min(lengths) < 1 or max(lengths) >= len(tokens):
        parser.error("The fixture must contain at least max(lengths) + 1 tokens")
    if opts.chunk < max(lengths):
        parser.error("Use a chunk at least as large as every length for a one-chunk comparison")
    if opts.max_context < max(lengths) + 2:
        parser.error("The context must fit the longest prefix and one generated token")
    if opts.repeats < 1:
        parser.error("repeats must be positive")
    opts.output.mkdir(parents=True, exist_ok=True)
    exe = opts.exe.resolve()
    env = dict(os.environ)
    env.update(STRATA_PREFILL_FIRST="0", STRATA_PREFILL_RING="8")
    env.pop("STRATA_PREFILL_TIMING", None)
    env.pop("STRATA_PREFILL_TRANSFER_TIMING", None)
    if opts.mode != "wall":
        env["STRATA_PREFILL_TRANSFER_TIMING"] = "1"
    if opts.mode == "profile":
        env["STRATA_PREFILL_TIMING"] = "1"
    report = {
        "label": opts.label, "mode": opts.mode,
        "exe": str(exe), "binary_sha256": sha256(exe),
        "fixture": str(opts.tokens_file.resolve()), "fixture_sha256": sha256(opts.tokens_file),
        "lengths": lengths, "chunk": opts.chunk, "max_context": opts.max_context,
        "expert_cache": opts.expert_cache, "mtp": str(opts.mtp) if opts.mtp else None,
        "env": {k: v for k, v in env.items() if k.startswith(("STRATA_", "SYCL_", "ONEAPI_", "NEO_"))
                or k == "LD_LIBRARY_PATH"},
        "notes": [
            "DMA active time sums memcpy event durations; copy span includes queue gaps.",
            "Host copy worker time sums parallel CPU copies; it is not elapsed wall time.",
            "GPU phase times partition the compute-queue timeline and include idle gaps and waits.",
            "Copy, CPU staging, and compute overlap. Do not add them or subtract DMA time from wall time.",
            "The local wall-time slope includes input-dependent routing and attention costs.",
            "Profiler results need a separate run without phase markers to check instrumentation overhead.",
        ],
        "runs": [],
    }
    target = opts.output / "run.json"

    def save():
        target.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for repeat in range(opts.repeats):
        order = lengths if repeat % 2 == 0 else list(reversed(lengths))
        for n in order:
            name = f"{opts.label}-{n}-r{repeat + 1}"
            fixture = opts.output / f"{name}-tokens.txt"
            fixture.write_text(" ".join(map(str, tokens[:n + 1])) + "\n")
            logits = opts.output / f"{name}-logits.bin"
            log = opts.output / f"{name}.log"
            run_env = dict(env, STRATA_DUMP_FIRST_LOGITS=str(logits.resolve()))
            args = [str(exe), "--pack", str(opts.pack.resolve()), "--native", str(opts.native.resolve()),
                    "--tokens-file", str(fixture.resolve()), "--max-new", "1", "--spec", "2",
                    "--max-context", str(opts.max_context), "--prefill", str(opts.chunk),
                    "--no-prefill-borrow", "--expert-cache", str(opts.expert_cache),
                    "--expert-profile", str(opts.expert_profile.resolve()), "--expert-cache-per-layer",
                    "--pool-workers", "5", "--pcie-frac", "0", "--adapt-swaps", "0",
                    "--suffix-draft", "0", "--greedy", "--stats"]
            if opts.mtp:
                args += ["--mtp", str(opts.mtp.resolve())]
            start = time.monotonic()
            try:
                with log.open("w") as handle:
                    result = subprocess.run(args, cwd=opts.cwd, env=run_env, stdout=handle,
                                            stderr=subprocess.STDOUT, timeout=opts.timeout)
                rc = result.returncode
            except subprocess.TimeoutExpired:
                rc = 124
            text = log.read_text()
            record = {"name": name, "tokens": n, "repeat": repeat + 1, "args": args,
                      "exit_code": rc, "process_wall_seconds": time.monotonic() - start,
                      "log": log.name, "input_sha256": sha256(fixture)}
            for prefix, key in (("strata prefill transfer: ", "transfer"),
                                ("strata prefill phases: ", "phases")):
                found = [line[len(prefix):] for line in text.splitlines() if line.startswith(prefix)]
                if found:
                    record[key] = json.loads(found[-1])
            match = re.search(r"^prefill\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s", text, re.M)
            if match:
                record.update(reported_tokens=int(match[1]), cli_wall_ms=float(match[2]),
                              cli_tokens_per_second=float(match[3]), wall_ms=float(match[2]),
                              tokens_per_second=float(match[3]))
            chunk_match = re.search(r"strata generate: prefill \d+ tokens in (\d+) chunks", text)
            if chunk_match:
                record["chunks"] = int(chunk_match[1])
            if "transfer" in record:
                # The engine captures this before querying/printing profiling
                # records; its CLI timer also includes that reporting work.
                wall_ms = record["transfer"]["wall_ms"]
                record.update(wall_ms=wall_ms, tokens_per_second=1000 * n / wall_ms)
            output = re.search(r"^output\s*:\s*(.*)$", text, re.M)
            if output:
                record["output_ids"] = [int(t) for t in output[1].split()]
            if logits.exists():
                values = array.array("f")
                values.frombytes(logits.read_bytes())
                record.update(logits_sha256=sha256(logits), logits_count=len(values),
                              logits_finite=bool(values) and all(map(math.isfinite, values)))
            report["runs"].append(record)
            save()
            valid = (rc == 0 and record.get("reported_tokens") == n
                     and record.get("chunks") == 1 and record.get("logits_finite")
                     and (opts.mode == "wall" or record.get("transfer", {}).get("chunks") == 1)
                     and (opts.mode != "profile" or "phases" in record))
            if not valid:
                raise RuntimeError(f"Invalid or failed profile {name}; evidence saved to {target}")
            print(json.dumps({k: record[k] for k in ("name", "wall_ms", "tokens_per_second")}
                             | {k: v for k, v in record.get("transfer", {}).items() if k in
                                ("expert_bytes", "expert_copies", "dma_active_ms", "host_copy_worker_ms")}),
                  flush=True)
    medians = []
    for n in lengths:
        rows = [r for r in report["runs"] if r["tokens"] == n]
        median = {"tokens": n, "wall_ms": statistics.median(r["wall_ms"] for r in rows)}
        if opts.mode != "wall":
            median.update({key: statistics.median(r["transfer"][key] for r in rows)
                           for key in ("dma_active_ms", "expert_bytes")})
        medians.append(median)
    report["medians"] = medians
    if len(medians) >= 2:
        a, b = medians[-2:]
        slope = (b["wall_ms"] - a["wall_ms"]) / (b["tokens"] - a["tokens"])
        report["longest_pair"] = {
            "lengths": [a["tokens"], b["tokens"]], "marginal_ms_per_token": slope,
            "marginal_tokens_per_second": 1000 / slope if slope > 0 else None,
        }
        if opts.mode != "wall":
            report["longest_pair"].update(expert_bytes_change=b["expert_bytes"] - a["expert_bytes"],
                                          dma_active_ms_change=b["dma_active_ms"] - a["dma_active_ms"])
    save()
    print(json.dumps({"medians": medians, "longest_pair": report.get("longest_pair")}), flush=True)


if __name__ == "__main__":
    main()
