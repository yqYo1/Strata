#!/usr/bin/env python3
"""Measure persistent real-model requests, repetition and cancellation recovery.

Run inside the initialized oneAPI environment. Prompt files contain token ids,
not text. Timings come from the engine's DONE record; initialization and total
request wall time are recorded separately. Raw protocol logs stay beside the
requested JSON output. This starts a local engine, not an HTTP server.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--exe", help="override the config executable")
    parser.add_argument("--prompt", action="append", required=True, metavar="NAME=TOKEN_FILE")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path, help="require identical completed-request ids")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--prefill", type=int)
    parser.add_argument("--max-context", type=int)
    parser.add_argument("--cache")
    parser.add_argument("--adapt", type=int)
    parser.add_argument("--spec-min-p", type=float)
    parser.add_argument("--pcie-frac", type=float)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if args.tokens < 8 or args.repeats < 1 or args.timeout <= 0:
        parser.error("tokens must be at least 8; repeats and timeout must be positive")
    prompts = []
    for entry in args.prompt:
        name, separator, filename = entry.partition("=")
        if not separator or not name:
            parser.error("each prompt must be NAME=TOKEN_FILE")
        ids = [int(x) for x in Path(filename).read_text().replace(",", " ").split()]
        if not ids or any(x < 0 for x in ids):
            parser.error("prompt ids must be nonempty and nonnegative")
        prompts.append((name, ids))
    if len({name for name, _ in prompts}) != len(prompts):
        parser.error("prompt names must be unique")
    config = json.loads(args.config.read_text())
    cwd = Path(config.get("cwd", os.getcwd())).resolve()
    executable = Path(args.exe).resolve() if args.exe else Path(config["exe"])
    if not executable.is_absolute():
        executable = cwd / executable
    args.exe = str(executable)
    environment = dict(config.get("env", {}), **os.environ)
    flags = config["args"].copy()
    if "--serve" not in flags:
        flags.append("--serve")
    for key, value in [("--pool-workers", args.workers), ("--prefill", args.prefill),
                       ("--max-context", args.max_context), ("--expert-cache", args.cache),
                       ("--adapt-swaps", args.adapt), ("--spec-min-p", args.spec_min_p),
                       ("--pcie-frac", args.pcie_frac)]:
        if value is not None:
            if key in flags:
                flags[flags.index(key) + 1] = str(value)
            else:
                flags.extend([key, str(value)])
    if "--stats" not in flags:
        flags.append("--stats")
    digest = hashlib.sha256()
    with open(args.exe, "rb") as binary:
        for block in iter(lambda: binary.read(1 << 20), b""):
            digest.update(block)
    result = dict(executable=args.exe, flags=flags, runs=[],
                  binary_sha256=digest.hexdigest(),
                  prompts=[dict(name=name, input_ids=ids) for name, ids in prompts],
                  environment={k: v for k, v in environment.items()
                               if k == "ONEAPI_DEVICE_SELECTOR" or k.startswith(
                                   ("STRATA_SYCL_", "STRATA_PREFILL_", "STRATA_STAGER_",
                                    "STRATA_NO_IQ", "STRATA_IQ_MT_MIN", "STRATA_POOL_SPIN_US", "UR_L0_"))})
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    messages = queue.Queue()
    with args.output.with_suffix(".stderr.log").open("w") as err, \
            args.output.with_suffix(".stdout.log").open("w") as raw:
        process = subprocess.Popen([args.exe] + flags, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1,
                                   cwd=cwd, env=environment)

        def reader():
            for line in process.stdout:
                raw.write(line)
                raw.flush()
                messages.put(line.strip())
            messages.put(None)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()

        def read():
            try:
                line = messages.get(timeout=args.timeout)
            except queue.Empty:
                raise TimeoutError(f"no engine protocol output for {args.timeout:g} seconds") from None
            if line is None:
                raise RuntimeError(f"engine ended ({process.poll()})")
            if line.startswith("ERR"):
                raise RuntimeError(line)
            return line

        def run(prompt, count, cancel=False):
            name, ids = prompt
            process.stdin.write(f"GEN {count} {','.join(map(str, ids))}\n")
            process.stdin.flush()
            start = time.monotonic()
            tokens, stopped = [], False
            while True:
                line = read()
                if line.startswith("T "):
                    tokens.append(int(line.split()[1]))
                    if cancel and not stopped:
                        process.stdin.write("STOP\n")
                        process.stdin.flush()
                        stopped = True
                if line.startswith("DONE "):
                    break
            fields = line.split()
            reason = fields[5]
            if int(fields[1]) != len(tokens) or len(tokens) > count:
                raise RuntimeError("DONE token count mismatch")
            allowed = ("cancel",) if cancel else ("length", "eos")
            if reason not in allowed:
                raise RuntimeError(f"unexpected finish reason: {reason}")
            if reason == "length" and len(tokens) != count:
                raise RuntimeError("incomplete length-limited request")
            row = dict(prompt=name, requested_tokens=count, output_ids=tokens, done=line,
                       wall_seconds=time.monotonic() - start, decode_ms=float(fields[4]),
                       decode_tok_s=len(tokens) * 1000 / float(fields[4]),
                       prompt_ms=float(fields[3]), reused_prompt_tokens=int(fields[8]),
                       cache_hits=int(fields[9]), cache_lookups=int(fields[10]), cancelled=cancel)
            result["runs"].append(row)
            save()
            print({k: v for k, v in row.items() if k != "output_ids"}, flush=True)
            return row

        try:
            start = time.monotonic()
            while True:
                ready = read()
                if ready.startswith("READY "):
                    break
            result.update(ready=ready, startup_seconds=time.monotonic() - start)
            for prompt in prompts:
                for _ in range(args.repeats):
                    run(prompt, args.tokens)
            run(prompts[0], args.tokens, cancel=True)
            last = run(prompts[0], 8)
            if last["output_ids"] != result["runs"][0]["output_ids"][:8]:
                raise RuntimeError("cancellation recovery changed the first eight ids")
            result["cancel_recovery_first_eight_ids"] = True
            process.stdin.write("QUIT\n")
            process.stdin.flush()
            process.wait(timeout=60)
            if process.returncode != 0:
                raise RuntimeError(f"engine exit {process.returncode}")
            result["exit_code"] = process.returncode
            if args.compare:
                previous = [r for r in json.loads(args.compare.read_text())["runs"] if not r["cancelled"]]
                current = [r for r in result["runs"] if not r["cancelled"]]
                if len(previous) != len(current) or any(
                        (a["prompt"], a["requested_tokens"], a["output_ids"]) !=
                        (b["prompt"], b["requested_tokens"], b["output_ids"])
                        for a, b in zip(previous, current)):
                    raise RuntimeError("completed output differs from comparison run")
                result["comparison_ids_equal"] = True
        except Exception as exc:
            result["error"] = str(exc)
            raise
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            thread.join(timeout=10)
            save()


if __name__ == "__main__":
    main()
