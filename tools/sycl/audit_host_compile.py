#!/usr/bin/env python3
"""Inventory original host compile blockers; this is not a parity/pass gate.

Probes the literal C++ sources of strata_core/strata_engine/strata_prefill,
plus host-only pinned.cu and the generate executable. It does not resolve all
CMake options, compile CUDA kernels or test linking/runtime behavior.
"""
import argparse
import collections
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--compiler", default="icpx")
parser.add_argument("--ggml", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
cmake = (root / "CMakeLists.txt").read_text()
version = re.search(r"project\(strata VERSION ([0-9.]+)", cmake).group(1)
sources = set()
for target in ("strata_core", "strata_engine", "strata_prefill"):
    declaration = re.search(r"add_library\(" + target + r"\s+STATIC\s+([^)]*)\)", cmake)
    if not declaration:
        raise SystemExit("source target absent: " + target)
    sources.update(re.findall(r"src/[A-Za-z0-9_/]+\.cpp", declaration.group(1)))
sources.update(("src/core/pinned.cu", "src/program/generate.cpp"))
base = [args.compiler, "-std=c++20", "-ferror-limit=0", "-fsyntax-only", "-x", "c++",
        "-DSTRATA_NATIVE_EXPERTS=1", "-DSTRATA_PREFILL_MMQ=1", "-DSTRATA_PREFILL_FUSED=1",
        '-DSTRATA_VERSION="' + version + '"',
        "-Iinclude/strata/sycl_upstream/cuda", "-Iinclude",
        "-I" + str(args.ggml.resolve() / "ggml/include"),
        "-I" + str(args.ggml.resolve() / "ggml/src")]

def probe(source):
    command = base + [source]
    result = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=60)
    return {"path": source, "source_sha256": hashlib.sha256((root/source).read_bytes()).hexdigest(),
            "command": command, "exit_code": result.returncode,
            "missing_identifiers": dict(collections.Counter(re.findall(r"undeclared identifier '([^']+)'", result.stderr))),
            "unknown_types": dict(collections.Counter(re.findall(r"unknown type name '([^']+)'", result.stderr))),
            "stdout": result.stdout, "stderr": result.stderr}

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as workers:
    records = list(workers.map(probe, sorted(sources)))
header = root / "include/strata/sycl_upstream/cuda/cuda_runtime.h"
result = {"schema": 1, "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "scope": "Syntax only, original Linux host sources with native experts, MMQ and fused declarations enabled. No feature stubs injected; failures are open integration work.",
          "limitations": ["Not a resolved build graph for all configurations", "Does not compile CUDA device/GEMM translation units", "Successful syntax is not successful linkage or execution", "No performance measurement"],
          "frontend_header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
          "compiler_version": subprocess.check_output([args.compiler,"--version"],text=True),
          "sources": records,
          "passed": sum(r["exit_code"] == 0 for r in records),
          "failed": sum(r["exit_code"] != 0 for r in records)}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k:result[k] for k in ("passed","failed")}))
for record in records:
    print(record["path"], "syntax OK" if record["exit_code"] == 0 else record["missing_identifiers"])
