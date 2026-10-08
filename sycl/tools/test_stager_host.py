#!/usr/bin/env python3
"""CPU fault-injection tests of the actual prefill Stager source fragment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parents[2]
source = root / "sycl/src/prefill/prefill.cpp"
env = dict(os.environ, ASAN_OPTIONS="detect_leaks=1:abort_on_error=1", UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
test = Path(__file__).with_name("stager_host_test.cpp")
control_ref = "f1dde98021477b11d39a69556cc08fbbd85928b7"
record = {"scope": "Actual extracted CPU Stager; fake allocation/device selection/deferred DMA queue; no SYCL runtime or GPU", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "tests": [], "passed": False, "sanitizer_options": {name:env[name] for name in ["ASAN_OPTIONS", "UBSAN_OPTIONS"]}}

def save():
    (out / "record.json").write_text(json.dumps(record, indent=2) + "\n")

def fragment(text):
    start = text.index("struct Stager {")
    end = text.index("// multi-GPU: the peer GPU's share", start)
    return text[start:end]

(out / "stager-under-test.hpp").write_text(fragment(source.read_text()))
command = ["g++", "-std=c++20", "-O1", "-g", "-pthread", "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-DSTRATA_STOP_WAIT=1", "-I" + str(root / "sycl/include"), "-I" + str(out), str(test), "-o", str(out / "stager-test")]
record["compile"] = command
save()
with (out / "build.log").open("w") as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=60, check=True)
for mode in ["finish-before-dma", "cancel-issuer", "reuse", "timeout"]:
    started = time.monotonic()
    result = subprocess.run([str(out / "stager-test"), mode], capture_output=True, text=True, timeout=20, env=env)
    (out / (mode + ".stdout")).write_text(result.stdout)
    (out / (mode + ".stderr")).write_text(result.stderr)
    expected = 1 if mode == "timeout" else 0
    assert result.returncode == expected, (mode, result.returncode, result.stderr)
    assert ("host wait timed out: test missing DMA completion" in result.stderr) if mode == "timeout" else ("PASS: " + mode in result.stdout)
    record["tests"].append({"mode": mode, "exit_code": result.returncode, "passed": True, "elapsed_seconds": time.monotonic() - started})
    save()
# Reproduce the former CPU hang with the pinned actual production control.
control_source = subprocess.check_output(["git", "show", control_ref + ":sycl/src/prefill/prefill.cpp"], cwd=root, text=True)
assert "std::atomic<bool> stopping" not in fragment(control_source)
control = out / "control"
control.mkdir()
(control / "stager-under-test.hpp").write_text(fragment(control_source))
control_command = [arg.replace("-DSTRATA_STOP_WAIT=1", "-DSTRATA_STOP_WAIT=0").replace("-I" + str(out), "-I" + str(control)).replace(str(out / "stager-test"), str(control / "stager-test")) for arg in command]
with (control / "build.log").open("w") as log:
    subprocess.run(control_command, stdout=log, stderr=subprocess.STDOUT, timeout=60, check=True)
try:
    subprocess.run([str(control / "stager-test"), "finish-before-dma"], capture_output=True, text=True, timeout=2, env=env)
    raise AssertionError("Old Stager unexpectedly completed without DMA acknowledgement")
except subprocess.TimeoutExpired as error:
    output = error.stdout or b""
    if isinstance(output, bytes):
        output = output.decode()
    assert "first two DMAs queued" in output
    (control / "finish.stdout").write_text(output)
    record["old_control"] = {"revision": subprocess.check_output(["git", "rev-parse", control_ref], cwd=root, text=True).strip(), "source_sha256": hashlib.sha256(control_source.encode()).hexdigest(), "expected_timeout_seconds": 2, "reproduced": True, "process_killed_by_test_runner": True}
dependencies = subprocess.check_output(["ldd", str(out / "stager-test")], text=True)
(out / "ldd.txt").write_text(dependencies)
assert all(name not in dependencies for name in ["libsycl", "libze_", "libur_"])
record["helper_sha256"] = {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in [test, Path(__file__), root / "sycl/include/strata/host_wait.hpp", root / "sycl/include/strata/host_completion.hpp"]}
record["fragment_sha256"] = hashlib.sha256((out / "stager-under-test.hpp").read_bytes()).hexdigest()
record["binary_sha256"] = hashlib.sha256((out / "stager-test").read_bytes()).hexdigest()
record["passed"] = True
save()
print(json.dumps(record, indent=2))
