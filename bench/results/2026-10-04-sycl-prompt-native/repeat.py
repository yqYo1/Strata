"""Repeat final-binary off/on pairs after loading oneAPI; fixtures/config match this workstation."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--exe", required=True)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[3]
profile = repo / "bench/results/2026-10-04-sycl-prefill-events/profile.py"
env = dict(os.environ, PERF_EXE=str(Path(args.exe).resolve()),
           ONEAPI_DEVICE_SELECTOR="level_zero:gpu", SYCL_CACHE_PERSISTENT="0",
           NEO_CACHE_PERSISTENT="1", STRATA_IQ_SINGLE_EXACT="1",
           STRATA_SYCL_MMQ_XMX="1", STRATA_SYCL_MMQ_XMX_TILE="8",
           STRATA_SYCL_MMQ_XMX_EXACT="1", STRATA_SYCL_MMQ_XMX_PACK="0",
           STRATA_SYCL_PREFILL_SAME_QUEUE="1", STRATA_PREFILL_RING="8",
           STRATA_SYCL_MMQ_XMX_COMPACT="1", STRATA_SYCL_PROMPT_DIRECT="1")
for key in ("STRATA_PREFILL_DEVICE_TRACE", "STRATA_PREFILL_ISSUER",
            "STRATA_SYCL_PREFILL_STAGER_CPU", "STRATA_SYCL_PREFILL_ISSUER_CPU"):
    env.pop(key, None)

def run(size, minimum, label):
    env["STRATA_SYCL_PROMPT_NATIVE_KV_MIN"] = str(minimum)
    subprocess.run([sys.executable, str(profile), size, "prompt-native-threshold-" + label],
                   cwd=repo, env=env, check=True)

for size in ("4k", "short"):
    run(size, 0, "off1")
    run(size, 2048, "on1")
for value, label in ((2048,"on2"), (0,"off2"), (0,"off3"), (2048,"on3")):
    run("4k", value, label)
