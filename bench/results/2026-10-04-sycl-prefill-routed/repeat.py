"""Run after loading oneAPI's environment; real model/config paths match this workstation."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--exe", required=True)
args = parser.parse_args()
profile = Path(__file__).resolve().parent.parent / "2026-10-04-sycl-prefill-events/profile.py"
env = dict(os.environ, PERF_EXE=str(Path(args.exe).resolve()),
           ONEAPI_DEVICE_SELECTOR="level_zero:gpu", SYCL_CACHE_PERSISTENT="0",
           NEO_CACHE_PERSISTENT="1", STRATA_IQ_SINGLE_EXACT="1",
           STRATA_SYCL_MMQ_XMX="1", STRATA_SYCL_MMQ_XMX_TILE="8",
           STRATA_SYCL_MMQ_XMX_EXACT="1", STRATA_SYCL_MMQ_XMX_PACK="0",
           STRATA_SYCL_PREFILL_SAME_QUEUE="1")
for key in ("STRATA_PREFILL_ISSUER", "STRATA_SYCL_PREFILL_STAGER_CPU", "STRATA_SYCL_PREFILL_ISSUER_CPU"):
    env.pop(key, None)
for ring, label in ((96,"off1"), (8,"on1"), (8,"on2"), (96,"off2"), (96,"off3"), (8,"on3")):
    env["STRATA_PREFILL_RING"] = str(ring)
    subprocess.run([sys.executable, str(profile), "4k", "routed-" + label], env=env, check=True)
