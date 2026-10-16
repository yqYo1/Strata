"""Validate both rings after loading oneAPI; run fixture.py with the serving venv first."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--exe", required=True)
parser.add_argument("--config", type=Path, default=Path.home() / ".local/share/strata-sycl/serve-config-iq3_s.json")
parser.add_argument("--tokens-file", default="/tmp/strata-sycl-goal-prefill-scale-routed-writing-tokens.txt")
args = parser.parse_args()
repo = Path(__file__).resolve().parents[3]
config = json.loads(args.config.read_text())
config["args"] += ["--no-prefill-borrow", "--prompt-cache", "0"]
env = dict(os.environ, ONEAPI_DEVICE_SELECTOR="level_zero:gpu", SYCL_CACHE_PERSISTENT="0",
           NEO_CACHE_PERSISTENT="1", STRATA_IQ_SINGLE_EXACT="1", STRATA_SYCL_MMQ_XMX="1",
           STRATA_SYCL_MMQ_XMX_TILE="8", STRATA_SYCL_MMQ_XMX_EXACT="1",
           STRATA_SYCL_MMQ_XMX_PACK="0", STRATA_SYCL_PREFILL_SAME_QUEUE="1")
for key in ("STRATA_PREFILL_ISSUER", "STRATA_SYCL_PREFILL_STAGER_CPU", "STRATA_SYCL_PREFILL_ISSUER_CPU"):
    env.pop(key, None)
with tempfile.TemporaryDirectory(prefix="strata-routed-config-") as directory:
    cfg = Path(directory) / "config.json"
    cfg.write_text(json.dumps(config))
    for ring in (96, 8):
        env["STRATA_PREFILL_RING"] = str(ring)
        command = [sys.executable, str(repo / "tools/sycl/serve_bench.py"),
                   "--exe", str(Path(args.exe).resolve()), "--config", str(cfg),
                   "--prompt", "writing=" + args.tokens_file, "--max-context", "8192",
                   "--prefill", "4096", "--cache", "512", "--adapt", "0", "--workers", "5",
                   "--tokens", "128", "--repeats", "2", "--cancel-during-prefill", "--timeout", "300",
                   "--output", f"/tmp/strata-sycl-goal-prefill-routed-full-serve-{ring}.json"]
        if ring == 8:
            command += ["--compare", "/tmp/strata-sycl-goal-prefill-routed-full-serve-96.json"]
        subprocess.run(command, cwd=repo, env=env, check=True)
