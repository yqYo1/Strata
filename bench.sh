#!/bin/sh
# Benchmark this Strata install the way it is used and write a report you can share (docs/COMMUNITY_BENCHMARKS.md).
# It starts the model with your settings (or uses the one already running), takes about 10 minutes and changes nothing.
# --prefill-only [--prefill-tokens N, 32768..131072] runs short warmup + fresh long prompts (thinking on).
# --prefill-scaling characterizes short/long input lengths, total prefill time and thinking-inclusive TTFT.
# --long includes the standard agent session; --long-generation adds two ~1K-output MTP workloads.
cd "$(dirname "$0")" || exit 1
if [ -x .venv/bin/python ]; then
  exec .venv/bin/python tools/strata_bench.py "$@"
fi
# The HTTP benchmark itself needs only Python's standard library.
exec python3 tools/strata_bench.py "$@"
