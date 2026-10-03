# B570 MTP probability-floor measurements, 2026-10-03

`run.json` summarizes all four screen settings and six repeated-pair runs,
including hardware, software, flags, prompts, per-request timing, cache counts,
output hashes, the fixed-cache numerical comparison and HTTP checks. The screen's
0.5 and 0.9 trials also form pair one; eight fresh engine processes were measured
in total. No measured sample was discarded.

The engine body is commit `63e5604`; checkout `af21a91` adds only IQ4 screening
records. The adopted change is the local server's `--spec-min-p 0.9`.

Raw execution logs, per-trial JSON, generated comparison CSV files and workstation
configs are not tracked. Local copies of the original SYCL measurement captures
are preserved at `~/.local/state/strata-sycl/measurement-archive/6ddaadd/`, with
their repository-relative paths. The verification scripts retain the measured
workstation's `/tmp` paths. To rerun them, restore their inputs from that local
archive and run from the repository root after sourcing oneAPI. Use the local
venv Python for the numerical check, which requires NumPy. Full-vocabulary binary
dumps must be regenerated; their hashes and reference paths are in `run.json`.

Throughput excludes startup and prompt processing. Requests within each process
retain prompt and expert caches; the first request includes graph preparation.
Background CPU workloads were not isolated. The fixed-cache check covers 64
committed rows sharing the same history; it does not compare different rejected
drafts. HTTP checks cover functionality and clean shutdown, not speed. See
[the implementation notes](../../../docs/SYCL_IMPLEMENTATION.md) for the current
settings, measured limits and server start command.
