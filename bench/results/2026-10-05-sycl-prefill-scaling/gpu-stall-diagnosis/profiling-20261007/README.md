# B570 profiling and completed workspace-reclamation capacity check

Measured on 2026-10-07 with the Arc B570 (10 GiB), Ryzen 5 5600X and
128 GiB RAM. The kernel was 7.0.0-38-generic, compute-runtime
26.31.39395.14, Level Zero loader 1.32.0 and oneAPI compiler 2026.1.1.
The executable was the previously validated workspace-reclamation candidate,
SHA256 `3f3ed0526848f6c7273a70da8802cbe00389b67791629c000c26399bc50823f2`.
Direct submission remained disabled, the Level Zero V2 adapter was retained,
copy offload remained disabled and the persistent SYCL cache remained off.
No GPU reset, driver, service or global security setting was changed.

## Profiler availability and first instrumentation check

Installed VTune 2026.4.0 build 632893 was attempted through its CLI.
Software Hotspots collection refused the current `ptrace_scope=1` setting.
XPU Offload collection, with CPU sampling and GPU hardware counters disabled,
refused this machine's CPU microarchitecture. Neither target was launched;
the receipts preserve the exact errors. These are observed collection
failures, not a claim that every VTune mode is unusable on every AMD host.

[Intel PTI/unitrace](https://github.com/intel/pti-gpu/tree/6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033/tools/unitrace)
was built privately from commit
`6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033`, with Level Zero and XPTI
support. Its Level Zero headers were v1.32.0; no installed loader or runtime
was replaced. The first build configuration failed because the shell's Nix
compiler wrapper and the system library paths conflicted. A fresh build
using explicit system GCC/G++ succeeded; both receipts are retained.
The binaries have SHA256
`5f90c453fbaf0b90a357a065cf4392f1ce3b157d5269af165c7cfe565cef1362`
and `c544dd2f6d2f6531cd529a9cc4ab9076b705eca8d63df56505e500942d128266`.

The first instrumented GPU test used the existing health executable with
the full diagnostic environment. Three rounds of 16,384 exact integer words
each passed H2D, kernel and D2H checks. The trace recorded all nine device
operations and host API activity; no new xe fault/reset appeared. Hardware
metric sampling/querying and driver tracing were not enabled.

## Real-model profiles and controls

Both model profiles kept the existing `STRATA_PREFILL_SYNC=1` workaround.
API logging and validation were removed after the logged smoke check and
previous exact-head validation. Each profiler run was supervised in its own
process session with a finite deadline and identity-checked cleanup.
Collection was paused during model loading, resumed after `READY`, and
paused before `QUIT`. All four real-model runs exited normally, used no
forced termination, left no owned process and recorded no new xe fault/reset.

The first pair used context 128, FP16 KV, 32-token chunks, layer-major mode
2, 600 expert slots and normal MTP. The four first/repeat/other/restored
requests each produced four tokens. IDs, printed logprobs and all 248,320
finite float logits matched the prior released control in both runs.
The first profiled request's explicit H2D copies carried 103.069 GB at
6.172 GB/s of summed copy execution time. Its 31.468 s observed request
envelope contained 17.853 s of copy intervals and 1.298 s of kernel intervals.
This short/chunk-32 configuration repeatedly loads layer weights; it is not
evidence for long-prompt transfer dominance. CPU trace parsing overlapped
part of the first unprofiled request, so that pair is only a preliminary
instrumentation-overhead comparison.

The second pair used 2,048 input tokens, context 4,096, INT8 KV,
1,024-token chunks, layer-major mode 1 and 128 expert slots. Main and MTP
decode weights were released and restored from RAM. The input ends with the
same nine-token assistant suffix used by the capacity controller. Both
runs produced the same four IDs, all printed logprobs and the complete
finite head. No CPU-heavy trace analysis overlapped this pair.

| Measured quantity | Profiled 2K request |
| --- | ---: |
| Observed request envelope | 47.931 s |
| Union of traced GPU command intervals | 17.527 s |
| Union of explicit copy intervals | 9.753 s |
| Union of kernel intervals | 7.773 s |
| Explicit H2D bytes and active-time rate | 49.272 GB; 6.163 GB/s |
| `zeCommandListHostSynchronize` calls; inclusive duration sum | 209,022; 8.405 s |
| `zeEventHostSynchronize` calls; inclusive duration sum | 24,389; 5.459 s |
| Gate/up expert dequantization kernel duration sum | 2.773 s |
| Down expert dequantization kernel duration sum | 1.856 s |

The engine reported prompt times of 36.600 s without unitrace and 47.590 s
with unitrace, a 30.0% increase for these individual runs. These still
include per-phase synchronization and are not clean unsynchronized
throughput measurements. Device-copy and kernel intervals can overlap;
host API time overlaps device work and nested UR/SYCL calls. Do not add
these columns. An interval without a traced GPU command is not proof of
CPU computation: it can include host dispatch, staging, file I/O,
synchronization and profiler overhead. Explicit copy bytes also omit PCIe
traffic caused by GPU kernels reading mapped host memory.

This profile gives two concrete candidates for subsequent measurements:
reduce the many host synchronization/submission calls, and reduce repeated
expert dequantization work. The 2K profile does not establish the cause of
the earlier hang or validate a proposed synchronization change. Proposed
changes still need exact output checks, unprofiled timing and all 256K gates.

An earlier raw-prefix 2K control generated a stop token after one output.
Its controller required four outputs and failed that assertion, then removed
its owned engine. That failed receipt is retained separately. No xe fault
occurred; it is not an engine/GPU-hang result.

## Workspace-reclamation 256K result

The already-running 262,144-cell normal-MTP diagnostic finished all 48
prefill layers and all 262,139 batched prompt tokens. It then failed an
8,388,608-byte physical allocation while restoring MTP decode weights,
after restoring the main expert cache and rebuilding verifier graphs.
The main-thread failure snapshot records SIGABRT from the restoration-failure
termination path, not a GPU completion wait. No new xe fault/reset was
recorded; both owned processes were removed. The subsequent logged
three-round exact-word GPU probe passed without reset.

| Trace point | Free device memory, bytes |
| --- | ---: |
| Prefill complete, temporary cache retained | 48,406,528 |
| Temporary buffers released | 1,411,559,424 |
| Main cache restored and verifier graphs rebuilt | 131,244,032 |
| MTP restoration allocation failed | 5,410,816 |

The 32 MiB reclamation therefore did not solve full-context restoration.
No output, complete head or subsequent capacity/refusal request was reached.
The 2,672.290 s elapsed time and final synchronization mark 20,839,685 are
diagnostic data, not throughput. The prior workspace archive preserves the
original `c7ba0822` source guard; the actually executed candidate was compiled
from `bdc0dea4`, as its build receipt records.

## Evidence

[The 2K analysis](unitrace-2048-analysis.json) and
[four-request analysis](unitrace-short-analysis.json) use interval unions
and preserve timing scope. Full private Chrome/Perfetto-compatible traces
are 542,056,567 and 966,980,342 bytes; their paths and SHA256 hashes are saved
in the analysis and timeline receipts. Large API/progress logs are also
retained privately, with hashes and bounded tails here.
[The full-context assessment](full-context-result.json) contains the memory
points, log digest and cleanup/health result. Scripts and source snapshots
are in `sources-used/`; the manifest covers every archived file.
