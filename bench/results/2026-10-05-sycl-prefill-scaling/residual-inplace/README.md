# In-place FP32 residual storage and full-context boundary checks

**2026-10-07: the newer lifetime/residency candidate did not finish its
262,144-cell CLI run within two hours. The small logged GPU probe afterward
passed without a reset. Full CLI last-cell and normal-MTP serve validation
remain pending for this candidate.** See the
[owned-debug evidence](../gpu-stall-diagnosis/owned-debug-20261007/README.md).
The successful older build's run below is retained as historical evidence.

**Short GPU residual comparisons and real full-256K CLI execution pass. Full
normal-MTP serve, checkpoint/cancellation and warm performance validation
remain pending; do not merge until they pass.**

The implementation adds `STRATA_PREFILL_LAYER_MAJOR_R_INPLACE=1` (off by
default). GPU-resident chunks compute directly in their persistent residual
rows. If every chunk stays in VRAM, the existing scratch allocation also stores
the first chunk, reducing the new allocation by that chunk's actual length.
Mixed GPU/RAM storage retains the scratch for the RAM tail. Its GPU prefix can
still skip residual copies. Arithmetic and GEMM shapes are unchanged, but
complete output and state equality must be measured before adoption.

The debug NaN scan now uses the actual final chunk length, including a short
first chunk. This avoids reading beyond an aliased final allocation. Temporary
residual pointers and the original scratch pointer are restored on every exit.

The SYCL generation driver also clips speculative verify windows and MTP draft
continuation to the available context. Serve's unconditional eight-token input
margin is removed; prompt plus requested output must fit the configured limit.
These changes need normal MTP, checkpoint and boundary integration tests.

## Checks completed

On Intel Arc B570 / Ryzen 5 5600X / 128 GiB RAM, the current precise SYCL JIT
build succeeds. Frozen candidate SHA-256:
`b1a7db41d1225d024d3455b9db58e71aebd13c8f88b8e737ae11049fe257b56d`.
Both Python controllers compile. The candidate correctly rejects these CLI
requests at `--max-context 262144`, with exit status 2 before loading the model:

- 262,144 prompt tokens plus one requested output token.
- 262,142 prompt tokens plus three requested output tokens.

These rejection checks do **not** validate prefill at the maximum length.
Their commands, environment and logs are in `boundary-262144`.

## GPU validation interrupted

The first candidate test (257 tokens, 128-token chunks) fails while uploading
native dense weights, before prefill or in-place storage executes. The kernel
records GuC timeouts, a device coredump, an unexpected engine-class warning and
repeated xe resets. A previously validated immutable control binary also stops
progressing during model load. No speed or output-equivalence result is claimed.
The cause has not been established; see `pending-gpu-checks` for raw records.

That status was historical. At the user's request, PID 1074147 was sent
SIGTERM and exited; SIGKILL was unnecessary. The user later rebound the GPU
and rebooted. The preserved xe dump identifies a kernel migration-queue
failure, and the installed module still contains a known recovery-order
defect. A separate SYCL persistent-cache CPU crash was reproduced in
llama-bench and avoided by disabling that cache. See
[the failure investigation](../gpu-stall-diagnosis/README.md) for the evidence
and the limits of each conclusion. The embedding service remains stopped.

After reboot, the immutable control completed 257-token layer-major prefill
and decode. Its full head, all residual rows and all dumped persistent state
match the original reference exactly. The candidate then completed all nine
short validation runs: six original-reference comparisons, a fresh short-first
control and two comparisons against that control. All head floats, residual
rows and dumped state bytes match their comparison targets. The new path
removes all counted VRAM residual copies; RAM residual traffic matches the
expected byte counts. No new xe events occurred in these runs. Records are in
[post-reboot-proof](post-reboot-proof/summary.json). An independent run of the
frozen legacy executable also matches the fresh short-first control; see
[legacy-short-first-equality](post-reboot-proof/legacy-short-first-equality.json).
At that point full-context execution remained pending; the subsequent CLI
result is below. Full normal-MTP serve and warm performance remain pending.

The subsequent [real full-context CLI run](post-reboot-proof/cli-262144.json)
passes at exactly 262,144 capacity: 262,142 inputs plus output IDs `40, 3172`.
All 248,320 head floats are finite and their hash is independently rechecked.
The process exits zero; the executable hash is unchanged and the kernel
records no new xe event during the run. Prefill processes 262,141 tokens in
1,340.298 seconds (195.58 token/s); the overall job takes 1,371.021 seconds.
This is one correctness observation with persistent device-code caching
disabled, not a paired warm speed estimate. The 1,370 memory samples peak at
59,935,868 KiB RSS and 10,048,828 KiB resident VRAM, with zero process swap and
no sampler errors. The [engine output](post-reboot-proof/cli-262144-engine.txt)
and [independent audit](post-reboot-proof/cli-262144-audit.json) are preserved.

The first full normal-MTP serve attempt fails before prefill: its 1,363,148,800
byte layer cache needs about 1.27 GiB, but only 0.43 GiB is available after
decode's expert-cache release. It returns no tokens and executes no verify
window. The [capacity failure](post-reboot-proof/serve-262144-capacity-failure.json)
retains the exact context and request. The solution needs more available
prefill memory; reducing the context would not satisfy this check. A BCS fault
and GuC recovery timeouts occur during/after its shutdown; they are recorded
separately in the failure investigation.

Candidate and control executables, the 256K fixture, original short equality
references and environment are preserved outside `/tmp` in:
`~/.local/state/strata-sycl/residual-inplace-recovery`.

## Required verification after recovery

Run from this directory, in order:

```sh
python3 check_residual.py
python3 check_full_context.py --stage serve --context 256 --gpu-rows 256
python3 check_full_context.py --stage cli
python3 check_full_context.py --stage serve
```

Repeat the CLI/serve checks for the combined opt-in GCC build using
`--executable ~/.local/state/strata-sycl/cpu-gcc-probe/engine-gcc`. Alternate
executables have separate report directories and their hashes are recorded.
Both builds require successful full-length GPU execution. GCC-build CLI
overflow refusals are recorded in `boundary-gcc-262144`; they do not establish
prefill/decoding correctness.

`check_residual.py` compares every head float, every FP32 residual row and all
dumped persistent state against original same-configuration references. It
includes all-GPU, mixed GPU/RAM, all-RAM, a single chunk, a short first chunk and
an incomplete final chunk. Removed VRAM copies and unchanged RAM traffic are
checked against transfer counters. Short-first variants compare with a fresh
same-configuration control rather than a differently chunked reference. That
control uses the frozen legacy executable, and generating its reference is
recorded separately from the eight equality comparisons.
The checker requires the actual compact-hc layout. It scans all head and
residual values for NaN/infinity directly: `STRATA_DBG_NAN` would disable the
layout under test. Persistent device-code caching defaults to zero for both
GPU checkers because of the reproduced oneAPI crash; an explicit inherited
`SYCL_CACHE_PERSISTENT` value overrides that diagnostic default.

The first real 256-token serve boundary run correctly failed the completion
guard: the raw coding prefix was an unfinished user message, and the model
emitted `im_end` after one token. The controller now reserves nine tokens
inside each requested input length for the verified assistant suffix used by
the normal-MTP/checkpoint helper. It keeps ordinary EOS handling and the same
context/request lengths; early EOS still fails. Source-fixture and actual
request hashes are recorded. The original failure is preserved in
`post-reboot-proof/serve-256-early-eos.json`.

The corrected real 256-token serve check passes with normal four-token MTP:
252 input plus four output tokens and 254 input plus two both finish with
`length` and execute KV cell 255. The two-token case's windows are `(253, 1)`
and `(254, 2)`. Full-prompt and overflow requests execute no verify window;
a valid request afterward succeeds. The 55 memory samples peak at
49,555,352 KiB RSS and 7,921,420 KiB VRAM, with zero process swap and no sampler
errors. See [serve-256](post-reboot-proof/serve-256.json). These are actual
small GPU integration results, not a full 262,144-cell pass. A BCS fault
occurred during the earlier EOS test's shutdown despite caching being disabled;
[the investigation](../gpu-stall-diagnosis/README.md) keeps that observation
separate from the cache crash.

`check_full_context.py` fixes the limit at 262,144, not 262,146. The CLI case
reads 262,142 input tokens and requests two output tokens. The normal-MTP serve
case reads 262,140 and requests four. Each must actually return enough tokens to
fill the advertised context. A second serve case reads 262,142 and requests two.
Its first verify window must be `(262141, 1)` and its second `(262142, 2)`. This
forces the normal four-token MTP window to shorten at the boundary regardless
of draft acceptance. Both requests must finish with `length`, matching prompt
and output counts in `DONE`. Serve then checks a completely full prompt, a
one-token overflow, and a valid request after refusal. Complete first heads and
all returned logprobs must be finite. Early EOS or insufficient output fails the
check rather than being reported as a full-context pass.

The CLI and serve checks inspect every traced verify window. CLI records a
window only after the verifier returns successfully and requires the exact
two windows `(262141, 1)` and `(262142, 2)`. No window may go
beyond 262,144; the full request must actually execute KV cell 262,143, the last
allocated cell. A logical input/output count alone is insufficient for that
assertion. Normal four-token MTP runs with its default confidence clipping
disabled. Refused requests must not execute a verify window or emit `REUSED`,
tokens or logprobs. These trace assertions pass at limit 256; their full-length
GPU validation remains pending.

`test_cli_capacity_trace.py` adds nine CPU stub cases for the completed CLI
window check and diagnostic environment. It accepts valid limits 64 and
262,144, rejects missing, shortened, oversized or incorrectly positioned CLI
windows, and checks boundary and serve modes. These are controller checks,
not physical GPU capacity measurements. `--diagnostics` applies the same
Level Zero and UR logging used by `sycl/tools/debug-run.py`.

`check_full_context_controller.py` tests the controller with a CPU protocol
stub at limits 64 and 262,144. It checks request lengths and deliberately emits
an overrun, a missing last cell, an incorrect shortened window, early EOS,
incorrect counts or finish reason, NaN logprobs, a missing fresh head, and
execution before refusal. This validates the test's pass/fail guards only. It
does not fill a real model context or provide GPU correctness evidence.
The recorded controller smoke run passes both valid limits and rejects all
nine injected faults; see [controller-smoke](controller-smoke/README.md).

`observe_memory.py` records process RSS/swap and DRM total/resident VRAM once
per second, with host available RAM/swap as context. It reads `/proc` and never
calls GPU APIs. Duplicate handles for one DRM client count once. CLI and serve
reports include the observed peaks and any sampler errors. The observer smoke
run records a 32 MiB CPU allocation over 14 samples; live xe fdinfo reading is
also checked against the stuck control's 1,716,344 KiB VRAM allocation. Those
checks validate the observer, not full-context memory use. The two capacity
refusals with observation are recorded in `boundary-observed-262144`.

CLI uses 1K chunks, matching the previous allocation-only 256K capacity probe.
Serve starts with 128-token chunks because it also allocates normal MTP and
verifier buffers. No full-length fit or speed is asserted for either path until
these runs finish. A capacity refusal needs investigation, not a smaller input
substituted for the maximum-length check.

Still required separately: normal checkpoint save/restore and cancellation
checks with in-place storage, paired warm performance measurements, and
successful full-length runs with the added trace/memory observations. None has
been established by a build, refusal test or observer smoke run.

The [full-context CPU PLE comparison](../ple-full-context/README.md) now
checks all output bytes at 262,141 prefill tokens across sixteen cases.
Whole-context PLE batching does not improve the observed direct reads.
Existing RAM-table mode gathers in 0.843–0.887 seconds after startup, whose
first table touch takes 42.043 seconds. These are CPU-only timings under
different file-cache states, not a model-level speed result.
`check_full_context.py --ple-io ram` retains the configured capacity and all
boundary assertions, records actual table startup/lock status, and uses a
separate `-ple-ram` directory. The updated fourteen-case CPU protocol check
also rejects a missing RAM startup record. Actual full-length RAM-table GPU
execution remains pending.
