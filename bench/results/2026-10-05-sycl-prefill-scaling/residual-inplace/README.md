# In-place FP32 residual storage and full-context boundary checks

**GPU correctness and performance validation are pending. These changes must
not be merged on the strength of a successful build or short capacity probe.**

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
match the original reference exactly. Candidate equality and actual full
256K validation remain pending.

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
same-configuration control rather than a differently chunked reference.
The checker requires the actual compact-hc layout. It scans all head and
residual values for NaN/infinity directly: `STRATA_DBG_NAN` would disable the
layout under test. Persistent device-code caching defaults to zero for this
checker because of the reproduced oneAPI crash; an explicit inherited
`SYCL_CACHE_PERSISTENT` value overrides that diagnostic default.

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

The serve check also inspects every traced verify window. No window may go
beyond 262,144; the full request must actually execute KV cell 262,143, the last
allocated cell. A logical input/output count alone is insufficient for that
assertion. Normal four-token MTP runs with its default confidence clipping
disabled. Refused requests must not execute a verify window or emit `REUSED`,
tokens or logprobs. These trace assertions are pending the actual GPU run.

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
