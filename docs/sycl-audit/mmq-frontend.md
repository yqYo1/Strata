# Original MMQ entry points on the source runtime

`strata_sycl_mmq_frontend` connects every public entry point from the original
`strata/prefill/moe_mmq.hpp` to the [source runtime](runtime-frontend.md).
The stream argument is an opaque `cudaStream_t`; null means the legacy default
stream. Callers can now use the original graph manager, copies and events around
MMQ without passing a SYCL queue into the original interface. This remains a
partial engine integration, with no new PP/TG measurement.

## Binding and lifetime

The new adapter and the earlier raw-queue component adapter compile the same
host source with different stream dispatch. They share one kernel library;
quantizer arithmetic, product tiles, stream-K planning, gather layouts and
SwiGLU expressions are unchanged. An executable must link exactly one adapter.
The raw-queue target remains for its existing independent component tests.

Each public launch resolves its stream through `cuda::submit`. Product main and
fixup are submitted together under the runtime lock and the context's scratch
lock, so other host submitters cannot insert work between them. Scratch remains
separate for each queue. Grouped gather returns its actual kernel completion
through an optional internal output argument; it adds no completion kernel or
host wait. Empty products/rows and invalid gather groups retain their original
no-launch checks before stream lookup.

The engine adapter retains upstream's fatal launch checks, including a previous
sticky error on the submitting host thread. Invalid handles produce a diagnostic
and exit code 1. A context must warm its scratch on each stream before recording
a graph that needs fixup storage. A cold allocation during capture is explicitly
rejected. Context, its scratch, and the caller's input/output buffers must outlive
all graph replays referencing them. Full prefill warmup and ownership still need
validation in the original engine caller.

The cold-capture test found a cleanup defect: `std::exit(1)` ran the frontend's
static domain destructor while a capture was active; its synchronization threw
and masked the launch error with `std::terminate`. Teardown now discards unfinished
capture definitions before waiting for already-submitted work and releasing
storage. The regression checks exit code 1 and the original scratch diagnostic.
This cleanup operation is only for teardown with no concurrent callers.

## Measured validation

Intel Arc B570, Ryzen 5 5600X, Linux, oneAPI 2026.1.1, Level Zero with the isolated
26.35.39758.10 driver / IGC 2.41.5 stack, 2026-10-04. Both JIT and `bmg_g21` AOT
pass the following connected checks:

- All nine default MMQ formats use two experts, five selected activation rows,
  19 weight rows, width 1024 and destination stride 23. Each runs once on the
  legacy default stream, then eight changed-input replays through the unchanged
  `core::CapturedGraph`. The named copy and compute streams fork/join using
  original event calls. Each capture contains exactly 13 nodes: five uploads,
  four event record/wait commands, quantization, main, fixup and result download.
- Replays change activation values, weight scales, expert bounds, input row IDs
  and destination row IDs. All 7,695 active products equal the independent pinned
  GGML CPU dequantizer plus double dot reference exactly for these fixtures;
  1,620 row-padding checks pass. Inputs deliberately quantize exactly. This result
  does not establish arbitrary-input bit equality or CUDA-device parity.
- Two host submitters enqueue 16 products with different activation scales behind
  a held host callback on one stream. Both finish submitting within two seconds
  before the gate opens. All 1,840 output/guard values match exactly, checking
  asynchronous submission and shared main/fixup scratch reuse.
- An 18-node graph runs native gather, grouped gather, full-layout Strata Q2
  repacking, split SwiGLU and iota with the original public API. Four changed-input
  replays verify 5,543,168 packed bytes/guards, 256 SwiGLU outputs against double
  `exp` (absolute bound `2e-6`), and 68 identity IDs. Earlier standalone tests
  retain broader alignment, shape and split/interleaved activation coverage.
- Separate bounded processes check invalid stream, pre-existing sticky error and
  cold capture. All three emit the expected diagnostic and exit 1.

The related ten CTest programs pass in both builds: public MMQ stages, context
lifetime, stream ordering, capture/replay, mapped handoff, memory lifetime,
original pinned shared backing, original host frontend, new MMQ frontend and
fatal errors. The large unchanged standalone product and quantizer suites were
not rerun for this binding change. Logs and source/binary hashes are in
[mmq-frontend-results.json](mmq-frontend-results.json).

## Remaining original host compilation

The new [host compile inventory](host-compile-results.json) probes 20 original
host translation units without editing them or injecting API implementations.
Native experts, MMQ and fused declarations are enabled, so the original prefill
fallback stubs do not hide their dependencies. Eleven files pass syntax and nine
fail. A syntax pass does not establish linking, execution or processing parity.
The original CUDA device/GEMM translation units and CPU kernel ISA builds are
outside this probe; this is not a resolved build graph for all options.

| Original host caller | Current compile blockers |
| --- | --- |
| `prefill.cpp` | Device get/set, free-memory query, timed events, peer copies |
| `layer.cpp`, `session.cpp` | Timed event creation and elapsed time |
| `expert_cache.cpp` | Free-memory query |
| `mtp.cpp` | Device get/set, free-memory query, graph upload |
| `verify.cpp` | Device get/set, graph upload and node inspection types/APIs |
| `remote_experts.cpp`, `peer_experts.cpp` | Device enumeration/selection, initialization/flags, peer admission/access, free memory |
| `generate.cpp` | Device properties/attributes/enumeration, timing, free memory, runtime/version metadata |

Compiler recovery can hide an API error behind an unknown argument type or
constant. The full diagnostics and [source call inventory](runtime-inventory.json)
remain authoritative; the table is a work list, not a complete API census.
No CUDA compute capability or runtime version is fabricated to admit this GPU.
The new MMQ binding still uses the frontend's single selected device domain;
multi-device behavior, optional formats, remaining prefill/decode kernel families,
full scheduling traces, layer/session tensors and actual model PP/TG are open.

Reproduce with the driver environment described in [runtime-memory.md](runtime-memory.md):

```sh
cmake --build build-upstream-sycl -j4
SYCL_CACHE_PERSISTENT=0 ctest --test-dir build-upstream-sycl \
  -R 'original_|mmq_public_stages|mmq_context_lifetime|runtime_' -V
# Repeat with build-upstream-sycl-aot for bmg_g21.
python3 tools/sycl/audit_host_compile.py --compiler icpx \
  --ggml /path/to/pinned/llama.cpp --output /tmp/host-compile-results.json
```
