# Upstream reproduction audit

Baseline: Strata `99f3dbd0b21d1401b3769e0c0d963913607f380b` (0.1.38), with
its pinned GGML `3cf03257f219afbe7334045ff7c6a06ac68c627d`. The previous SYCL
implementation is preserved at `22e61b9`; its measurements and rejected trials
remain on `feature/sycl`. The new `feature/sycl-upstream-faithful` branch starts
from the upstream commit, not from the previous implementation.

The audit is **in progress**. No full-engine SYCL reproduction has been established.
Source identity, mathematical equivalence, device numerical parity, scheduling
parity, and measured performance are separate findings. A passing test against a
reference derived from the previous implementation does not establish upstream
parity. There is no suitable CUDA device/toolkit here for the full CUDA baseline;
source-derived checks must be labelled as such.

## Reproducible inventory

```
python3 tools/sycl/audit_upstream.py --ggml /path/to/pinned/llama.cpp \
  --output docs/sycl-audit/source-inventory.json
```

The fixed source snapshots contain 314 file paths: 220 identical, 36 changed,
58 added by the previous port, and none removed. The manifest includes complete
`src/`, `include/`, `cmake/`, root CMake, and vendored GGML file inventories, with
SHA-256 hashes. All semantic and runtime verdicts initially remain unverified.
It also records 381 pinned external files (a conservative CUDA/CPU/header superset)
and 53 literal `.cu` references in root CMake. These 53 include a parity program;
they are not a count of production kernels or a resolved build graph.

## Confirmed differences in the previous implementation

| Area | Upstream processing | Previous SYCL processing | Consequence / open check |
| --- | --- | --- | --- |
| MMQ activations | `ggml-cuda/mmq.cuh`: 128 values per 144-byte block, blocks transposed across rows; type-dependent D4/DS4/D2S6 metadata | `src/prefill/sycl/moe_mmq.cpp`: row-major ordinary 32-value Q8_1 blocks | Different representation; quantizer and consumer must be ported together |
| MMQ scale / codes | `quantize.cu:quantize_mmq_q8_1`: `127/amax`, multiply then `roundf`, reciprocal for scale; D4 retains FP32 scales for the mixed IQ expert types | Divide by `amax/127`; all scales stored in FP16 | Different rounding and precision; existing output is not the upstream numerical oracle |
| MMQ sums | Four adjacent values per lane, then XOR tree over 4 or 8 lanes | One value per lane, XOR tree over 32 lanes | Different FP addition order for formats using sums |
| Prefill transfer overlap | Separate copy and compute streams, ring events | `prefill.cpp` aliases `m.copy = m.cs` for SYCL | Original overlap removed; restore and diagnose dependencies under realistic sizes |
| QSA tensor scorer | CUDA `qsa_block_scores_tc` has a tensor implementation | SYCL function unconditionally returns false | Scalar fallback exists; accuracy and performance contract still needs comparison |
| Router cluster path | CUDA optional cluster implementation | SYCL cluster entry returns false | Check selection conditions, tie order and actual hot-path use |
| Host/device flag handoff | CUDA mapped-memory wait kernels | SYCL wait functions throw, host flow uses event-completed alternatives | Audit memory visibility, CPU/GPU scheduling, graph capture and cancellation together |
| Decode token graph | Upstream token capture can retain a whole token graph with host flag handoff | `session_capture_token` refuses SYCL; CLI defaults `no_token_graph=true` | A major scheduling difference, separate from kernel throughput; end-to-end impact not isolated |
| Verify window | Upstream captures a whole window with GPU/CPU handoff | SYCL splits it into event-completed segments and forces PCIe mode 0 | Compare accepted/rejected drafts, commit state, CPU/GPU overlap and cancellation |
| Host expert memory | CUDA can register/map the arena | SYCL uses ordinary host arena plus bounded staging | Direct GPU reads and copies have different requirements; preserve the visibility reproducers |
| Multi-GPU | Peer experts and layer split have upstream paths | SYCL peer admission is rejected and several tables bind to device 0 | Explicitly unsupported, not reproduced |
| Device timestamps | CUDA device timer stamps | SYCL entry throws | Diagnostic capability absent; not itself evidence of inference loss |
| Fused experts | CUDA integer matrix kernels, optional IQ fused path | Portable integer loops / separate XMX implementation | Match opt-in conditions, layouts, reductions and epilogues before choosing device tiles |
| CPU work split | Upstream defaults can send cache misses to GPU | Recorded local runs used `--pcie-frac 0` plus private CPU dispatch settings | Benchmark configuration differs independently of the GPU port |

`cvec_upload` returning false on an exception is **not** an unconditional unsupported
stub. The previous control-vector implementation exists and needs its own numerical,
state, lifetime and graph tests. Do not count every `return false` as a missing feature.

## Required coverage and exit evidence

Every row remains open until its evidence is attached. File inventories prevent
omissions; they do not replace entry-point and execution-path review.

| Processing family | Scope | Required evidence |
| --- | --- | --- |
| Build and dispatch | CUDA/HIP/default options, architecture guards, optional MMQ types, native experts, CPU ISA selection | Resolved target/source/flag graph for each supported variant; explicit exclusions |
| Runtime | Allocation kinds, pinned/mapped buffers, copies, default/nonblocking streams, events, error propagation | Ordering/lifetime tests; separate transfer/compute trace; no hidden synchronization |
| Graphs and handoff | Capture, replay, graph updates, CPU worker flags, multi-stream dependencies | Replay changes inputs/state correctly; handoff visibility; cancellation and teardown |
| Artifact/layout | GGUF types, pack offsets, expert strides, codebooks, scales, endianness, padding | Known block bytes against pinned GGML; all supported model types |
| Quantization | Decode Q8, prefill MMQ D4/DS4/D2S6, row selection, zero/padding, rounding, sums | Byte comparisons with source-derived oracle; actual CUDA differential still needed |
| Dense/low-bit products | Every CUDA GEMV/dequant/native MMVQ entry, shared experts, native dense/head, BLAS GEMM | Per-type dot and whole-product comparisons, matrix orientation, accumulation/epilogue order |
| MoE/routing | Top-k ties, gate weights, expert permutation, gather/scatter, bounds, fused/unfused | Same routes and weights; same grouped products including empty/uneven groups |
| Prefill orchestration | Chunking, stager/ring, MMQ eligibility, cache borrowing, transfers, fallback, output scatter | Intermediate layer tensors and transfer trace on the same model/token IDs |
| Attention/QSA | Q/K norms, RoPE/mRoPE, KV Q4/Q8, index pooling/top-k, prompt/decode/flash paths | Cache bytes, selected IDs and outputs; partial blocks and long-context cases |
| GR/GDN | Native/fused preprocess, recurrent updates, normalization, prefill state conversion | Every recurrent state and output over multiple tokens and chunk boundaries |
| PLE/ngram/control | Hash/lookup, PLE postops, ngram state, control vectors | Same indices, embedding/output and state after reuse and graph replay |
| Session/state | Layer order, residuals, conversation snapshots, prefix reuse, cache eviction, remote/peer experts | Token-by-token state comparison; misses/hits/reset/save/restore/cancel |
| Generation | Logits, penalties, sampler RNG/ties, stop/EOS, streaming, verify and MTP | Identical input IDs and settings; pre-sampler logits; accepted/rejected draft paths |
| CPU scheduling | Native expert kernels, thread split, NUMA/memory, CPU/GPU job partition | Upstream CPU path retained first; task counts and outputs; defaults documented |
| Full model | Qwen/Coder/Swift/Unsloth configurations and every enabled format | Explicit tested/untested matrix; PP/TG under identical prompt/cache/context/config |

Hardware-specific CUDA instructions need a SYCL implementation of their operation.
Changing the matrix tile is allowed; silently changing scale precision, disabling a
fast path, reducing cache capacity, or serializing streams is a separate deviation
that must stay visible. An unsupported feature remains unsupported until implemented
or explicitly excluded with its caller/fallback documented.

## Knowledge retained

Use the previous branch's `docs/SYCL_*`, `bench/results/`, Intel capability catalog,
and test failures as evidence and regression inputs. In particular, preserve the
queue-stall and host/device atomic-visibility reproducers. They justify focused
runtime work, not assuming all transfer overlap is impossible. The rejected
quantizer microbenchmarks at `bench/results/2026-10-04-sycl-quant-layout-trials/`
measured the old layout and do not measure upstream MMQ reproduction.

## First port component

The isolated [MMQ quantizer](../../src/sycl_upstream/mmq_quantize.cpp) implements
the upstream 2D activation layout. Its [validation record](mmq-quantizer.md)
separates source-reference agreement from still-missing CUDA-device comparisons.
The [first routed product consumer](mmq-product.md) now uses this layout for all nine
default MMQ formats. [Stream-K scheduling and fixup](mmq-stream-k.md) have also
been ported and tested. [Gather, SwiGLU and connected products](mmq-stages.md)
now cover all nine default formats and the Strata Q2 packed layout. The
[public MMQ adapter and scratch context](mmq-context.md) now have queue/lifetime
evidence. Full runtime and engine integration remain open; this is not a complete
translation-unit parity verdict.

## Runtime core

[Stream/event ordering](runtime-streams.md) now has an implementation and B570
validation, including a reproduced host-blocking barrier and a dependent-command
replacement. Its [pinned call inventory](runtime-inventory.json) keeps the remaining
CUDA API scope visible. Graphs, mapped-memory handoff and full engine integration
remain open.
