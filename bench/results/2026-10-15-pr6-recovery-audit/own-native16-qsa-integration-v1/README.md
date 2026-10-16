<!-- SPDX-License-Identifier: MIT -->
# R818: bounded native BF16 QSA prefill proposal

**SOURCE ONLY, NOT INTEGRATED. Every compile/link/build/test/GPU/numerical,
model-quality, fault, physical 262144-context, RESTORE and performance gate is
UNRUN.** Main exclusively owns execution after the current oneDNN and R810
cohorts close, under the shared measurement lock. This package makes no speed,
quality, allocation reduction or backend qualification claim.

`proposal.patch` is the integration artifact. `source-manifest.json` pins the
exact original/proposed file hashes; the supplied source boundary is `9c1a2020`.
The authoritative base is the fully hashed source snapshot, not a claim that
the current report-only HEAD has that short revision. No Git command was run.
The package does not store another whole `prefill.cpp` or another whole source
tree. `assemble.py` constructs the unified diff in memory and refuses changed
original hashes; it writes only inside this directory. Existing product files,
packages, builds, tests and environments were not changed.

Main's R787 review was read first:
`../own-native16-attention-main-review-v1.ja.md`. The old R787 base
`375a6d6ee905859ef176997262e3baef6e739ad2` is not this integration base. The actual
appended BF16 attention core is reused: its `launch_prompt_bf16` function is
byte-identical, SHA256
`c0563f01fd3a69e6c7badfbfbd7a06dcaeff475f468bc9eca0bb747720f30ba8`.
The current old FP32 attention file remains a byte-identical prefix. The R787
endpoint membership helper is replaced by `validate_device_span`, using the
current live Strata allocation extent registry and exact queue context/device.
A separate `qsa_prompt_attn_bf16_validate` entry shares all capability/span
checks with the submitting entry and never submits or waits.

The opt-in is `STRATA_SYCL_PREFILL_NATIVE16_QSA=1`, together with existing
`STRATA_SYCL_PREFILL_NATIVE16=bf16` and long layer-major prefill. Zero/unset
retains the original QSA route, including native16 HC/MoE with old FP32 QSA.
Short prefill, decode, GDN internals, recurrent state, typed BF16 residual
callbacks and the final MTP/decode FP32 boundary keep their existing route.
No new GEMM is imported; R810 is neither read as an implementation source nor
assumed qualified or faster, and R816's live package was not read.

Supported v1 conditions are deliberately narrow:

- Existing layer-major geometry, token chunks 1..8192 and session limits through
  physical 262144 cells; all existing layer-major restrictions still apply.
- Text positions, `STRATA_ROPE_TABLE=1`, the registered tables must be exactly
  the current session-owned tables, with the same `max_cells`. Scaling is baked
  into those existing tables. M-RoPE and analytic/table fallback are rejected.
- Resident cache (`QsaState::kv_mode==0`), unrotated INT8 K and V with per-64
  FP16 scales. This is attention **Mode1**, distinct from state streaming mode1.
  The attention core uses Mapping0, WG128/SG16 and 12 heads plus four padding
  rows per KV group. `STRATA_SYCL_PROMPT_K_CELL_MAJOR` must be unset/zero;
  `STRATA_PROMPT_ATTN_OLD` must be unset (its old semantics are presence-based).
  Other old attention optimization
  flags do not select this explicit typed core.
- `n_slots==n_pages`; no host mirror, streaming, ring, Q4, hybrid or FWHT.
  Unsupported states/device geometry/domains reject in `run_layer_major`
  before this request's layer submissions. A later typed attention rejection
  also fails explicitly; it never feeds BF16 buffers to the old FP32 fallback.

Streamed support is intentionally held out: the current extent registry covers
device allocations, whereas a typed append to host mirrors needs a separately
reviewed host-allocation owner/range contract. This proposal introduces neither
that mechanism nor a CPU math fallback.

The actual operator chain is:

1. Existing HC mix produces FP16 `mixed_h` and BF16 `mixed_bf` directly.
2. Existing `native_proj16` dequantizes each native GGUF Q/K/V weight tile into
   the existing reusable FP16 `Gemm::scratch()`, then calls `f16_out16` with
   BF16 output. Input and dequantized weights are FP16; output is explicitly
   BF16. There is no added persistent weight bank or decode-weight copy.
3. `native16::qsa::norm_rope_split` consumes BF16 projections directly. A WG256
   head computes a scaled FP32 RMS reduction, FP32 weight multiply and the
   first 64 dimensions' NeoX rotation, then stores BF16. For Q the input head
   stride512 skips the gate half; Q/full remains unchanged for the later gate.
   For K raw and transformed tensors use disjoint halves of the old K slot.
4. `append_q8` consumes BF16 K/V, performs FP32 per-64 max and rounding, and
   writes INT8 codes plus FP16 scales directly on device. Its explicit mutable
   `Q8WritePools` comes from owner pointers, without casting away a read API's
   constness. Nonfinite groups write a NaN scale and zero codes; no NaN/Inf is
   converted to integer or silently made into a healthy cache entry.
5. Indexer K/Q projections, normalization, RoPE, pooled keys, selection scores
   and top-k retain the existing FP32 path. This is a named precision boundary,
   not a claim that all activations are 16-bit.
6. R787 BF16 attention consumes BF16 Q and writes BF16 attention. Q and P each
   use **one FP16 local fragment**, with per-head/pool scaling; the old hi/lo
   FP32 emulation is absent. Scores, online softmax and accumulators are FP32.
   Underflow/range and model quality need separate numerical gates.
7. `gate` consumes BF16 attention and the BF16 projection's gate half, performs
   local FP32 sigmoid/multiply and stores FP16 directly. Existing
   `native_proj16` consumes that FP16 tensor and dequantized FP16 output weights
   to produce BF16 `bo_b16`, then the existing BF16 residual writer consumes it.

There is no full FP32 QSA hidden/projection/attention tensor restoration in this
typed chain. BF16-to-FP32 widening is scalar/register work; BF16-to-scaled-FP16
Q and FP16 probability fragments are local attention storage, and the gate's
BF16-to-FP16 output is a declared device conversion. The GEMM backend and its
arithmetic/provider execution must still be qualified independently by Main.

The unchanged QSA indexer seam was audited at source level as well.
`Gemm::bf16` in `src/prefill/sycl/gemm.cpp` passes typed BF16 inputs/weights to
oneMKL and requests FP32 output; its wrapper adds no whole-hidden FP32 restore.
The provider's internal GPU arithmetic or CPU emulation is **not proved** by
that call or by an attached GPU queue, and was not loaded/inspected here.
`qsa_block_scores_tc` currently returns false on SYCL, then the caller launches
the explicit SYCL FP32 `qsa_block_scores` kernel, followed by device top-k.
That is a GPU scalar-score alternative, not a CPU fallback or a qualified
tensor-core selector. The native indexer uses GPU append/reduction kernels;
its old batch option may loop on the host to submit a kernel per token, while
the existing device-batch option performs its plan in bounded GPU launches.
Host frequency constants/step records and launch loops remain existing
preparation/control work. No new CPU grouping or inference math is added.
Provider arithmetic placement and CPU time in submission versus completion
polling remain Main-owned attribution gates; no bottleneck claim follows from
a saturated thread or this source audit.

The fused RMS is a numerical change: it scales before squaring, accumulates
normalized squares in FP32, combines norm and rotation without an intermediate
BF16 normalized store, and narrows once on output. Nonfinite projection inputs
diagnose NaN for the normalized head. Zero source with epsilon0 is NaN, while
zero source with positive epsilon is zero. Weights and rotary tables retain
their own NaN/Inf behavior. There is no new clamp or saturation. Existing expert
hidden saturation and existing GEMM/attention tests are not edited or relaxed.
Using session tables is explicit; match `STRATA_ROPE_TABLE=1` in reference arms
to avoid mixing this change with analytic RoPE differences.

The following are per-token logical view sizes, excluding allocator guards.
Multiplying by the actual carved T gives bytes. All QSA/MoE scratch reuse is
ordered on the existing in-order compute stream.

| View | Owner reservation | Bytes/token | Live range |
| --- | --- | ---: | --- |
| BF16 K raw | first half of old FP32 Kc | 1024 | K projection -> K norm/RoPE |
| BF16 K transformed | second half of old FP32 Kc | 1024 | K norm/RoPE -> Q8 append |
| BF16 V | first half of old FP32 Vc | 1024 | V projection -> Q8 append |
| BF16 Q/full + gate | first half of old FP32 Qf | 24576 | Q projection -> attention gate |
| BF16 Q | first half of old FP32 q | 12288 | fused Q norm/RoPE/split -> attention |
| BF16 attention | first half of old FP32 attn | 12288 | attention -> gate |
| FP16 gated attention | existing attn_h | 12288 | gate -> output GEMM |
| FP32 idx_raw / q_idx | existing separate slots | 512 / 2048 | indexer projection -> append/selection |
| BF16 block output | existing bo_b16 view | 5120 | output GEMM -> residual update |

Raw K and transformed K are adjacent, never overlapping. No other typed QSA
view is assigned another live tensor's bytes. Q/full must survive both the
attention and gate. `attn_h` is disjoint from BF16 attention, so the gate does
not need cross-element in-place ordering. After the attention half writes the
residual, the existing MoE region may overwrite all these attention aliases.
`carve(T)` rebuilds every typed alias for that actual T, including tails.

**Physical arena reservation is unchanged.** The listed old K/V/Q/full/Q/attn
and FP16 gate slots reserve 114688*T bytes (896 MiB at T8192), before indexer,
selection and alignment. The new declared typed views total 64512*T bytes
(504 MiB at T8192); their contents do not imply those bytes are concurrently
live. A Q/full + Q + attention or Q/full + attention + gate logical live set is
49152*T bytes (384 MiB at T8192). No freed allocation or smaller shared region
is claimed: the original `max(GDN,QSA,MoE)` allocator and float slots remain.
No scratch alias depends on reduced allocation reservation.

Capacity/collective source review:

- `qsa_state_init` allocates the physical pools using `p.slots`, publishes
  `n_slots=p.slots`, and allocates `n_pages=p.pages` table entries. The caller
  passes `{n_pages,n_slots}`. Pool codes use
  `n_slots*2*page_size*256` bytes each; scales use
  `n_slots*2*page_size*4*2` bytes each; the table uses `n_pages*4` bytes.
  Session rotary tables use `max_cells*32*4` bytes each. These are actual owner
  layouts, not logical maximum-context guesses about physical pool allocation.
- The allocation registry proves the complete range belongs to one live
  registered device allocation in the queue domain. State/arena ownership still
  proves the individual suballocation layout and lifetime; range validation
  cannot prove that a pointer refers to the intended logical tensor.
- Appending checks token/position byte products, logical table limits, all
  output/output and output/input overlaps, type alignment and allocation spans
  before submission. Each code/scale element has one writer for the owning
  resident identity map. The independent API contract requires an injective
  write mapping; concurrent table mutation or aliased physical write cells
  are invalid. Negative/out-of-physical-range page entries are skipped after
  all uniform group reductions, without page/pool access.
- The norm has one WG256 per real (token,head), with WG-wide reductions reached
  by all lanes. It has no conditional barriers/collective calls. The Q8 append
  has WG64/SG32 per token/side/head/group64; every lane performs both reductions
  before the uniform page-write condition. The gate has one element owner.
  No T padding is read or written. No inference math is performed on the CPU.
- The unchanged R787 core has WG128/SG16, declared SLM32336 bytes and a
  capability floor36432 bytes. All WG barriers, shared KV tile handoffs,
  subgroup matrix operations, head padding and missing-ID/page semantics are
  unchanged. These are source properties; compiler resources, device behavior
  and occupancy are UNRUN. Its width is per-query and cap-bounded. Causality is
  still the selector's responsibility, including future cells already stored.

`files/tests/sycl/qsa_native16.cpp` is an original, small FP64/dyadic oracle.
Its arithmetic IEEE encoders do not call the production conversion helpers.
Source cases cover Q and K with T1/3/17, strided head/gate separation, positive
epsilon and epsilon0, exact dyadic rotations, BF16 very large/subnormal range,
NaN source, gate NaN/Inf and half overflow/underflow, page-boundary T3 at pos3,
independent logical/physical page counts, missing and out-of-range pages,
guard/padding preservation, immutable original inputs and recovery. Validation
cases include overlap, insufficient registered range, wrong context,
misalignment, bad stride/position/epsilon and folded outputs. Its finite RMS
screen is `2*BF16_spacing(FP64 result)+2e-6`; exact dyadic/gate/Q8 cases require
rounded bits/codes, with separate explicit nonfinite diagnostics. This new
component screen is not a model quality tolerance.

`files/tests/sycl/prompt_native16.cpp` is explicitly derived from the own-MIT
R787 fixture at
`../own-native16-attention-v1/files/tests/sycl/prompt_native16.cpp`, original
SHA256 `d3791231919fe137cd4e3349d0d97ea1c3df44842d3bd2b6eb78cd9f45f396ab`.
Its FP64 attention oracle, every numerical test/tolerance, width/ID masks,
guard/capacity cases and batch/repeat identity checks are retained. Only
exception lifetime is hardened: a checked drain precedes the first upload, and
upload/readback catch paths drain while referenced host owners remain alive.
Both fixtures fail-stop on an unsuccessful drain, before unwinding owners.
Neither changes environment variables. CMake adds only manual,
`EXCLUDE_FROM_ALL` targets `sycl_native16_qsa_test` and
`sycl_prompt_native16_test`; no automatic test registration or benchmark
controller is added.

After full source review, Main can integrate the unified patch into a private
tree at the pinned base and select an independently qualified backend below
`Gemm::*_out16`. A suggested Main sequence, all currently UNRUN, is strict
component compilation and numerical checks; R787 attention smoke and full
fixture; old FP32/exact GEMM regressions without weakened criteria; a matched
native16 QSA0/QSA1 model check with identical backend and RoPE/cache settings;
then natural long-input quality, physical262144/RESTORE and repeated clean
whole-dependency performance samples. Main supplies immutable child
environments, deadlines, driver/fault receipts and retention records. Reject
unsupported modes before layer submission and verify typed dispatch from the
actual configured SYCL resolver (`cmake/sycl_backend.cmake` ->
`strata_prefill`/`strata_kernels` -> layer-major/QSA calls).

Current sparse 32K quality evidence is not acceptance for this proposal or for
general BF16 quality: the existing FP32 controls were identical and BF16 top1
was 59/64, with a lower sampled mean NLL. No quality tolerance is accepted;
matching generated text is not used as a requirement. GDN, PLE and other
remaining FP32 boundaries are unchanged and separately documented upstream in
`docs/SYCL_NATIVE16_PREFILL.md`.

All new package files and proposed project additions are MIT. `provenance.json`
records complete derivation and the indexed pinned SGLang/vLLM dtype design
contracts. No framework kernel code, XeStrata source/helper/test/proposal,
unreviewed LLVM fork, R810 kernel or R816 source was imported. `manifest.json`
pins the deliverable payload and all unchanged execution statuses.
