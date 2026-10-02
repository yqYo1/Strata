# SYCL backend implementation

The backend is under development. The runtime and device arena run on Intel Arc
B570; the SYCL build runs short native-pack decode with CPU experts.
Short-prompt logits have been compared with a CPU model reference, as recorded
below. Broad quality validation and the target inference performance remain open.
The implementation follows [the port research](SYCL_RESEARCH.md) and
[the operation inventory](SYCL_BATTLEMAGE_OPERATIONS.md).

## Build and validate

On Linux with Intel oneAPI DPC++ and oneMKL installed:

```sh
source /opt/intel/oneapi/setvars.sh
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
cmake -S . -B build-sycl -DCMAKE_CXX_COMPILER=icpx \
  -DSTRATA_ENABLE_SYCL=ON -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=OFF
cmake --build build-sycl --target strata-device sycl_host_engine_test sycl_compat_test sycl_runtime_test sycl_kernels_test sycl_gdn_test sycl_fused_gdn_test sycl_sampler_test sycl_coupled_sampler_test sycl_sequence_test sycl_native_flash_attn_test sycl_fused_gr_test sycl_quantize_test sycl_bf16_test sycl_mmvq_test sycl_gr_test sycl_rope_test sycl_kv_test sycl_attention_test sycl_qsa_index_test sycl_decode_attention_test sycl_ple_test sycl_moe_test -j2
ctest --test-dir build-sycl -R '^sycl_' --output-on-failure
build-sycl/strata-device --list-devices
```

`STRATA_SYCL_TESTS` controls the independent SYCL tests. The native CPU expert
library is disabled in this command to avoid fetching ggml. Enable it with
`-DSTRATA_NATIVE_EXPERTS=ON`; `STRATA_GGML_DIR` can point to a checkout at the
commit pinned in the root CMake file. The shared `strata_engine` static library
and inference executable compile with that configuration. CUDA, HIP and SYCL
are mutually exclusive.

For the separate large-address check, with at least 5 GiB of free VRAM:

```sh
timeout 60 build-sycl/sycl_runtime_test --large
```

This allocates 4 GiB + 64 KiB, writes different 64-bit values below and above
the 4 GiB boundary, and reads both back. It does not measure memory bandwidth.

## Runtime contract

- Compute and transfer queues use the same device and context. Each queue is
  ordered; dependencies between queues use explicit events.
- CPU access to host USM follows completion of its producing GPU command. No
  concurrent CPU/GPU atomic access to host/shared USM is required.
- Device and bounded host-staging allocations own their context lifetime and
  complete pending work before freeing storage. Bulk expert RAM will remain
  ordinary host memory rather than a shared-USM allocation.
- Asynchronous errors are retained and rethrown by runtime checks and waits.
  A timeout reports incomplete work; it does not cancel commands or authorize
  releasing their buffers.
- `DeviceArena` preserves the engine's bump-allocation interface, with checked
  arithmetic and alignment of actual pointers. Memory planning queries free
  VRAM rather than treating total capacity as available memory.

The shared host engine has a SYCL implementation of its CUDA-named runtime
API. Stream handles are in-order SYCL queues; event waits cross queues in the
same context. Graph capture/finalization/replay uses SYCL command graphs.
The default compute queue, like explicit engine streams, enables device event
profiling. Startup PCIe calibration previously failed because the default queue
lacked that property; a regression test now times copies on that queue.
Events recorded inside a graph report completion of the entire graph, so a CPU
handoff cannot precede a later copy in that graph. Their elapsed-time queries
are rejected. Graph execution is serialized across repeated launches, including
launches on different queues. Arbitrary host-memory registration is unsupported;
only host-USM allocations have a mapped device alias. This adapter currently
supports device ordinal zero for engine execution.

The shared per-layer scheduler reads the host doorbell only after its completion
event succeeds. Whole-token polling graphs reject SYCL use. Speculative windows
use separate graphs with completed events at CPU expert boundaries, as recorded
below. The ordinary expert arena skips host
registration, and blocking cache fills reuse a host-USM staging buffer sized to
one largest expert.

Prefill projections use oneMKL's SYCL BLAS with FP16 or BF16 inputs and FP32
accumulation. Native weights are expanded into reusable FP16 scratch in row
slices when a matrix does not fit at once. The caller's output row stride and
accumulation coefficient are preserved. oneMKL manages its own internal
workspace; the engine's CUDA workspace argument is not used by this path.
Prefill normalization, residual updates, mixing, gates, conversions and MoE
output combination are also implemented. Prompt GDN uses the same recurrence as
decode over the whole chunk, retaining each state column in registers. Paged KV
append, scaled RoPE and canonical expert expansion reuse the decode formats.
Prompt attention currently selects the batched FP32 split-attention fallback;
a matrix attention kernel and end-to-end prompt validation remain outstanding.
These post-operations disable relaxed
floating-point transformations: with the compiler's default device settings,
the two normalization paths disagreed in their BF16 correction component.

## Validation recorded on 2026-10-02

Intel Arc B570, DPC++ 2026.1.1, Level Zero driver `1.17.39395+14`, normal driver
settings; see the research document for the machine configuration:

- 64 changing-input CPU → transfer queue → compute queue → CPU round trips,
  each with 4,097 integer elements: all results matched.
- Invalid subranges, impossible device allocation size and move ownership:
  expected rejection/ownership behavior.
- A deliberately throwing host task: the asynchronous error reached the caller
  and remained observable on subsequent checks.
- Bounded completion wait and the existing `strata-device --selftest`: passed.
- Engine startup PCIe probe: 6.5 GB/s host-to-device for four 256 MiB copies
  after warm-up, measured by device events. This selected the existing 0.14
  PCIe-share heuristic instead of its unmeasured 0.55 fallback; the selected
  share still needs end-to-end calibration.
- oneMKL GEMM: 22 CPU-reference cases passed, including FP16 and BF16,
  T/N/K of 1/17/32, 17/259/320 and 8/640/2560, beta 0/0.5/1, output row
  padding, and Q8_0 weights split across six dequantization slices at K of
  32, 64, 256 and 640. Odd row slices include 2-byte-aligned packed inputs.
  Native dequantization checks the format's block width; a blanket 256-value
  restriction had rejected valid shared-expert projections. The error bound
  was `4e-6 * (1 + sum(abs(reference terms)))`. A 40-product pipeline with an
  eight-slot device ring, sixteen-slot host ring and re-recorded events also
  preserved every product across asynchronous copies and GEMMs.
- Prefill post-operations: three-token normalization and recomputation paths
  produced identical BF16 high/correction images and mixed FP32 values.
  Residual updates, strided RMS normalization, FP16 SwiGLU saturation,
  GDN gates, attention splitting/gating and MoE combination passed CPU checks.
- Prompt GDN at the real 16 key / 48 value heads and head width 128:
  17 tokens together, 17 individual decode steps and chunks of 5 + 12 produced
  identical convolution outputs/history, normalized outputs and recurrent
  state. The FP16 output image matched conversion of the FP32 result.
- Prompt storage: FP16/INT8 KV with permuted and nonresident pages matched CPU
  values in physical, host and staging pools, including untouched guards.
  Strided RoPE matched decode for none/linear/YaRN scaling. Both FP16 and BF16
  canonical expert matrices matched scalar expansion of every stored code.
- Control vectors: 12 graph-replay cases covered add/project modes, pending
  residual writes and request-time enable/disable. CPU comparisons and row
  padding checks passed; disabled vectors left unwritten residuals bit-identical.
- Host engine adapter: twelve changing-input graph replays alternating two
  queues, event-completed D2H handoff, cross-queue dependencies, event timing,
  graph node inspection, pitched-copy padding, host callbacks and freeing with
  pending transfers passed. Unsupported host registration and last-error
  behavior were checked.
- Shared host engine: the existing captured-graph wrapper passed sixteen
  changing-input replays. An ordinary 64 KiB expert arena was allocated and
  released without a runtime error. Cache slots of 4/8/16 KiB passed eight
  changing-payload rounds over two close/reopen cycles through bounded staging.
- Distinct 64-bit writes below and above 4 GiB: both values matched.
- FP16/BF16 conversion over 73,748 inputs, including every FP16 encoding,
  rounding boundaries, random FP32 bit patterns and NaNs: passed against host
  `_Float16` and oneAPI BF16 conversions (NaN payload identity is not required).
- Weighted RMS normalization with row tails/canaries, batched GDN gate and
  elementwise arithmetic: passed against double-precision CPU calculations.
- Generic and subgroup-32 native routing: batched stable-ID and weight checks
  passed, including ties and widely separated logits. The CPU reference restores
  the default floating-point environment so subnormals participate in sorting.
- Packed 2/4/8-bit embedding gathers, expert row exclusion and event-completed
  payload publication: passed. GPU polling for CPU-written flags is explicitly
  rejected; the scheduler must use events.
- Fused hyper-connections: the three-stage SYCL read at width 2,560, four
  residual streams and rank 320 passed double-precision CPU comparisons with
  and without a pending residual write and injection projection. Against the
  existing native composed path, the maximum absolute difference was
  `3.8147e-6`; comparisons used `3e-6 * (1 + abs(reference))`. The paths are not
  bitwise identical. Single and three-token calls were bitwise identical to
  each other, and the pending residual write matched the composed write.
  Normalized activations are reconstructed inside projections, so this path
  needs no additional activation workspace. It has one variant and does not
  implement CUDA's device-timestamp profiling or variant benchmark.
- Native short-context attention: 22 masked/unmasked cases at widths 1..256
  passed a double-precision CPU softmax/weighted-sum reference for Q24x256 and
  KV2x256. Unused KV/mask padding contained NaNs, F16 inputs had only two-byte
  alignment, and invalid step records returned the explicit failure status and
  NaN outputs. Output guards and alias rejection passed. The SYCL kernel
  translates the pinned CUDA adapter's subgroup/online-softmax arithmetic;
  this check does not establish cross-device bitwise equality.
- Sequence helpers: device position/window records, token maps, residual
  selection, broadcasts, conditional copies, unaligned byte-row gathers,
  bounded blob fetches, pointer rebasing, all-resident expert plans, selected
  token probabilities and 2/4/8-bit embedding gathers passed CPU/layout checks.
  Tests cover duplicate experts, nonresident fallback, invalid negative indices,
  uninitialized output tails and buffer guards. APIs without source capacities
  retain the caller's obligation to provide valid positive indices and extents.
  Mapped-flag waits and device global-timer stamps explicitly reject use;
  completion and timing use SYCL events outside captured polling paths.
- Sampling: the existing sampler parity suite passed, including the full
  248,320-token vocabulary, all 64 shortlist positions under ties, signed
  zeros/NaNs/infinities, penalties, filter ordering, counter segmentation and
  graph capture/replay. The SYCL baseline selects in one work-group per row;
  its ordered FP64 tail preserves the engine's counter-based draw contract.
  Coupled draft sampling passed 64 captured rounds of three chained draws with
  changing parameters, full/subset token maps, penalties, probabilities,
  history updates and buffer guards against a CPU reference. These kernels
  allocate no memory during capture or replay. Split-vocabulary selection is
  not yet implemented in this backend.
- Fused GDN: convolution/SiLU/L2, BF16 alpha/beta projection and recurrent
  update/output normalization passed double-precision CPU checks for 1/3/8
  tokens at 1/1 and 2/6 key/value heads. Batched outputs and committed state
  matched repeated single-token calls bit for bit. Verification leaves history
  and state untouched; partial commits and skipped output prefixes were checked.
  This preserves the fused CUDA path's four row partitions and explicit FMA
  placement, but is not a cross-device bitwise comparison or speed measurement.
- Generic and native GDN recurrence: six changing steps checked against a
  double-precision CPU oracle with a different state layout. This includes the
  real 128-wide, 16-key-head/48-value-head geometry, modulo head pairing and
  native readout scaling. Convolution history, canaries, L2 normalization and
  the sigmoid-gated output norm also passed.

- Canonical Q8_0, CPU-expert scaled Q8_0 and Q8_K activation quantization:
  2,048 blocks per variant matched CPU reference bytes, including zero blocks,
  halfway rounding cases and opposite-sign maximum ties. FP32 expert scales
  also matched bit for bit. Inverse conversion, output canaries and invalid
  shapes/overlaps passed. Quantization uses explicitly rounded division: the
  ordinary SYCL FP32 division differed by one ULP on the scale of test block 14.

- BF16-weight/FP32-activation MMVF: 73 shapes matched an ordered-FMA CPU
  reference bit for bit, including all adaptive work-group sizes, padded
  activation/output strides and 1, 2, 3, 4, 5 and 8 tokens. Each batched output
  also matched its separate single-token call and a double-precision dot
  product within the test tolerance. Fourteen legacy BF16-activation shapes,
  including odd reduction widths and split sizes 1 through 256, passed the
  double-precision reference and canary checks.

- Composed gated residual: twelve layer/head cases passed CPU references,
  including the real 2,560-wide, four-stream, rank-320 geometry in BF16,
  FP32 and native MMVF modes. Checks include per-stream normalization,
  activation rounding, all projections, absent head injection, saturated/zero
  gates, exact in-place writes, workspace sizing and output canaries.
- NEOX RoPE at positions 0 through 262,144, widths 128/256, unscaled,
  linear and YaRN scaling, text/image positions and in-place updates: table
  rotations matched bit for bit. Analytic rotations used 32 host-libm float32
  frequencies captured by value and strict host floating-point compilation;
  the maximum absolute difference from the float32 CPU reference was
  `2.385e-7`. Standard SYCL device `pow` had produced up to `0.02439` on the
  same inputs. This is a correctness measurement, not a speed comparison.
  Registered angle tables currently belong to runtime device 0.
- Paged FP16, INT8 and Hadamard-Q4 KV storage: 15 format/page/width
  combinations passed byte-exact CPU checks, with permuted page tables,
  nonresident pages, host-USM authoritative copies and output canaries.
  Coverage includes stored-FP16-scale INT8 rounding, Q4 maximum ties,
  changing device steps, empty gathers, Q4 prompt/step equivalence and
  prompt staging. The 256-point Hadamard transform matched the CPU
  butterfly sequence bit for bit and its inverse recovered the input
  within `4e-7` absolute error.
- Gathered GQA attention with the real 24-query/2-KV-head, 256-wide
  geometry and a 6-query/3-KV-head, 68-wide fixture: counts 0, 1, 7, 257
  and 2,051 passed double-precision softmax/value references. Normal-input
  maximum absolute error was `3.358e-7`; a fixture with query/key magnitudes
  up to 30/10 had `7.169e-5`. Scalar and device-step entry points matched
  bit for bit. Empty selections, weight/output canaries, FP16/FP32 gates,
  in-place native gates and weighted RMS at widths 66 through 2,560 passed.
- Generic indexer pooling: eight geometry/scaling/image-position cases,
  each with 19 changing steps, matched CPU pooled/spare keys bit for bit.
  Sixteen score cases passed double-precision per-head ReLU references.
  Twenty-four selection cases through 32,768 cells matched independently
  sorted IDs exactly, including ties, signed zero, NaNs and infinities.
  Signed zero is canonicalized by bits because arithmetic `+0` was optimized
  away by the compiler and initially broke the ascending-ID tie rule.
- Native indexer: FP16-rounded input pooling passed a CPU reference at
  `2e-6 * (1 + abs(reference))` tolerance for none, Linear and YaRN scaling.
  Twenty-three cells split into six uneven batches left the same tail, spare,
  pooled keys and block position as single-cell calls, bit for bit. Invalid
  device positions left pooled keys unchanged. The initial batch path submits
  one kernel per cell; prompt throughput has not been measured.
- Batched FP32 block scoring passed a CPU double-precision reference at
  `3e-6 * (1 + abs(reference))` tolerance. Twenty-four weighted block-selection
  cases through 262,147 cells matched independently sorted cell IDs exactly,
  including partial blocks, ties, signed zero, NaNs and infinities. The initial
  selector uses 32 binary threshold passes; its reference entry point uses the
  same kernel. Tensor-core scoring reports unavailable so callers use FP32.
- KV streaming passed byte comparisons for FP16, INT8 (codes and scales)
  and Q4 transfers. Three-slot tests cover duplicate selections, hits, CLOCK
  eviction, epoch wrap, bounded miss-list overflow, metadata canaries, ring
  restore across multiple wraps and full staging. Metadata resolution is a
  serial GPU task followed by parallel copies; throughput is unmeasured.
  Counter reads drain runtime device 0 queues; external queue users must
  complete their work before reading counters.
- Split attention directly over paged FP16, INT8, Q4 and hybrid K8/V4:
  twelve format/page cases passed the double-precision reference with maximum
  absolute error `3.541e-7`. Three queries with 13/67/129 selected cells and
  capacity 133 matched their single-query calls bit for bit. Checks include
  permuted pages, missing pages, all-masked/empty selections, chunk tails and
  scratch/output canaries. Quantized direct reads retain FP32 dequantization;
  they do not insert the FP16 rounding used by the separate gather path.
- PLE postprojection stages at 1/3/9/12 tokens matched a CPU reference at
  `8e-6 * (1 + abs(reference))` tolerance. Batched keys, gates, gated values,
  normalized history and results matched sequential calls bit for bit, including
  the documented in-place result and query/normalization aliases. Twelve full
  PLE fixtures cover canonical S2/Q8_0, native Q2_0/Q8_1 and BF16 key weights,
  both value activation precisions and both postoperation modes. These use
  synthetic weights; they are not a real PLE-layer comparison.
- Canonical matrix-vector products passed 24 code/offset/activation cases
  (S2/S4/S8/IQ4NL weights with FP16, Q8_0 or Q8_K inputs), against independent
  scalar decoding and double accumulation. The tolerance was
  `5e-7 * (1 + sum(abs(reference terms)))`; staged and direct S2 reads agreed.
  Sixteen shared-expert cases covered all partial native-projection overrides
  and both BF16 gate modes. Native batched shared experts matched separate
  calls bit for bit. Eight MoE reductions matched ordered CPU FMA exactly,
  with the shared output added once after the routed sum.
- Native Q8_1 and MMVQ for Q2_0, Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K,
  Q6_K, IQ4_NL and IQ4_XS: 150 synthetic cases passed, including small/large
  reduction widths, row tails, 1/3/8 columns and both multi-column layouts.
  Q8_1 bytes matched the independent host quantizer; the exact layout matched
  separate single-column calls bit for bit. The largest
  `abs(error)/(1 + sum(abs(reference terms)))` was `5.099e-8` against scalar
  dequantization and double-precision dot products. The reference includes the
  original-input-sum correction required by affine Q4_0/Q5_0.
- Native grouped experts: ten format pairs at width 256 and Q2_0 at the
  real 2,560/640 geometry matched separate quantization/projection/activation
  calls bit for bit. Tests include repeated tokens, reordered destinations,
  device and host-USM blobs, groups crossing the eight-entry tile boundary,
  one-entry tiles, empty counts and scratch/output guards. Transfer-time
  dequantization, embedding gathers and interleaved gate/up conversion passed
  layout/rounding checks for these ten formats plus BF16. The dequantizers
  reuse the repository scalar definitions; this is not an independent check
  of those definitions. Other IQ1/IQ2/IQ3 formats remain unsupported.
- Real Q2_0 GGUF: first/middle/last rows from all 448 tensors of those ten
  formats passed with three activation columns. The same normalized maximum
  error was `4.699e-8`. This samples weights, not a forward pass.

The real-weight check is optional and reads the model in place:

```sh
build-sycl/sycl_mmvq_test --gguf /path/to/model-00001-of-00002.gguf
```

The recorded model is `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, Q2_0,
revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`. Both GGUF shard sizes
and SHA-256 hashes were checked before use. The model and native pack live
outside the repository, under `$XDG_DATA_HOME/strata-sycl` (or
`~/.local/share/strata-sycl` when unset). The pack contains 1,079 index entries;
its tokenizer passed all 16 strings in `strata_tokenizer.py --check`.

These checks establish runtime and kernel behavior, not model correctness or inference
performance. Remaining work includes event-completed speculative verification
and end-to-end launch integration, followed by layer/logit comparisons and
real-model inference validation.

### Canonical expert routing

`moe_hit_select` and `moe_hit_select_multi` compact resident entries in routing
order, for up to 32 and 128 entries respectively. `moe_group_resident` groups
entries by first occurrence of the expert and preserves entry order inside each
group. Negative expert ids are skipped. Positive blob extents remain the caller's
responsibility. These small routing operations use one device work item.
`moe_hit_add` adds each selected row at its original routing destination; its
input destinations must be distinct. Nonpositive or over-capacity device counts
leave the output untouched. Positive destination extents remain caller-owned.

`sycl_expert_routing_test` checks selection, group pointers and starts, token
indices, untouched rows, invalid counts, and graph replay with changing resident
hits against CPU expectations. Canonical S2 projections now implement per-hit,
device-count, multi-token and grouped routing. Each subgroup computes a row
and reuses its weights across up to eight tokens. The existing parity test
checks single-entry versus tiled paths bitwise, including intermediate storage,
and checks projections against a double-precision CPU reference. It covers
FP16/FP32 activation scales, partial counts, grouped host-USM blobs and
misaligned inputs. The separate CPU-order diagnostic also passed the
double-precision projection reference. These checks passed on the B570; they do not establish
end-to-end expert throughput.

### Native per-layer decode

Native packs can use `--spec 0` (the default) with the captured per-layer
scheduler. Each CPU handoff uses the native row-split expert pool with one token;
canonical S-form planes are not required for projections served from native
GGUF bytes. Native GPU cache hits use the layer's GGUF format and each slot's
actual pointer, including variable-size slots. The CPU pool consumes the same
hit decisions and leaves those routed rows at zero before GPU results are added.
Remote expert caches and `--expert-cache-cpu-order` remain unsupported on this
per-layer native path. Speculative window boundaries are described below.
`sycl_native_single_dispatch_test` compares two changing activations and layers
against direct native CPU row calls, including routing order, unweighted outputs,
invalid ids and rejection of an unsupported GPU residency table. It also checks
GPU hit handoff with mixed hit/miss rows, changing hit positions, miss-only
layers, variable slot offsets, and prevention of duplicate result addition.

A B570 smoke run on 2026-10-02 used the Q2_0 model/pack identified above, FP16 KV,
context 64, four CPU workers, mmap experts, no GPU cache or PCIe miss split, and
PLE enabled. The command starts directly with `strata --pack`, without a
`generate` subcommand:

```sh
ONEAPI_DEVICE_SELECTOR=level_zero:gpu build-sycl/strata \
  --pack "$PACK" --native "$SHARD1" --tokens 1,2 --max-new 3 \
  --max-context 64 --mmap-experts --pool-workers 4 --expert-cache 0 \
  --pcie-frac 0 --check-logits --dump-logits /tmp/sycl-logits.bin \
  --dump-layers /tmp/sycl-layers.bin --dump-routing /tmp/sycl-routing.bin
```

Two fresh processes produced ids `220 100561 198`. Each dump contained four
positions: all 248,320 logits per position and all 49 residual records (input
plus 48 layers) were finite; the 192 routing records each held ten distinct ids
in 0..511 and finite weights. Logits, residuals and routing were byte-identical
between those two runs. The logits SHA-256 was
`19dcf9b7ddeabb4c3a34c91e9574c6bdadbb21a46aaa568965c6a5ea058faa55`.
A separate one-token run with fused GR enabled produced id 198 from input 1,
with all logits finite. Layer dumping disables fused GR, so the repeated run
covers the composed GR path. These checks establish execution and repeatability,
not agreement with another backend or model quality; no speed claim is made.

### Short chat and CPU reference

On the same B570/5600X machine, a 25-token rendered chat asked
`What is 2+2? Reply with just the number.` with an empty completed thinking
block. Native per-layer decode, Q2_0, FP16 KV, context 128, four CPU workers,
mmap experts, cache disabled and eight generated tokens produced token 19
(`4`) followed by the turn delimiter. EOS stopping was disabled for this probe.
All 32 positions contained 248,320 finite logits. Two fresh processes produced
byte-identical dumps (SHA-256
`f2f279b63b4280878c2909cb65a82d71a0c65bee4e55246e47107e8ad546072e`).
The first run measured 1.06 generation tok/s and the subsequent warm-cache run
7.14 tok/s; eight generated tokens are not a steady-state benchmark.

A separate CPU-only llama.cpp build at the pinned commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d` read the same GGUF shards and
teacher-forced the 25 input tokens one at a time, with four threads, FP16 KV
and Flash Attention disabled. All 25 argmax ids matched, including the answer.
The logits were not bit-identical: minimum/mean cosine were
0.989024/0.997810, maximum per-row RMSE 0.271464, and maximum absolute difference
1.789672. This is one short prompt, not a broad model-quality result.

The optional reference driver lives in `tools/sycl/llama_logits.cpp` and uses
llama.cpp's public API. Build the pinned dependency in a separate directory:

```sh
cmake -S "$LLAMA_DIR" -B build-sycl-oracle \
  -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_MTMD=OFF -DBUILD_SHARED_LIBS=ON \
  -DGGML_CUDA=OFF -DGGML_SYCL=OFF -DGGML_NATIVE=ON -DGGML_CPU_ALL_VARIANTS=OFF
cmake --build build-sycl-oracle --target llama -j2
c++ -std=c++17 -O2 tools/sycl/llama_logits.cpp \
  -I "$LLAMA_DIR/include" -I "$LLAMA_DIR/ggml/include" \
  -L build-sycl-oracle/bin -Wl,-rpath,"$PWD/build-sycl-oracle/bin" \
  -lllama -o build-sycl-oracle/bin/llama_logits
build-sycl-oracle/bin/llama_logits "$SHARD1" /tmp/tokens.txt /tmp/cpu-logits.bin
python3 tools/sycl/compare_logits.py /tmp/cpu-logits.bin /tmp/sycl-logits.bin --rows 25
```

The comparison requires NumPy and validates complete finite dumps before
reporting differences. `--rows` explicitly selects a common prefix when the
SYCL dump also includes generated positions. `--require-exact` additionally
fails on any bit difference, for repeated runs using the same backend. The
ordinary report does not impose a model-quality threshold.

A separate four-position run used the same model, context and CPU workers with
`--tokens 1,2,220,100561 --max-new 1 --expert-cache 96 --expert-cache-per-layer`
and the same dump options. The 96 slots were admitted, with 142 later resident
hits and 1,682 CPU misses across 1,920 routed entries. All logits were finite;
top-1 ids matched the CPU-only run at all four positions. Maximum absolute logit
differences per position were 1.104, 0.525, 0.450 and 0.712; CPU-to-cache softmax
KL divergences were 0.01426, 0.00472, 0.00373 and 0.00240. Routing id sets matched
in 154 of 192 layer records (ordered lists in 99). The GPU expert uses Q8_1
activation blocks, whereas the CPU pool follows its native CPU activation
contract. This run validates the cache execution path, not numerical equivalence
or a model-quality gate; the observed differences still require oracle analysis.

The 25-position chat above also ran with the 96-slot cache and fused GR.
The final answer was still token 19 (`4`), but only 24/25 intermediate argmax
ids matched the CPU oracle. At position 18, the cached run chose 74455 instead
of 248045; minimum cosine was 0.10695 and maximum absolute logit difference
15.31690. Disabling fused GR kept this argmax discrepancy. Its cause remains
unresolved, so GPU cache model-level numerical validation is not complete.
[The cache comparison records](../bench/results/2026-10-02-sycl-native-cache/run.json)
include prompt ids, configuration, dump hashes and comparison reports. The
handoff test passed separately for synthetic Q8_0 and Q2_0 experts, including
nonzero references; maximum CPU differences were 0 and 8.81701e-7 respectively.

### Prefill queue ordering on B570

With oneAPI 2026.1 on the Arc B570, a two-queue host/device ring stalled
when GEMMs used expert-sized matrices. A standalone reproduction uses 200
products, 32 host staging threads, 128 host slots, eight device slots, and
FP16 matrices with `T=3, N=1280, K=2560`. The main thread waited for a host
slot whose transfer event did not complete. The same test completed with a
single in-order queue; a smaller two-queue case (`N=32, K=640`) also completed.
Changing event markers to empty kernels did not resolve the large case.
This isolates the observed failure to this execution pattern, without
establishing a driver or runtime root cause.

SYCL prefill therefore uses its caller's compute queue for transfers too.
CPU staging remains parallel, but transfer/compute overlap is not enabled in
this path. The queue is borrowed and is not destroyed by prefill. The GEMM
test covers the small two-queue case and the large single-queue case, including
all 200 output matrices and host-buffer reuse.

The 25-token arithmetic chat above completed with `--prefill 32 --spec 0
--expert-cache 0 --mmap-experts --pool-workers 4 --max-new 8 --max-context 128
--check-logits --stats`. The 24 conditioning tokens took 3171.4 ms (7.57 tok/s),
streaming 4,248 experts; eight output tokens took 1250.2 ms (6.40 tok/s).
Their ids matched the sequential run, starting with 19 (`4`). These are one
short-run measurements on the B570, with filesystem cache state uncontrolled;
they are not a steady-state throughput result or a broad quality evaluation.


Logits dumps now finalize their row count from rows actually written. Batched
conditioning positions are omitted; `--stop-eos` can shorten the file further.
An unfinished file keeps a zero row count and is rejected by the comparison
tool. On this chat, full batched generation wrote eight rows, EOS stopping wrote
two, and `--prefill-until 10 --max-new 1 --logits-stride 7` wrote three rows
(positions 14, 21 and 24). Exact file sizes and finite values were checked.

To compare the eight generated positions with the sequential dump:

```sh
python tools/sycl/compare_logits.py sequential.bin batched.bin --reference-start 24
```

All eight argmax ids matched. The minimum cosine was 0.964518, maximum per-row
RMSE 0.500223, and maximum absolute logit difference 2.771040. The first answer
position compared with the CPU reference had cosine 0.994826, RMSE 0.196610,
and maximum absolute difference 1.154035; both predicted `4`. EOS stopping's
two rows were bitwise equal to the corresponding prefix of the eight-row run.
These are limited checks, not numerical equivalence or a broad quality gate.
[The prefill records](../bench/results/2026-10-02-sycl-prefill/run.json) include
prompt ids, options, dump sizes and hashes, and the complete comparison reports.

### ESIMD Q5_K output projection

The 2,560-wide Q5_K vocabulary projection uses a 16-lane ESIMD kernel with
packed integer dots. Eight partial accumulators preserve the ordinary MMVQ
mapping; FP16 scale decoding includes subnormals. The accumulator is DP4A's
first source operand, as documented in the
[Intel instruction specification](https://intel.github.io/pisa/instructions_arithmetic.html).
Other widths and formats retain their existing kernels.

On B570, three alternating before/after processes, each with 20 GPU-only
replays, measured the head at `4.011..4.023 ms` before and `1.520..1.523 ms`
after. Layer time remained about `35.1 ms`. The combined GPU interval fell from
about `39.2 ms` to `36.6 ms`; it excludes CPU experts and token preparation.
The 37-token writing prompt generated 128 identical token IDs at `13.57 tok/s`
before and `13.79 tok/s` after, in one pair on the shared machine. The target
generation performance remains open. Evidence is in
`bench/results/2026-10-02-sycl-head/run.json`.

All 170 argmax IDs of the context comparison matched. Maximum logit difference
was `3.8147e-6`, and maximum row RMSE was `4.60358e-7`; outputs are not bitwise
equal to the preceding SPMD head. The independent MMVQ test passed with 4,097
output rows, three/eight columns, finite FP16 scale edge cases and buffer guards.
Exact multiple-column calls launch this same kernel once per column, retaining
single/multiple-column bit parity. Their speculative-decoding speed has not
been established.

### Shared expert and CPU overlap

The ordinary per-layer SYCL scheduler now completes a route graph before it
reads the CPU expert inputs. It queues the shared expert in a separate graph
after that completion event, so shared GPU work can continue while the CPU
experts run. Both graphs and the later result combination use the same ordered
queue. The full pre graph remains available for GPU-only replays and for the
`--no-overlap` comparison. With shared-early disabled, the original post path
computes the shared expert.

On B570 with the Ryzen 5 5600X, the 37-token writing prompt generated 128 tokens
in three alternating before/after pairs. Experts used ordinary RAM, 2,048
profile-filled GPU slots and four CPU workers; prefill used 16-token chunks.
Decode measured `14.32..14.65 tok/s` before and `15.58..15.62 tok/s` after.
The median increased from `14.48` to `15.60 tok/s` (7.7%). All 128 output IDs
matched across the six processes. A separate 32-row comparison of all 248,320
logits was bitwise equal.

Shared GPU work was still pending at `6110..6131` of `6144` CPU boundaries in
each after run. This counter establishes unfinished GPU work when the CPU can
start; it does not measure overlap duration. The existing mid-graph doorbell
counter remains zero because the route graph has completed.
`sycl_host_engine_test` passed twelve changing-input replays of three synthetic
layers, alternating the separated graphs and full-pre fallback. It checked the
current routed inputs, exact residuals and one shared execution per layer.

These measurements ran on a shared machine, and 128 tokens truncate the
requested 300-word story. MTP retains its existing event boundaries. The
approximately `29 tok/s` reference target remains open. Full options and
results are in `bench/results/2026-10-02-sycl-shared-overlap/run.json`.

The optional `sycl_projection_bench` target measures supported dense GGUF
projections in isolation. `sycl_projection_bench SHARD1 50 1` repeats each
matrix fifty times in one graph and reports the median of three device-event
intervals. It excludes quantization and transfers, and repeatedly reuses warm
weights. On B570, the 300 matrices totalled `20.688 ms` of these isolated
intervals; the ninety Q3_K matrices accounted for `10.351 ms`. These sums are
not generation times. The matrix directory and individual timings are in
`bench/results/2026-10-02-sycl-projections/before.csv`.

### Q3_K dense ESIMD path

Q3_K single-column and exact multiple-column projections now use signed DP4A
instructions in a 16-lane ESIMD kernel. The kernel keeps eight virtual block
accumulators and the original four-warp reduction order. Two-byte gathers
handle Q3_K's 110-byte block stride. Explicit `fma(a, b, 0)` calls round each
float product before its next accumulation: source `-ffp-contract=off` alone
did not preserve those boundaries on this installed ESIMD backend. The grouped
expert and ordinary multiple-column SPMD paths retain their existing arithmetic.
Exact multiple-column calls launch the new kernel once per column.

On B570, the ninety real dense Q3_K matrices took `3.982 ms` in total isolated
device intervals, compared with `10.351 ms` before (2.60 times faster). The
300-matrix sum fell from `20.688` to `14.323 ms`. These are warm-weight graph
measurements using fifty repeats and three device-event samples per matrix;
they exclude quantization and transfers and are not generation times.
`sycl_projection_bench SHARD1 50 1 --compare-q3` also checks the new output
against the original three-column SPMD arithmetic. All ninety matrices matched
bitwise for the diagnostic input.

Four alternating before/after pairs generated 128 tokens from the same
37-token writing prompt, on the B570 and Ryzen 5 5600X with ordinary RAM,
2,048 profile-filled GPU slots, four CPU workers, 16-token prefill chunks and
MTP disabled. Before measured `15.81 / 15.46 / 3.45 / 15.10 tok/s`; after
measured `17.38 / 17.27 / 17.17 / 17.20 tok/s`. The medians were `15.28` and
`17.235 tok/s` (12.8%). Before-3 spent `28.839 seconds` blocked in PLE SSD
reads. It remains in the median; the fourth pair was added after this stall.
The shared machine's background load was not isolated.

All 128 output IDs matched across all eight processes. A 32-row generated
logit comparison and a 165-row fixed-input comparison were bitwise equal over
all 248,320 vocabulary entries. `sycl_mmvq_test` passed the ten-format synthetic
checks, grouped experts, changing columns, FP16 scale edge cases and buffer
guards. Its real-GGUF check passed the first, middle and last rows of 448
tensors with three input columns; Q3_K checks also require exact ESIMD/SPMD
bits with the original four-warp path.

Results and individual projection intervals are recorded in
`bench/results/2026-10-03-sycl-q3-esimd/`. The approximately `29 tok/s` reference
target remains open. This change has no measured MTP speed result.

### Q2_0 resident expert ESIMD path

Native Q2_0 projections now unpack signed codes `-1, 0, 1, 2` into DP4A
operands in a 16-lane ESIMD kernel. Masked gathers handle partial virtual block
groups and the 18-byte block stride. The products and original 128 virtual
lanes retain the SPMD accumulation order. The grouped expert path reads group
counts, entry ranges, token indices and destinations on the GPU. It preserves
zero outputs for invalid token rows and leaves inactive groups untouched.
`STRATA_OLD_IQ_MMVQ=1` selects the original Q2_0 SPMD kernels before capture.

On B570, real inputs from position 18 of a saved arithmetic trace replayed ten
experts in each of the 48 layers. Full gate/up, activation, hidden quantization
and down graphs took a median `0.44445 ms` per layer with SPMD and `0.15920 ms`
with ESIMD (2.79 times faster). All grouped output bits matched. These timings
use fifty graph repeats and the median of three device-event samples, with
resident warm weights and input Q8_1 blocks; CPU work, input quantization and
weight copies are excluded. They are not generation times.
The 57 isolated dense Q2_0 projections fell from `1.755` to `0.553 ms` in total
warm-weight intervals measured by `sycl_projection_bench`.

For generation, three alternating before/after pairs used the B570 and Ryzen
5 5600X, ordinary RAM, four CPU workers, a 37-token writing prompt, 16-token
prefill chunks and MTP disabled. The 2,048-slot per-layer cache started empty
and admitted experts until full. Before measured `16.82 / 16.80 / 17.05 tok/s`;
after measured `19.00 / 18.96 / 18.92 tok/s`. The median increased from `16.82`
to `18.96 tok/s` (12.7%). All six processes generated the same 128 IDs and had
`30429 / 61440` GPU-cache hits (49.53%). The background load was not isolated.

Separate 32-row comparisons with an empty cache and with the static profile
were bitwise equal to their respective before runs over all 248,320 logits.
A 165-row fixed-input comparison with the profile was also bitwise equal.
`sycl_mmvq_test` passed changing columns, signed-zero/subnormal/negative FP16
scales, 640/2560/6144-wide row tails, original/new grouped paths, invalid token
rows, empty groups and buffer guards. Three-column checks of the first, middle
and last rows of 448 real GGUF tensors also passed. The optional
`sycl_expert_contract PACK SHARD1 TRACE FIRST COUNT --bench` records the two
GPU timings and requires exact SPMD/ESIMD grouped output bits in addition to
its independent Q8_1 and CPU references.

The options, trace hash and per-layer/matrix results are in
`bench/results/2026-10-03-sycl-q2-esimd/`. The writing run is truncated at 128
tokens. These results do not establish MTP throughput; its multiple-entry
weight reuse remains a separate measurement. The approximately `29 tok/s`
reference target is still open.

### Event-completed speculative windows

SYCL verification captures the window in segments. The first segment embeds
the tokens and runs the first layer's mixer, router and shared expert. Each
following segment consumes the preceding CPU expert results and GPU plan,
combines them, then prepares the next layer. A completion event outside capture
must finish before the host reads routed activations or writes results. Expert
DMA completes before the next segment starts. No device kernel waits for a
host flag. The last segment finishes the final layer and output head.

On the 48-layer model, an unsplit window has 49 segments; `--spec-split` uses
97 for two token groups. This first implementation serializes each CPU boundary
with its GPU segment. CUDA's device-plan shortcut and global-timer profiler are
disabled on SYCL. PCIe mode selects explicit DMA, since ordinary host expert
storage has no mapped device alias. GPU-resident expert arithmetic remains the
native Q8_1 path. Verification requires an initially profile-filled expert cache,
the default fused native kernels, and `--spec T` with T from 2 through 8. A
multi-token native prompt also needs `--prefill CHUNK`. A one-token native
prompt now starts verification at position zero; zero previously collided with
the CLI's sentinel for an unused speculative loop.

On 2026-10-02, B570/5600X, Q2_0, FP16 KV, context 256, four CPU workers,
2,048 profile-filled cache slots, resident RAM experts and prefill chunks of
16, a 37-token story prompt generated the same 32 output ids in all four runs:

| Run | Drafts accepted | Generated token/s |
| --- | ---: | ---: |
| Ordinary decode | n/a | 14.41 |
| T=4, known continuation supplied as drafts | 24/24 | 16.72 |
| T=4, every draft deliberately corrupted | 0/93 | 5.47 |
| T=8, two groups, every third draft corrupted | 16/112 | 4.90 |

These single trials ran on a shared machine. The verifier variants retain the
default adaptive policy, which updates residency between windows; ordinary
decode keeps the initial profile placement. The known-continuation run measures
an artificial drafting case; it includes no MTP prediction cost. The rejection
runs check GDN commit, indexer-tail restoration and PLE history across changing
windows. They establish this fixture's output agreement, not bitwise logit
agreement across all prompts. A separate one-token prompt check and a withheld
host-result test cover position zero and error cleanup. The latter exits with
an error after draining submitted GPU work, before any graph consumes withheld
results. The runtime test also checks twelve changing producer/CPU/consumer
rounds on alternating queues with external completion events.

Flags, ids and results are in
[`bench/results/2026-10-02-sycl-verify/run.json`](../bench/results/2026-10-02-sycl-verify/run.json).

### MTP drafting on SYCL

MTP catch-up selects an accepted residual and token into the source buffers'
first row and first id. SYCL's original span validation rejected those exact
aliases. `mtp_select` now permits them while rejecting partially overlapping
residual rows. Each lane copies its own residual element, and the sole token
writer reads the selected token before overwriting the first id. Twelve
captured replays with changing negative, zero and later row selections passed,
including untouched-row and output guards.

The MTP assets were fetched as 31 tensor ranges from the pinned checkpoint
revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`; all 31 passed the fetcher's
SHA-256 checks. `tools/mtp_pack.py` and `tools/mtp_rt.py` produced the canonical
Q2_0 expert blobs and Q8_0 draft projections. The CJK-inclusive draft subset
contains 106,299 ids. On B570, the runtime reported about 968 MiB of MTP VRAM,
including the 178.4 MiB draft head.

The same 37-token story prompt was measured with 128 generated tokens,
context 512, FP16 KV, four CPU workers, 2,048 profile-filled cache slots,
resident RAM experts, prefill chunks of 16 and suffix drafting disabled:

| Run | Drafts accepted | Tokens/round | Generated token/s |
| --- | ---: | ---: | ---: |
| Ordinary decode | n/a | n/a | 14.44 |
| MTP T=4, adaptive swaps | 64/192 | 1.98 | 10.28 |
| MTP T=4, swaps disabled | 70/177 | 2.17 | 10.69 |
| MTP T=4, swaps disabled, probability threshold 0.5 | 66/109 | 2.05 | 13.12 |

Both fixed-placement MTP runs produced exactly the ordinary run's 128 ids.
Threshold 0.5 exercised T=1,2,3,4 windows in 13,14,13,23 rounds respectively.
The adaptive run moved 1,536 experts and first differed at output index 60;
output agreement here is limited to fixed placement. These are single trials
on a shared machine. Generation wall time includes first-use graph preparation;
the phase timer for MTP drafting starts after round-graph capture. MTP remains
slower than ordinary decode on this fixture, and the NVIDIA comparison target
has not been reached.

A persistent `--serve` engine accepted two identical requests, generated the
same eight ids twice, reused 30 of 37 prompt tokens on the second request and
exited successfully after `QUIT`. In a separate process, `STOP` after the first
emitted token cancelled a request for 128 tokens; the next request produced the
same eight reference ids and the engine exited successfully. This tests the engine stdin protocol; Python
HTTP endpoints have not been tested by this probe. Flags, outputs, source and
runtime artifact hashes are in
[`bench/results/2026-10-02-sycl-mtp/run.json`](../bench/results/2026-10-02-sycl-mtp/run.json).

### Native Q2 CPU-order diagnostic

`--expert-cache-cpu-order` now accepts native Q2_0/Q2_0 experts on SYCL with
widths 2,560/640, the AVX2 CPU path and `--spec 0`. It remains opt-in. Its
quantizer uses FP32 scales and the compiled AVX2 quantizer's reciprocal and
half-away rounding. The projections retain eight independent AVX2 accumulators
and their reduction order. SwiGLU uses Intel's high-accuracy exponential.

On B570, replaying all 48 layers at position 18 of the 25-token arithmetic
prompt produced bitwise equal up projections and down projections when given
the same quantized inputs. Hidden quantization codes also matched. Floating
point hidden values and scales can still differ. Maximum final expert error
against the CPU pool was `1.19209e-6` with the high-accuracy exponential;
the ordinary Q8_1 expert path's maximum was `0.00620055` in this replay.
The CSV files and full-model comparisons are recorded in
`bench/results/2026-10-02-sycl-native-q2/`.

With 96 per-layer cache slots, the arithmetic prompt matched all 25 CPU
reference argmax IDs. Against SYCL with CPU experts, minimum logit cosine was
`0.961894`. Using the ordinary SYCL exponential in this CPU-order experiment
instead gave minimum cosine `0.130277` and a mismatch at position 18. The two
memory sources, mmap and resident RAM, produced identical logits for that
experiment, so the difference was not explained by mmap staging alone.

With automatic cache sizing and resident RAM, the 167-token context prompt
matched 165/167 CPU model argmax IDs and generated `blue`. Against SYCL with CPU
experts it matched 169/170 positions, with minimum cosine `0.985900`. This
CPU-order path generated four tokens at `6.67 tok/s`, below the ordinary native
cache result recorded below. It is a numerical diagnostic, not the default
performance path; these two prompts do not establish broad quality parity.

The CPU-order activation blocks now retain their signed code sum in the unused
two-byte header. Projections read it instead of summing the same 32 codes for
every output row. With cache capacity fixed at 2,893 slots, all 170 logits rows
of the context run remained bitwise equal to the preceding CPU-order build.
This run generated four tokens at `8.62 tok/s`; background CPU load differed,
so it does not establish an isolated kernel speedup. These private activation
blocks have Q8_0's 34-byte stride but use a code-sum header, not an FP16 scale.

`sycl_native_single_dispatch` also checks the FP32 activation scales and codes
against the actual AVX2 quantizer, including zeros and rounding ties. For manual
expert replay, build `sycl_expert_contract` and provide a trace recorded with
`--dump-expert-inputs`:

```sh
cmake --build build-sycl --target sycl_expert_contract -j2
build-sycl/sycl_expert_contract PACK SHARD1 TRACE FIRST_RECORD COUNT
```

The tool checks an independent Q8_1 dot interpretation, reports CPU-order
differences and separates projection errors from hidden quantization differences.

### Longer prompt and parallel GR streams

A 167-token synthetic chat mixed English notes, Japanese text and a Python
function, then asked for a color mentioned earlier. CPU-reference and SYCL
cache-off argmax ids matched at 164/167 positions; mean cosine was 0.998259,
minimum cosine 0.963235. Both answered `blue`. Splitting its 166 conditioning
positions into chunks of 64 produced the same four generated ids as sequential
conditioning, with minimum cosine 0.995759 on those four positions.

On this B570/Ryzen 5 5600X/125 GiB machine, keeping the 31.64 GiB expert arena
in ordinary RAM avoided rebuilding expert blobs from the GGUF mappings on each
layer. The 170-row mmap and arena dumps were bitwise identical. In these short
runs, the CPU-pool call averaged 97.87 ms/position with mmap versus 26.70 ms
with the arena; generation measured 6.80 versus 13.70 tok/s. Auto-sized,
per-layer GPU caching admitted 2,874 slots and measured 14.72 tok/s, with
162/167 prompt argmax ids matching the CPU reference (minimum cosine 0.957857).
These four-output-token timings are not an isolated or steady-state benchmark:
other host workloads remained active, and filesystem caches were not cleared.

The fused GR up projection now assigns its four residual streams to four
subgroups of one workgroup. It keeps the five original partial reductions and
their addition order, then combines the four gates after one local barrier.
The previous kernel visited streams sequentially with three barriers per
stream. The fused GR unit test passed, and all 170 real-model logit rows were
bitwise identical before and after this change.

Three alternating before/after GPU-only runs (`--gpu-only-full`, context 512,
20 replays per process) measured layers at 36.967/36.954/36.957 ms before and
35.205/35.152/35.146 ms after. Head time remained about 4 ms. This measurement
omits expert computation and CPU handoffs and must not be reported as decode
throughput. [The context records](../bench/results/2026-10-02-sycl-context/run.json)
contain the prompt, dump hashes, comparisons, settings and individual timings.
