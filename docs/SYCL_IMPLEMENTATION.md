# SYCL backend implementation

The backend is under development. The runtime and device arena run on Intel Arc
B570; the SYCL build runs short native-pack decode with CPU experts.
Layer/logit agreement with an independent model oracle and inference performance
have not yet been validated.
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
event succeeds. Whole-token polling graphs and the polling-based speculative
verifier explicitly reject SYCL use. The ordinary expert arena skips host
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
per-layer native path. The polling verifier still needs further integration.
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
