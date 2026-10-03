# SYCL backend implementation

The SYCL backend runs native Q2_0 and mixed IQ3_S inference on this Linux
Intel Arc B570 machine, with GPU expert caching, CPU experts in resident RAM,
prefill, MTP and persistent
requests through the Python server. The current measured configuration and its
limits are below. The later dated sections retain the earlier measurements.
Short-prompt logits have been compared with a CPU model reference; broad quality
validation and measurements on other machines remain open.
The implementation follows [the port research](SYCL_RESEARCH.md) and
[the operation inventory](SYCL_BATTLEMAGE_OPERATIONS.md).

## Single-GPU feature ports (2026-10-04)

Against upstream `99f3dbd` (engine 0.1.38), SYCL now supplies all 17 native
MMVQ formats, routed quantized MMQ, fused Q2_0/native MoE prefill, matrix prompt
attention and the optional image encoder below. Multi-GPU and Intel setup
automation remain outside this work. The shared APIs, web app and model
handling come from the imported upstream code.

Native MMQ is built into the SYCL backend and selected for supported expert
layouts. `STRATA_PREFILL_MMQ=0` selects the previous FP16 expert path.
Matrix prompt attention is selected when the device and geometry support it;
`STRATA_PROMPT_ATTN_OLD=1` selects split attention. Fused native expert prefill
requires `STRATA_PF_FUSED=1` and the shared streaming threshold. The measured
native fused fixture used `STRATA_PREFILL_STREAM_MIN=16` and
`STRATA_PREFILL_RING=128`. Canonical Q2_0 fused prefill follows upstream's
default selection. These variables are read at process startup.

MMQ and fused expert products use integer SIMD; prompt attention uses Intel
`joint_matrix` with FP32 scaling and softmax. The ports provide the operations
without reproducing CUDA-specific instructions or scheduling. Their
[validation record](../bench/results/2026-10-04-sycl-functional-parity/run.json)
and [status report](SYCL_STATUS_2026-10-03.md#single-gpu-feature-ports-2026-10-04)
state the tested fixtures and numerical limits.

## Intel GPU image encoder (2026-10-04)

`strata-vision` can now be built with the pinned llama.cpp SYCL backend. Use
the same llama.cpp revision as the engine's GGML, `3cf03257f219afbe7334045ff7c6a06ac68c627d`,
and an activated oneAPI environment:

```sh
cmake -S tools/vision -B build-vision-sycl -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DLLAMA_DIR=/path/to/pinned/llama.cpp -DSTRATA_VISION_SYCL=ON
cmake --build build-vision-sycl --target strata-vision -j2
```

CUDA and SYCL encoder options are mutually exclusive. For a CPU encoder,
build the same target with both GPU options off and the ordinary C/C++
compilers. These manual builds do not require changes to the setup script.

Add a `vision` section to the existing server config, using the encoder
executable, the model's `mmproj-Qwen3.8-Flash-Next-BF16.gguf`, and its first
text-model GGUF shard:

```json
"vision": {
  "exe": "/path/to/build-vision-sycl/bin/strata-vision",
  "mmproj": "/path/to/mmproj-Qwen3.8-Flash-Next-BF16.gguf",
  "model": "/path/to/text-model-00001-of-00002.gguf",
  "gpu": true,
  "threads": 4,
  "max_tokens": 256
}
```

The server enables the engine's image path from this section. Retain the
config's `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` environment and loopback
host. The encoder starts before the model so automatic expert-cache sizing
can account for its VRAM. Here `max_tokens` is the encoder's image-token
limit; the text engine's context must also hold those tokens and the reply.

On the B570/5600X/125 GiB machine, a solid 224-by-224 PNG produced 49 image
tokens of width 2560. CPU and SYCL encoders each produced finite, bitwise
repeatable embeddings within their own backend. Between backends, cosine
was `0.99780219`, RMSE `0.00200989` and maximum absolute difference `0.04712183`.
This is one encoder fixture, not an image-quality benchmark.

With the SYCL encoder and IQ3_S text model on the same B570, the real
loopback server passed OpenAI and Anthropic image requests, repeated image
reuse, OpenAI streaming, text after an image, two images in order, and a
single image after two. Answers were `red`, `4` and `Red, Blue` for the
corresponding fixtures. The tested config used context 1024, 16-token prefill,
automatic cache sizing (1603 slots), adaptation off and MTP width four at
minimum probability 0.9. The server shut down cleanly. Large images,
grounding and broad image quality have not been measured here.

The local tested preset is
`~/.local/share/strata-sycl/serve-config-vision-sycl.json`; the text-only
default remains the IQ3_S preset described below.

## Current IQ3_S server configuration

The local default now uses the verified mixed IQ3_S model. On 2026-10-03,
three fresh-process pairs on this B570/5600X measured median first/repeated story decode at
12.94/14.37 token/s and first/repeated Python merge-prompt decode at
16.43/18.30 token/s. The layer-sized cache holds 2,137 experts in approximately
4.09 GiB; the preceding uniform layout held 1,649 in the same budget. Settings
are context 512, FP16 KV, four workers, 16-token prefill, MTP width four,
minimum draft probability 0.9 and 64 adaptive replacements every four rounds.
Startup and prompt processing are excluded; background workloads were active.
IQ3_S remains slower than the earlier Q2_0 result and has not reached
approximately 29 token/s on this fixture.

The [detailed status report](SYCL_STATUS_2026-10-03.md) explains acquisition,
all parameter samples, numerical checks and real HTTP validation. Upstream
0.1.38 at `99f3dbd` has since been merged into the fork's main through PR #1,
and the SYCL branch rebased and checked again; the report distinguishes
the subsequent single-GPU ports from imported shared changes. Those prompt
changes have not been measured with new decode medians. The canonical API model name is
`qwen3.8-flash-next-iq3_s-sycl`, with the tested existing alias
`qwen3.8-flash-next-sycl`. The preceding Q2_0 config is saved at
`~/.local/share/strata-sycl/serve-config-q2_0.json`. Models, logs and local configs
remain outside Git; reviewed results are in
[`bench/results/2026-10-03-sycl-iq3s/run.json`](../bench/results/2026-10-03-sycl-iq3s/run.json).

## Measured Q2_0 server configuration (2026-10-03)

On 2026-10-03, Intel Arc B570 with 10 GiB VRAM, Ryzen 5 5600X and 125 GiB RAM,
Qwen3.8-Flash-Next GSQ-RCO Q2_0 ran with DPC++ 2026.1.1, Level Zero driver
`1.17.39395+14` and default driver settings. The measured Q2_0 server config uses
context 512, FP16 KV, four CPU workers, 16-token prefill, the shipped profile,
automatic expert-cache sizing (3,792 slots here), 64 adaptive replacements every
four verification rounds, MTP width four, minimum draft probability 0.9 and
PCIe share zero. PLE reads use the existing direct-I/O path. Short verification
windows use the default native SYCL graph recording; adaptive refills complete
before the next window uses their residency map.

Three fresh persistent processes per probability setting each served a
37-token story prompt twice, then a 45-token Python merge-function prompt twice.
Each request generated 128 tokens. Median decode rates were:

| Request in each process | Probability 0.5, token/s | Probability 0.9, token/s |
| --- | ---: | ---: |
| First story | 19.26 | 21.27 |
| Repeated story | 22.05 | 25.46 |
| First Python merge function | 23.40 | 26.02 |
| Repeated Python merge function | 26.05 | 29.70 |

The repeated Python request reached the approximately 29 token/s reference
target on this fixture. First requests and story requests remain slower. The
timers exclude startup and prompt processing; first requests include lazy graph
preparation. Repeated story and Python requests reused 30/37 and 38/45 prompt
tokens respectively, and retained expert-cache contents from preceding requests.
Background workloads were not isolated. This is a measurement on a 10 GiB B570,
not a comparison with NVIDIA under matched memory capacity and conditions.

`--spec-min-p` stops further MTP proposals when draft confidence is low; the
target model still verifies the proposals. At 0.9, the first story accepted
38/48 drafts rather than 63/116 at 0.5. The repeated Python request accepted
88/90 rather than 88/115. Less rejected draft work improved these measurements.
This change affects only the local server config, not tensor arithmetic,
quantization or the engine's global CLI default.

All three fresh processes with the same setting retained identical output ids,
cache counts, prompt reuse and cancellation behavior across six requests.
Different probability settings with adaptive placement can produce different
continuations: the placement schedule changes which experts run on CPU or GPU.
No cross-setting output equality or broad quality result is claimed. A separate
fixed-placement check with 2,048 slots and adaptive changes disabled retained
the same 64 output ids and all float32 bits in their 64 full-vocabulary heads
between probabilities 0.5 and 0.9. It compares committed rows with the same input
history; rejected draft rows differ between proposal windows and are excluded.

An initial four-setting screen retained all samples for 0.5, 0.7, 0.9 and 0.99.
The first repeated pair comes from that screen; the following two pairs ran 0.5
then 0.9 in alternating fresh processes. Each process also cancelled a 128-token
story after its first emitted token, generated the known eight-token continuation
on the next request and exited successfully. After updating the local config to
0.9, real HTTP checks passed for the web page, `/v1/models`, OpenAI chat and
streaming, Anthropic messages, repeated output and cancellation recovery.
The test server stopped cleanly. This HTTP check measures functionality, not
the throughput in the table.

Flags, prompts, per-request timings, output hashes, numeric comparisons and
HTTP check results are summarized in
[`bench/results/2026-10-03-sycl-mtp-floor/run.json`](../bench/results/2026-10-03-sycl-mtp-floor/run.json).
Raw execution logs, generated trial files and workstation configs stay outside
Git; see the [measurement storage policy](../bench/results/README.md).
The measured Q2_0 config is now saved as
`~/.local/share/strata-sycl/serve-config-q2_0.json`; its preceding 0.5 configuration
is backed up as `serve-config-before-floor-tuning.json` in
the same directory. [The server instructions](#mtp-drafting-on-sycl) give the
start command and API URL. Context 512 is the tested local setting; longer
contexts have not been benchmarked with this configuration.

## Mixed IQ3_S model support

The IQ3_S GGUF also contains IQ3_XXS and IQ2_S gate/up experts. SYCL now
supports all three formats in native single- and multi-column MMVQ, grouped
GPU experts, FP32/FP16 transfers and interleaved prefill transfers. Codebooks
and decoders follow the pinned ggml definitions; the model is not requantized.
Existing Q2_0, IQ4_NL and IQ4_XS paths remain in use for its other expert layers.

On this B570, `sycl_mmvq` passed with an independent ggml decoder oracle,
all 512/256/1,024 codebook indices, signs, subscales and finite FP16 scale
edge cases. Mixed grouped experts cover IQ3 gate/up with IQ4_NL or Q2_0 down.
The IQ3_S GGUF row check passed for the first, middle and last rows of all
447 supported tensors with three activation columns. Maximum
`|error| / (1 + sum|terms|)` was `6.59863e-8` against a scalar dequantized dot
product. This is a kernel check, not a whole-model quality measurement.

```sh
build-sycl/sycl_mmvq_test --gguf /path/to/IQ3_S-00001-of-00002.gguf
```

## Per-layer native cache storage on SYCL

SYCL native packs with `--expert-cache-per-layer` now allocate each slot for
the format of its owning layer. The existing admission ranges and adaptive
replacement policy remain in use. Explicit slot counts retain their count;
automatic sizing fits more slots into the preceding uniform plan's byte budget.
If allocation retries shrink the count, layer ranges and sizes are regenerated
together. Profile-ranked sizes remain a separate path for global admission.

On the B570 IQ3_S model, a fixed 1,649-slot cache used 3.15 GiB instead of
4.09 GiB. A before/after generation check kept the same 64 output ids and all
float32 bits in 69 full-vocabulary verification rows across 36 windows, including
rejected draft rows. Adaptive replacement was disabled for that comparison;
cache placement and the input history stayed fixed. The host-engine test also
checks mixed slot sizes, nonuniform last-layer ranges, fills and smaller retries.

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
the real-model prompt checks are recorded below. A matrix attention kernel
remains a possible optimization.
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
  of those definitions. The later mixed-IQ3_S section adds IQ2_S, IQ3_XXS
  and IQ3_S, with an independent ggml transfer oracle. IQ2_XXS, IQ2_XS
  and IQ1 formats remain unsupported on the native GPU path.
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

The one-row native Q2_0 ESIMD path unpacks signed codes `-1, 0, 1, 2` into DP4A
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

### Packed Q2_0 experts on XMX

For grouped Q2_0 experts with embedding width 2,560 and FF width 640, each
ESIMD work-item now computes sixteen weight rows. Integer DPAS uses those rows
as sixteen columns, with one activation row and depth 32. The existing 18-byte
Q2_0 blocks supply unsigned two-bit weights; activations remain signed Q8_1.
Starting the integer accumulator at minus the activation code sum implements
the weights' `-1` offset exactly. FP16 scale decoding, both float product
boundaries and the original 128 virtual-lane additions retain their preceding
order. Other dimensions and dense Q2_0 projections use the preceding DP4A
kernel. `STRATA_SYCL_Q2_XMX=0` selects that kernel for comparison; the SPMD
diagnostic switch still applies.

On 2026-10-03, B570/5600X, twelve recorded sets of ten real experts used
warm resident weights and Q8_1 inputs. The sum of their device-event intervals
was `1.869820 / 1.869753 / 1.869940 ms` before and
`0.960162 / 0.960415 / 0.960467 ms` after. Each interval is the median of
three samples of fifty captured repetitions after warm-up. The medians are
`1.869820` and `0.960415 ms`, a `1.95x` improvement in this isolated workload.
All grouped output bits matched SPMD, alongside independent Q8_1 numeric
references. These intervals exclude CPU experts, input quantization and copies.

The integer packing probe checked 2,560 varying DPAS results at M=1/4,
N=16, K=32, including inputs -128, 127 and zero and both Q2_0 block halves.
All matched its integer reference. Grouped tests cover host-USM/device weight
pointers, multiple entries, invalid tokens, empty plans and buffer guards;
a separate canonical fixture compares all output bits with zero, signed-zero,
subnormal, negative and maximum finite FP16 scales. A 64-token MTP diagnostic
retained all 30 windows/90 rows of float32 logits bit for bit, including
rejected draft rows.

Three alternating 128-token MTP runs used the preceding overlap experiment's
fixed 2,048-slot profile and flags. Before measured
`15.43 / 16.94 / 16.78 tok/s`; after measured
`16.84 / 16.86 / 16.85 tok/s`. The medians differ by only `0.4%`;
these shared-machine samples do not establish a generation speed improvement.
All six retained the same ids, 63 rounds, 66/109 accepted drafts and
8,705/82,560 cache hits. The isolated kernel gain therefore remains separate
from decode performance.

Three alternating ordinary 128-token runs used 4,135 empty per-layer cache
slots, four CPU workers, prefill 16, context 512 and the RAM PLE table.
Before measured `21.25 / 19.84 / 22.12 tok/s`; after measured
`21.44 / 22.86 / 22.87 tok/s`. The medians were `21.25` and `22.86 tok/s`
(`7.6%` higher). All six retained the same 128 ids and 43,845/61,440 cache
hits. The variation between runs on this shared machine limits the speed
conclusion; all samples are retained. A separate 32-token cold-cache run
with 3,038 slots and direct PLE reads retained every float32 logit bit.

Flags, comparison results and raw intervals are in
[`bench/results/2026-10-03-sycl-q2-xmx/run.json`](../bench/results/2026-10-03-sycl-q2-xmx/run.json).

### Native graph recording probe

The installed DPC++ supports the public
[`enable_native_recording` graph property](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc).
It records through the backend and does not expose SYCL graph nodes. A
temporary adapter selected this property only with
`STRATA_SYCL_NATIVE_RECORDING=1`; the prototype is archived and removed from
the default engine.

On 2026-10-03, B570/5600X, three alternating ordinary 128-token runs used
4,135 empty per-layer slots, four workers, context 512, prefill 16 and RAM
PLE. Normal recording measured `23.08 / 22.82 / 22.98 tok/s`; native
recording measured `22.21 / 18.78 / 21.52 tok/s`. Medians were `22.98` and
`21.52 tok/s`. CPU pool intervals also varied: `5.631..6.173` versus
`6.490..8.727 ms/token`. Background workloads were not isolated, so this
does not establish the cause of the difference. It provides no generation
speed reason to change the default. All six retained identical 128 ids and
43,845/61,440 cache hits.

The synthetic allocation probe recorded 192 graphs with 15 small kernels
each on one shared queue. Normal recording consumed `238 MiB`; native
recording consumed `38 MiB`. These figures measure that fixture, not the
full engine's graph footprint. Changing host-USM inputs, graph-internal
event completion, alternate replay queues and destruction of the recording
queue before replay passed in both modes. Every logit bit matched in a
32-token ordinary run and 30 MTP windows/90 rows, including rejected drafts.

The prototype, independent probes, flags, comparisons and all samples are in
[`bench/results/2026-10-03-sycl-native-recording/run.json`](../bench/results/2026-10-03-sycl-native-recording/run.json).

### Short verifier windows use native recording

SYCL verification now records one-, two- and three-token windows with Intel's
`enable_native_recording` graph property. Four-token and larger windows, the
commit graph and generic CUDA-compatible capture retain normal recording.
The adapter exposes an explicit native-capture entry point; it rejects node
queries for those graphs. Normal graphs retain node queries. Replay completion
still covers the entire graph, including host-USM copies and captured events.
`STRATA_SYCL_VERIFY_NATIVE_CAPTURE=0` selects normal recording for all verifier
graphs; `1` selects native recording for all; `small` selects the default.

On 2026-10-03, B570/5600X, three alternating pairs compared normal recording
with native recording of every verifier graph. Each fresh server generated
128 tokens twice from the story prompt, then twice from the code prompt. It
used context 512, prefill 16, MTP T=4, probability floor 0.5, 3,792 automatic
expert slots, the shipped profile, four workers and 64 adaptive replacements
every four rounds. Three later processes measured the short-window selection
with the same flags and GPU kernels. These later runs were not alternating
with the controls. Median decode rates, excluding prefill and startup, were:

| Request in each fresh process | Normal, tok/s | All native, tok/s | Short windows native, tok/s |
| --- | ---: | ---: | ---: |
| First story | 10.51 | 18.42 | 19.13 |
| Repeated story | 19.57 | 20.29 | 21.95 |
| First code after the story requests | 16.23 | 20.06 | 23.65 |
| Repeated code | 25.98 | 24.18 | 25.74 |

All nine processes retained the same ids, cache-hit counts and reused prompt
lengths for each request. Cancellation after one emitted token and the next
eight-token request also matched. Background CPU workloads were not isolated;
the table does not establish the cause of each speed difference. Recording
every graph natively did not improve repeated-code generation, so that mode
is a diagnostic option.

After the four window shapes were recorded, the device free-memory query
reported about `3 / 241 / 147 MiB` for normal, all-native and short-window
recording. DRM fdinfo after the first story reported `10,136 / 9,907 / 10,001 MiB`
of allocated and resident VRAM. These are backing-storage measurements;
they do not prove GPU paging. Summed shape-capture times were less than one
second in every run, so capture time alone does not explain the first-request
speed differences.

Native recording retained every logit bit in 30 MTP windows/90 rows,
17 split-eight windows/129 rows with corrupted oracle drafts, and seven
position-zero windows/17 rows. The default short-window mode with adaptive
refills retained all bits in 32 windows/98 rows. Rejected draft rows are
included. Withheld host results failed after GPU work had drained. Adapter
tests check both recording modes, changing host inputs, cross-queue completion
and replay after destruction of the recording queue. The default also passed
the real-model OpenAI/Anthropic HTTP checks, complete streaming, repeated
requests and cancellation recovery. The HTTP server stopped cleanly.
Flags, memory measurements, comparison summaries and scripts are in
[`bench/results/2026-10-03-sycl-verifier-native-recording/run.json`](../bench/results/2026-10-03-sycl-verifier-native-recording/run.json).

### Captured ordinary post and next route

Native ordinary sessions with fixed cache sizes now capture the completed
host-USM miss copy and a conditional device hit addition inside each post
graph. A host-USM flag snapshots `hit_pending` before the Combine callback
clears it; that callback retains its completion counters. Miss-only layers
skip the addition, preserving their float bits and ignoring stale hit rows.
With separate early shared work, the post also records the next layer's
route. Its completion event covers the current post and the next routed
payload before the CPU reads it. The next shared expert stays outside that
graph and can overlap the CPU pool. Kernel order, GDN state slices and QSA
staging addresses retain their preceding behavior.

`STRATA_SYCL_POST_HANDOFF=0` selects the preceding host submissions;
`STRATA_SYCL_POST_ROUTE=0` keeps next routing separate while retaining the
captured copy/add. Early auto-cache capture keeps its preceding path. Layer
dumps and late shared work keep routing separate; GPU-only diagnostic graphs
and MTP use their existing paths.

On 2026-10-03, B570/5600X, three alternating ordinary 128-token runs used
4,135 empty per-layer slots, four workers, context 512, prefill 16 and the RAM
PLE table. Before measured `22.85 / 22.90 / 22.90 tok/s`; after measured
`23.03 / 23.01 / 23.11 tok/s`. Medians were `22.90` and `23.03 tok/s`
(`0.6%` higher). Median host-after-ring intervals fell `12.516 -> 11.489
ms/token`; these intervals include CPU work and submission. The small
generation difference on this shared machine establishes no large speed gain.
All six retained identical 128 ids and 43,845/61,440 cache hits.

All logits remained bitwise equal in a 32-token cold-cache run and 165 fixed
input positions. Auto sizing selected 4,135 slots in both eight-token probes,
with every logit bit equal. MTP retained its 32 ids and 19/28 accepted drafts
over 13 rounds. The synthetic three-layer chain covers separate, captured
and combined-next-route posts, with each shared expert executed once. A
24-replay handoff fixture changes host inputs and flags, compares with the
original memcpy/add and a scalar float reference, and checks negative zero,
subnormal values, miss-only NaN payloads and an output canary.

A preceding scalar kernel read all misses directly from host-USM rather than
using DMA. Its three-pair median was `22.86 -> 22.74 tok/s`; the direct-read
kernel was replaced. Its raw measurements and prototype are retained in
[`bench/results/2026-10-03-sycl-post-direct-read/run.json`](../bench/results/2026-10-03-sycl-post-direct-read/run.json).
The final copy/routing flags and comparison summaries are in
[`bench/results/2026-10-03-sycl-post-route/run.json`](../bench/results/2026-10-03-sycl-post-route/run.json).

### Captured native cache-hit work

The ordinary native Q8_1 hit path now captures its output clear, input
quantization and grouped expert kernels at session setup. One graph is shared
by layers with the same expert types and dimensions. The fixed device buffers
keep their addresses; at this change's introduction the routed hit pointers,
count and destinations were uploaded before each graph launch. The native path also omits the slot
list upload that only the canonical S2 path consumes. Capturing and allocating
these graphs happens before token processing. CPU-order diagnostic hits retain
their existing kernels. `STRATA_SYCL_NATIVE_HIT_GRAPH=0` or `--no-capture`
selects the uncaptured hit path.

Three alternating before/after pairs on B570 and Ryzen 5 5600X used the same
empty 2,048-slot per-layer cache, ordinary RAM, four workers, 16-token prefill
chunks, 37-token writing prompt and 128-token output budget. Before measured
`19.01 / 18.93 / 18.92 tok/s`; after measured `19.60 / 19.51 / 19.02 tok/s`.
The median increased from `18.93` to `19.51 tok/s` (3.1%). Every process had
`30429 / 61440` cache hits, and all 128 output IDs matched. A separate 32-row
comparison was bitwise equal across all 248,320 logits.

`sycl_native_single_dispatch` passed captured/uncaptured bit comparisons with
Q8_0 and Q2_0 fixtures, changing activations, reordered hits, padded slot sizes,
a miss-only layer and a duplicate result-combine call. Duplicate graph setup
is rejected, and CPU-order mode bypasses the graphs. Resources are owned by
the dispatch session and released before its device buffers at normal teardown.

The machine's background load was not isolated; these short runs are not a
completed 300-word story. Results are in
`bench/results/2026-10-03-sycl-native-hit-graphs/run.json`. The MTP verifier uses
its separate graphs, and the approximately `29 tok/s` generation target
remains open.

### Native hit-plan staging inside capture

The captured hit path now owns one bounded host-USM plan for expert pointers,
group metadata and destinations. The output-clear kernel copies this plan
into the existing device buffers, replacing three small transfer submissions
per layer. Expert kernels keep reading metadata from device memory. Ten routed
experts need a 208-byte plan, allocated before token processing. The current
layer's route completion follows the preceding layer's hit graph, so the host
can update the plan before launching the current hit graph. Graphs are released
before the plan is freed. Uncaptured and CPU-order paths keep their transfers.

Three alternating before/after pairs on B570 and Ryzen 5 5600X used ordinary
RAM, an initially empty 2,048-slot per-layer cache, four workers, 16-token
prefill chunks and the same 37-token writing prompt. Before measured
`19.43 / 19.42 / 19.65 tok/s`; after measured
`19.67 / 19.93 / 19.56 tok/s`. The medians were `19.43` and `19.67 tok/s`
(1.2%). The third pair was slower after the change. Background load was not
isolated, and the small generation-speed difference needs that context.
Host time after the route completion fell from `18.714..19.274` to
`17.757..18.045 ms/token` in these runs; this phase includes CPU work and
result submission, so it does not isolate metadata-transfer overhead.

All six processes generated the same 128 output IDs and had `30429 / 61440`
cache hits. A separate 32-row comparison was bitwise equal across all 248,320
logits. `sycl_native_single_dispatch` passed the captured/uncaptured comparisons
with changing inputs and hit order, miss-only layers, padded slots and Q8_0
and Q2_0 fixtures. Full options and results are in
`bench/results/2026-10-03-sycl-native-hit-plan/run.json`. These truncated writing
runs do not establish MTP throughput or the approximately `29 tok/s` target.

### GR projections with a normalized workspace

The fused SYCL GR read can use its existing normalized-activation workspace.
The norm stage writes each normalized residual once; ESIMD down and up
projections then read it. The read still uses three kernels and allocates no
additional session workspace. Single-token layers and multi-token verifier
and MTP callers use this path. A call without the workspace retains the SPMD
implementation used for comparison.

The projections preserve the original FP32 product boundaries, FMA order and
subgroup partial sums. The down projection uses eight programs per row and
64 bytes of local storage per workgroup. The up projection reduces five
partials through a zero-padded vector tree. Scalar SYCL exponential functions
retain the activation and gate arithmetic. An optional raw up-dot output
supports comparison with the independent MMVF kernel.

On 2026-10-03, B570/5600X, the real Q2_0 pack's 96 layer reads and final output
read had bitwise equal outputs. With warm weights, fixed synthetic FP32
residuals, 50 captured repeats and the median of three device-event intervals
per variant, the sum fell from `3.515` to `2.210 ms`. These timings include
normalization, projections, nonlinear functions and mixing; they omit weight
uploads and CPU experts.

The same 37-token story prompt, resident RAM, four CPU workers, 2,048 initially
empty per-layer cache slots, prefill 16, FP16 KV, context 512 and speculative
decoding disabled gave these alternating 128-token runs:

| Trial | Before token/s | After token/s |
| --- | ---: | ---: |
| 1 | 20.09 | 19.98 |
| 2 | 20.13 | 20.20 |
| 3 | 20.13 | 19.99 |

Medians were `20.13` and `19.99 token/s`: these runs establish no generation
speed improvement. All six produced the same 128 ids and cache-hit counts.
All logits were bitwise equal in a 32-output run and a separate 165-position
fixed-input run. A short MTP T=4 comparison with a static profile and adaptive
swaps disabled produced the same 32 ids and accepted 19 of 28 proposals in
both builds; this single pair does not establish MTP throughput. Background
workloads were not isolated. The unit test also checks exact single/batch
outputs, raw MMVF dots, pending residual updates, final mixing and aliases.

Flags, ids and timings are in
[`bench/results/2026-10-03-sycl-gr-esimd/`](../bench/results/2026-10-03-sycl-gr-esimd/).

### Single-token IQ4_XS projections

Single-column IQ4_XS dense projections now use ESIMD DP4A. The nonlinear
nibble codebook is selected from four packed constants before forming signed
bytes. Two explicit rounded products and the original 128-lane reduction
preserve the SPMD arithmetic. `STRATA_OLD_IQ_MMVQ=1` selects the SPMD path.

On 2026-10-03, B570/5600X, all outputs of the real pack's 56 IQ4_XS matrices
were bitwise equal to SPMD for fixed synthetic Q8_1 inputs. With warm weights,
50 captured repeats and the median of three device-event intervals, their
single-column sum fell from `3.524` to `2.565 ms`. A three-column prototype
increased the sum from `5.542` to `7.697 ms`; multiple columns therefore retain
the original SPMD weight reuse. These sums exclude uploads and CPU work.

With the same 37-token story, resident RAM, four CPU workers, 2,048 initially
empty per-layer cache slots, prefill 16, context 512, FP16 KV and speculative
decoding disabled, three alternating 128-token runs measured
`20.14 / 19.66 / 19.48 token/s` before and
`20.51 / 19.83 / 20.59 token/s` after. Medians were `19.66` and `20.51 token/s`
(`+4.3%`). All six produced the same 128 ids and cache-hit counts. The final
lookup also preserved every logit in the 165-position fixed-input comparison.
Background workloads were not isolated, and the approximately 29 token/s
comparison target remains open.

The MMVQ test covers the original SPMD oracle, finite FP16 scale edge cases,
partially filled virtual blocks, row tails and 1/3/8 activation columns. The
first/middle/last-row real-weight check passed all 448 tensors. Full options,
timings and comparisons are in
[`bench/results/2026-10-03-sycl-iq4xs-esimd/`](../bench/results/2026-10-03-sycl-iq4xs-esimd/).
The projection tool's `--compare-iq4xs` checks complete matrix outputs;
`--spmd-iq4xs` selects the original kernels for timings.

### Automatic cache sizing after ordinary graph preparation

SYCL ordinary decode now finalizes its layer graphs before sizing an automatic
expert cache. Those graphs read session buffers and expert results without
cache-slot addresses. Before recording queues were shared, preparing those
graphs on B570 reduced free VRAM by about `1.64 GiB`; the earlier calculation
had assigned that room to the cache.
Standalone CLI generation also resets its prompt workspace after prefill.
A 16-token chunk reclaimed `136 MiB` in the measured fixed-capacity run.
The persistent server retains its reusable prompt workspace. Verifier/MTP
graph preparation order is unchanged.

On 2026-10-03, B570/5600X, the same 37-token story, resident RAM, four workers,
FP16 KV, context 512, prefill 16 and speculative decoding disabled gave:

| Trial | Before auto token/s | After auto token/s |
| --- | ---: | ---: |
| 1 | 15.57 | 21.27 |
| 2 | 16.01 | 21.13 |
| 3 | 16.04 | 21.25 |

Auto sizing selected `4,321` slots before and `3,038` after. Medians were
`16.01` and `21.25 token/s` (`+32.7%`). Each variant repeated its own 128 ids;
the two variants differed from output index 52 because cache placement changes
the mix of CPU and GPU arithmetic. This comparison includes that different
continuation. With placement fixed at 3,038 slots, all 32 logit rows remained
bitwise equal after graph preparation and prompt workspace changes. A short
MTP probe retained its 32 ids and 19/28 accepted drafts.

A separate configuration screen, before these changes, measured 3,072 slots
with direct PLE I/O at `20.78 token/s` and PLE RAM I/O at `21.31 token/s`, with
the same 128 ids. RAM I/O loaded the 28.8 GB table in `46.1 s`; locking failed,
so its pages were touched and remained unlocked. PLE host time fell from
`1.647` to `0.017 ms/token`. This single pair does not establish a sustained
speedup or guarantee that those pages remain resident. Background workloads
were not isolated. Settings, output ids, memory traces and individual timings
are in
[`bench/results/2026-10-03-sycl-cache-memory/run.json`](../bench/results/2026-10-03-sycl-cache-memory/run.json).

### Shared recording queue for ordinary layer graphs

Ordinary SYCL sessions record all layer graphs on one queue, ending each
recording before starting the next. They destroy that queue after finalizing
the graphs. Previously, every graph used a separate recording queue. Finalized
graphs retained the queues' device resources. Sharing them leaves the same
graph segments and replay boundaries in place. Capture errors end unfinished
recordings and discard their graphs before releasing the queue.

On 2026-10-03, B570/5600X, native Q2_0, context 512 and 3,038 fixed cache slots,
192 ordinary graphs used `238 MiB` instead of `1,684 MiB`. Free VRAM after graph
preparation increased from `869` to `2,315 MiB`. All 32 vocabulary logit rows
were bitwise equal with placement fixed. Automatic sizing selected 4,135 slots;
its 32 logit rows also matched the old implementation fixed to 4,135 slots.
The runtime regression records two graph topologies on one queue, destroys
that queue, then replays both with changing host inputs on alternating queues.

The same cold 37-token story, resident RAM, four workers, prefill 16,
context 512 and speculation disabled gave these automatic-cache runs:

| Trial | Separate queues, 3,038 slots, token/s | Shared queue, 4,135 slots, token/s |
| --- | ---: | ---: |
| 1 | 20.97 | 20.54 |
| 2 | 20.15 | 20.49 |
| 3 | 21.15 | 20.36 |

The medians were `20.97` and `20.49 token/s` (`-2.3%`); this change establishes
VRAM savings, with no generation throughput improvement. Each variant repeated
its own 128 output ids, but the variants differed from index 36. Cache placement
changes CPU/GPU arithmetic and the generated continuation. A separate single
pair fixed both implementations to 3,038 slots and measured `21.21` and
`21.07 token/s`; this pair does not establish a speed difference. Background
workloads were not isolated. Full settings, output ids, traces and the standalone
192-graph memory probe are in
[`bench/results/2026-10-03-sycl-capture-queue/`](../bench/results/2026-10-03-sycl-capture-queue/).

### Native FP16 scale conversion in ESIMD kernels

Q2_0, Q3_K, IQ4_XS and the Q5_K vocabulary head share a native FP16-to-FP32
conversion helper. It converts finite scales, including subnormals and signed
zero, through the hardware instruction. It expands infinities and NaN payloads
from their original bits to preserve the previous decoder's behavior. The
accumulation, reduction and explicitly rounded product boundaries stay the same.
An exhaustive test checks all 65,536 FP16 encodings.

On 2026-10-03, B570/5600X, the warm single-column dense Q3_K projection sum was
`3.982` before and `3.391 ms` after. Twelve sets of ten real Q2_0 experts measured
`1.907` and `1.870 ms` with ESIMD; their SPMD control was `5.322 ms` in both runs.
These are 50-repeat captured graphs, with the median of three device-event
intervals, excluding CPU expert work and generation. All expert output bits
matched the SPMD control.

The cold 37-token story with 4,135 fixed slots, resident RAM, four workers,
prefill 16, context 512 and speculation disabled gave:

| Trial | Manual conversion, token/s | Native conversion, token/s |
| --- | ---: | ---: |
| 1 | 20.48 | 20.56 |
| 2 | 20.37 | 20.79 |
| 3 | 20.35 | 19.88 |

The medians were `20.37` and `20.56 token/s` (`+0.9%`). In the third after run,
PLE host time increased to `2.456 ms/token`; the other two after runs measured
`0.627` and `0.656 ms/token`. All six runs produced the same 128 ids and cache
hit counts. All 32 cold logit rows and 165 fixed-input rows were bitwise equal.
The first/middle/last-row real-weight test passed all 448 tensors with three
activation columns. A functional MTP probe retained the same 32 ids, 13 rounds
and 19/28 accepted drafts; it does not establish MTP throughput. Background
workloads were not isolated. Settings and raw measurements are in
[`bench/results/2026-10-03-sycl-native-half/`](../bench/results/2026-10-03-sycl-native-half/).

### Single-column Q4_K ESIMD projections

Q4_K single-column projections use packed DP4A in a 16-lane ESIMD kernel.
Eight accumulators preserve the original 128 virtual lanes, ascending
cross-warp sums and XOR reduction. The affine scale products round before
subtraction and accumulation. This source uses the same strict floating-point
compile settings as the other ESIMD projection kernels. Batched columns retain
the SPMD implementation that reuses decoded weights. The old-kernel setting
selects the SPMD single-column oracle too.

On 2026-10-03, B570/5600X, all 38 complete dense Q4_K matrices were bitwise equal
to the original SPMD outputs. Their warm single-column sum was `2.221 ms` before
and `0.889 ms` after (`2.50x`), using 50 captured repeats and the median of three
device-event intervals. This excludes CPU expert work and generation.

With the cold 37-token story, 4,135 fixed cache slots, resident RAM, four workers,
PLE RAM I/O, prefill 16, context 512 and speculation disabled:

| Trial | SPMD Q4_K, token/s | ESIMD Q4_K, token/s |
| --- | ---: | ---: |
| 1 | 21.29 | 21.61 |
| 2 | 21.01 | 21.84 |
| 3 | 20.21 | 20.89 |

The medians were `21.01` and `21.61 token/s` (`+2.9%`). All six runs produced the
same 128 ids and cache hit counts. All 32 cold logit rows and 165 fixed-input
rows were bitwise equal. The test covers both affine FP16 scales, subnormal and
negative scales, partial block iterations, row tails and 1/3/8 activation
columns. The real-weight check passed all 448 tensors. A functional MTP probe
retained its 32 ids and 19/28 accepted drafts.

Background workloads were not isolated. PLE RAM pages were touched at startup;
locking failed, so later residency is not guaranteed. Loading time is excluded
from decode timing. Full settings, measurements and complete-row comparisons
are in
[`bench/results/2026-10-03-sycl-q4k-esimd/`](../bench/results/2026-10-03-sycl-q4k-esimd/).
The projection tool provides `--spmd-q4k` and `--compare-q4k` for these checks.

### Exact batched Q4_K projections

With `multi_exact` enabled, batched Q4_K dense projections now use one ESIMD
row/column grid. Each column retains the original 128 virtual lanes and
floating-point operations. Single-column execution and the alternative
`multi_exact=false` SPMD arithmetic are unchanged. The old-kernel flag also
selects the original exact batched SPMD path.

On 2026-10-03, B570/5600X, the 38 real Q4_K projection matrices, warm weights,
50 captured repeats and the median of three device-event intervals gave:

| Input columns | Original exact SPMD sum, ms | Batched ESIMD sum, ms |
| ---: | ---: | ---: |
| 2 | 3.108 | 1.705 |
| 3 | 4.040 | 2.377 |
| 4 | 4.949 | 3.112 |
| 8 | 8.650 | 6.043 |

Every output row and column was bitwise equal for fixed distinct Q8_1 inputs.
The unit test also covers finite affine FP16 scales, partial block iterations,
row tails, and exact single/batch parity. The independent real-weight check
passed all 448 tensors. These sums exclude transfers and CPU work.

Three alternating MTP T=4 runs used the 37-token story, 128 generated tokens,
2,048 fixed per-layer slots filled from the shipped profile, four CPU workers,
direct PLE I/O, prefill 16, context 512, probability threshold 0.5, and adaptive
swaps disabled. Before rates were `15.30 / 15.63 / 15.31 token/s`; after rates
were `15.17 / 15.70 / 15.65 token/s`. Medians were `15.31` and `15.65 token/s`
(`+2.2%`). All six produced the same 128 ids, 63 rounds and 66/109 accepted
drafts. This is output agreement, without a full speculative-logit comparison.
Background workloads were not isolated, startup loading is excluded, and the
approximately 29 token/s comparison target remains open.

The projection tool's `--compare-q4k` accepts multiple columns and checks all
of them against the original SPMD outputs. Full flags, timings and comparisons
are in
[`bench/results/2026-10-03-sycl-q4k-batched-esimd/`](../bench/results/2026-10-03-sycl-q4k-batched-esimd/).

### Q5_K head weight reuse across small windows

The native 2,560-wide Q5_K vocabulary head now reads and expands each weight
block once for 2, 3 or 4 activation columns. Each column retains the original
single-column ESIMD dot products, accumulator lanes and reduction. One-column
and 5..8-column execution keep their existing implementation.

On 2026-10-03, B570/5600X, the full 248,320-row head occupies 437 MB of weights.
Warm weights, 50 captured repeats and the median of three device-event
intervals gave:

| Input columns | Independent column calls, ms | Shared weight expansion, ms |
| ---: | ---: | ---: |
| 2 | 2.562 | 1.554 |
| 3 | 3.863 | 1.937 |
| 4 | 5.153 | 2.315 |

Every output row and column was bitwise equal to the original single-column
ESIMD calls. The unit test covers affine FP16 scale edge cases, exact batch
parity and the 4,097-row tail. All logits in a 32-output ordinary run were
bitwise equal with placement fixed at 3,038 slots. These component timings
exclude transfers and CPU work.

The same fixed-profile MTP configuration as the batched Q4_K experiment gave
`15.57 / 15.71 / 15.70 token/s` before and `15.30 / 15.47 / 15.76 token/s`
after. Medians were `15.70` and `15.47 token/s`; these trials establish no
generation improvement. All six retained the same 128 ids, 63 rounds and
66/109 accepted drafts. Background workloads were not isolated. All trials,
flags and comparisons are retained in
[`bench/results/2026-10-03-sycl-q5-head-batch/`](../bench/results/2026-10-03-sycl-q5-head-batch/).
`--single-q5head` and `--compare-q5head` select the head-only component checks.

### IQ4_XS integer matrix and table-decoder screen

On 2026-10-03, B570/5600X, standalone prototypes computed sixteen weight rows
with signed-eight-bit DPAS, or eight useful rows by duplicating DPAS's upper
columns. Each integer dot covered the same 32 values; subscales, both float
product boundaries and virtual-lane additions followed the current kernel.
Another prototype selected nonlinear coefficients from a sixteen-byte register
table. These prototypes retained every output bit against the existing ESIMD
kernel on 2,560/6,144-wide matrices, row counts 32..10,240 and finite FP16 scale
edge cases. The weights and Q8_1 inputs were synthetic, not model traces.

With 100 kernels captured in one graph, warm weights and the median of three
device-event intervals, the final matrix prototype measured:

| Input width, output rows | Existing ESIMD, ms | DPAS 16 useful rows, ms | DPAS 8 useful rows, ms |
| --- | ---: | ---: | ---: |
| 2,560, 6,144 | 0.0686 | 0.0772 | 0.0950 |
| 2,560, 10,240 | 0.0734 | 0.0888 | 0.2076 |
| 6,144, 2,560 | 0.0501 | 0.2029 | 0.1564 |

The public `grf_size_automatic` kernel property measured `0.0772 / 0.0887 /
0.2029 ms` on those shapes. Its selected register count was not inspected.
The register-table decoder also lost to its own original-kernel control in
all three timed shapes. None of these component measurements justifies a
production change. The existing kernels remain in use; no full-model speed
claim follows from this screen. Sources, all timings, build commands and
exact-comparison fixtures are in
[`bench/results/2026-10-03-sycl-iq4-xmx-screen/run.json`](../bench/results/2026-10-03-sycl-iq4-xmx-screen/run.json).

### Cache experiments retained as measurements

On 2026-10-03, B570/5600X, normal cached L1/L2 gather properties were tested
on Q2_0 expert kernels. Twelve recorded sets of ten experts used warm weights,
50 captured repeats and the median of three device-event intervals. Three
alternating trials gave median sums of `1.870 ms` with default gathers,
`1.874 ms` with hints on Q8 inputs only, and `1.912 ms` with hints on inputs
and weights. Every set retained the SPMD/ESIMD output bits. Neither variant
improved these measurements, so the default gathers were restored.

A separate profile experiment ranked expert pairs from 128 generated tokens
on each of three prompts: seasons, a community garden, and notebook arithmetic.
`tools/make_profile.py --no-base` ranked those traces without placing the
shipped profile first. Story and Python merge-function prompts were withheld.
With 2,048 fixed per-layer slots, four workers, MTP T=4, probability threshold
0.5, prefill 16, context 512, direct PLE I/O and adaptive swaps disabled,
single 128-token trials measured:

| Withheld prompt | Shipped profile, token/s | Trace-ranked profile, token/s |
| --- | ---: | ---: |
| Story | 15.50 | 14.31 |
| Python merge function | 18.78 | 19.79 |

Changing placement changes which experts use GPU Q8_1 arithmetic instead of
CPU arithmetic. The continuations first differed at output indices 32 and 46;
these rates therefore include different generated text and MTP acceptance.
On the story, CPU pool time fell from `52.1` to `43.3 ms/round`, while accepted
proposals changed from `66/109` to `54/119`. This experiment does not justify a
default profile change. No output-quality comparison was performed.
Background workloads were not isolated.

Options, calibration prompts, continuations, an experimental profile and
individual cache-hint measurements are in
[`bench/results/2026-10-03-sycl-cache-experiments/`](../bench/results/2026-10-03-sycl-cache-experiments/).

### Cache, worker and large-page screens

After the verifier overlap change and before Q2_0 XMX, one 128-token run per
configuration used B570/5600X, Q2_0, FP16 KV, resident RAM, context 512,
prefill 16, the shipped profile, MTP T=4, probability threshold 0.5 and direct
PLE I/O. Auto sizing selected 3,792 expert slots. Workers below exclude the
host thread.

| Cache and workers | Story, token/s | Python merge function, token/s |
| --- | ---: | ---: |
| Fixed 2,048 slots, 4 workers | 14.61 | 16.54 |
| Auto, 4 workers | 13.77 | 21.17 |
| Auto, 5 workers | 17.25 | 19.11 |
| Auto, 4 workers, up to 512 swaps every 4 rounds | 15.73 | 16.55 |

Dynamic placement reached 67.33% story cache hits and 56.00% coding hits,
but refill time was respectively `12.88` and `21.03 ms/round`. Continuations
first differed from the fixed-cache run at index 32 for the story, 46 for
static auto coding and 96 for dynamic coding. These settings therefore include
different output text and acceptance counts. Shared background load was not
isolated; single samples do not select a worker count or justify a default
change. Flags, counters and outputs are in
[`bench/results/2026-10-03-sycl-config-screen/run.json`](../bench/results/2026-10-03-sycl-config-screen/run.json).

A separate allocation prototype requested `MADV_HUGEPAGE` on just the
anonymous expert arena, following the [kernel's per-region THP interface](https://www.kernel.org/doc/html/latest/admin-guide/mm/transhuge.html).
The system's existing THP policies were `madvise`; the prototype changed no
global setting. At session readiness, `/proc/PID/smaps` reported zero arena
`AnonHugePages` before and `32,006,144 / 32,563,200 KiB` after in the story
and coding processes. Advice succeeded, but their single paired runs slowed:
`16.66 → 15.23` and `17.34 → 16.32 tok/s`. Arena loading also slowed from
`0.88 → 0.55` and `1.28 → 0.53 GiB/s`. Both pairs retained identical ids,
acceptance and cache counts, with Q2_0 XMX disabled and 2,048 fixed slots.
The measurements do not isolate the cause of the slowdown. The prototype was
reverted. Its patch and measured mappings are in
[`bench/results/2026-10-03-sycl-arena-thp/run.json`](../bench/results/2026-10-03-sycl-arena-thp/run.json).

### Small adaptive refills and completed window boundaries

SYCL now completes scheduled adaptive refills before the next verify window
uses their residency map. Copies still overlap the preceding commit and
draft. `STRATA_SYCL_ADAPT_SYNC=0` retains the earlier readiness query. Failed
event creation, synchronization or query stops the request before committing
an incomplete refill. Tensor kernels and quantization are unchanged.

The earlier readiness query could change which windows computed an expert
on the CPU or GPU. Two identical fresh-process story runs with auto sizing
and 64 swaps every four rounds first differed at output index 85, with
41,287 versus 42,635 cache hits. The completed boundary fixes that timing
choice; different cache sizes or policies can still produce different ids.

On 2026-10-03, B570/5600X, two 64-token diagnostic runs with the completed
boundary retained all float32 bits across 32 windows/98 rows, including
rejected drafts. Three alternating 128-token pairs per prompt used 3,792
auto-sized slots, the shipped profile, four workers, context 512, FP16 KV,
prefill 16, MTP T=4, minimum probability 0.5, PCIe share 0 and direct PLE:

| Prompt | Readiness query, median token/s | Completed boundary, median token/s |
| --- | ---: | ---: |
| Story | 14.17 | 14.00 |
| Python merge function | 24.88 | 25.05 |

These runs establish no substantial generation speed gain. CPU workloads
were not isolated. All three completed-boundary runs per prompt retained
identical ids, cache counts and draft acceptance.

A preceding single-trial settings screen compared static residency with
64 refills every four rounds. Story measured `16.07 -> 19.18 token/s`;
Python measured `23.26 -> 25.32 token/s`. More frequent or larger refills
did not establish a better setting. These are exploratory configuration
measurements with different continuations, not repeated speed trials. Their
CPU intervals differ from the later boundary tests.

Two fresh persistent processes each served three 128-token story requests,
then three Python requests, cancellation and another eight-token story
request. Their first six requests retained identical ids and cache counts;
both resumed the known eight ids after cancellation. The third Python
request measured `28.88` and `28.77 token/s`; its expert cache was already
warmed by the preceding requests. The first story request measured `4.95`
and `11.13 token/s`, including lazy graph preparation. Warm story requests
measured `21.26..21.59 token/s`. These conditions remain distinct from fresh
CLI generation and do not establish a general 29 token/s result.

Flags, continuations and all samples are in
[`bench/results/2026-10-03-sycl-small-adaptive/run.json`](../bench/results/2026-10-03-sycl-small-adaptive/run.json)
and [`bench/results/2026-10-03-sycl-adapt-boundary/run.json`](../bench/results/2026-10-03-sycl-adapt-boundary/run.json).

### Graph submission completion events

The SYCL adapter can bind a disabled-timing event directly to a graph
submission. Completion covers every node, including host payload copies,
without a separate barrier request. Ordinary route, shared-expert and native
hit graphs use this binding. Verifier segments and asynchronous state commits
use it too. Profiling events keep the existing barrier path.

The regression test destroys the shared recording queue before replaying two
graphs on alternating queues. It changes host payloads at completed boundaries,
checks a cross-queue dependency on graph completion, and rebinds the event after
that dependency has been submitted. The adapter rejects timing-enabled events
and elapsed-time queries for these completion bindings.

On 2026-10-03, B570/5600X, the cold 37-token story, 4,135 fixed slots, resident
RAM, four workers, PLE RAM I/O, prefill 16, context 512 and speculation disabled:

| Trial | Separate event recording, token/s | Direct completion binding, token/s |
| --- | ---: | ---: |
| 1 | 21.93 | 22.10 |
| 2 | 21.90 | 21.88 |
| 3 | 21.84 | 22.15 |

Medians were `21.90` and `22.10 token/s` (`+0.9%`). All six runs produced the
same 128 ids and cache hit counts. All 32 cold logit rows were bitwise equal.
A functional MTP probe retained its 32 ids, 13 rounds and 19/28 accepted drafts;
it does not establish MTP throughput. Background workloads were not isolated.
PLE RAM pages remained unlocked. Full settings and individual measurements are
in
[`bench/results/2026-10-03-sycl-graph-completion/run.json`](../bench/results/2026-10-03-sycl-graph-completion/run.json).

### Event-completed speculative windows

SYCL verification captures the window in segments. The first segment embeds
the tokens and runs the first layer's mixer and router. Its completion event
must finish before the host reads routed activations. The CPU publishes its
expert plan, then a separate graph runs the shared expert, activation
quantization and resident expert projections while the CPU computes missed
experts. After the CPU and expert DMA finish, the next segment consumes their
results, combines them with the GPU results, and prepares the next layer. All
graphs use the same ordered queue. The next routing event completes the prior
plan reads before the CPU reuses its host plan or results. No device kernel
waits for a host flag. The last segment finishes the final layer and output head.

On the 48-layer model, an unsplit window has 49 segments; `--spec-split` uses
97 for two token groups, with respectively 48 or 96 shared/hit graphs.
`STRATA_SYCL_VERIFY_OVERLAP=0` retains the previous serialized graph order for
comparison. CUDA's device-plan shortcut and global-timer profiler are
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

On 2026-10-03, three alternating serialized/overlapped MTP runs on B570/5600X
used the same story, 128 generated tokens, context 512, FP16 KV, four CPU
workers, 2,048 fixed per-layer slots from the shipped profile, resident RAM
experts, 16-token prefill, T=4, minimum draft probability 0.5 and PCIe share 0.
Before measured `15.67 / 15.98 / 15.43 tok/s`; after measured
`16.82 / 16.31 / 16.44 tok/s`. The medians were `15.67` and `16.44 tok/s`,
a `4.9%` increase. All six runs retained the same 128 ids, 63 rounds,
66/109 accepted drafts and 8,705/82,560 expert-cache hits. Afterward,
`3019..3021` of `3024` shared/hit graphs were complete when the CPU pool
returned. GPU-reach wait fell from `52.02..52.17` to `42.32..43.13 ms/round`;
CPU pool time was `51.18..55.73 ms/round` across both variants. Startup and
prefill are excluded; background workloads were not isolated. This remains
below the approximately `29 tok/s` target.

Separate diagnostic runs compared every float32 logit bit, including rejected
draft rows: 30 windows/90 rows for 64-token MTP; 17 windows/129 rows for
T=8 split groups with every third oracle draft corrupted; and 7 windows/17
rows for a prompt starting at position zero. All bits matched the serialized
order. These diagnostic runs are excluded from the speed comparison.
`STRATA_SYCL_VERIFY_LOGITS=PATH` appends complete windows; use fresh paths and
`tools/sycl/compare_verify_logits.py --require-exact` to compare them. The
withheld-result probe still drains GPU work before returning an error.
Both graph orders also passed persistent requests: two identical 8-token
requests, cancellation after the first emitted token, then another 8-token
request. Each completed request retained the known continuation's ids.

Flags, output ids, full-window comparisons and persistent-request checks are
in [`bench/results/2026-10-03-sycl-verify-overlap/run.json`](../bench/results/2026-10-03-sycl-verify-overlap/run.json).

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

On 2026-10-03, the Python server also passed real-model HTTP checks on
`127.0.0.1:18085`, with the B570/5600X, context 512, native Q2_0, 2,048
profiled expert slots and MTP width four. `/v1/models`, the web page,
OpenAI non-streaming and streaming chat, and Anthropic messages returned
success. Two identical eight-token requests returned the same text. Closing
a 128-token stream after its first content chunk cancelled generation; the
next eight-token request returned the preceding text. A complete stream
also matched the non-streaming response. The server stopped cleanly after
the checks. This is functional validation, without an HTTP speed comparison.
Response and validation summaries are in
[`bench/results/2026-10-03-sycl-http/run.json`](../bench/results/2026-10-03-sycl-http/run.json).

The same HTTP checks also passed after changing the local config to automatic
cache sizing (3,792 slots on this run) and 64 adaptive replacements every four
verification rounds, with the completed SYCL refill boundary. Repeated output,
complete streaming output and cancellation recovery matched. This check does
not compare speed. Response summaries, flags and validation results are in
[`bench/results/2026-10-03-sycl-adaptive-http/run.json`](../bench/results/2026-10-03-sycl-adaptive-http/run.json).

The local config is `~/.local/share/strata-sycl/serve-config.json`. From this
checkout, start it with the installed environment:

```sh
source /opt/intel/oneapi/setvars.sh
PYTHONPATH=tools ~/.local/share/strata-sycl/venv/bin/python -m serve.server \
  --engine strata --config ~/.local/share/strata-sycl/serve-config.json \
  --host 127.0.0.1 --port 18085
```

Its OpenAI base URL is `http://127.0.0.1:18085/v1`; the current IQ3_S config names the model
`qwen3.8-flash-next-iq3_s-sycl` and retains `qwen3.8-flash-next-sycl` as an alias. The persistent engine currently requires MTP and
prefill. The local config and model assets are outside the checkout.

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
The expert and full-model comparison summaries are recorded in
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


### IQ expert and prompt performance tuning (2026-10-04)

On the B570/5600X with oneAPI 2026.1.1, native IQ3_XXS, IQ3_S,
IQ2_S, IQ4_NL and IQ4_XS grouped experts now use explicit SIMD where
measured faster. The reduction keeps the original 128 virtual lanes and
FP32 addition order. Wider verification windows keep the existing gate/up
weight reuse; prompt products with multiple rows keep their eight-row tile.
`STRATA_SYCL_IQ_ESIMD=0` restores the preceding IQ dispatch for comparison.

`sycl_moe_bench` reads eight real experts from each of the IQ3_S model's
seven gate/up–down type pairs. Against the preceding default (including
its eight-row grouped path and Q2 XMX), complete grouped products took
1.50–2.17 times less time for one row per expert. For four rows, pairs with
IQ4_NL down weights improved by 1.47–1.51 times; pairs with Q2_0 down
weights were unchanged within 0.3%. All 56 measured result arrays were
bitwise identical, including SwiGLU and hidden-activation quantization.
These GPU timings exclude transfers and CPU misses. They do not measure
whole-model generation speed.

Aligned native prompt gathers copy 16 bytes per work-item; unaligned sizes
retain the scalar path. Tests cover both paths and untouched output padding.
The full explicit-SIMD implementation passed 306 native projection cases;
17-format prompt tests and real-weight comparisons also passed. Persistent
writing, coding, cancellation and recovery requests retained the baseline
output ids. First-use compilation makes that initial integration run
unsuitable for a throughput claim.

The benchmark uses five graph replays of ten products after warm-up. Raw
captures remain outside Git; the reviewed timings and validation scope are
in [`bench/results/2026-10-04-sycl-performance/run.json`](../bench/results/2026-10-04-sycl-performance/run.json).


For persistent engine measurements, `tools/sycl/serve_bench.py` reads a serve
config and named token files. It records engine-reported prompt/decode times,
request wall time, ids, acceptance counters in each `DONE` line, binary hash,
and tuning environment. It runs each prompt twice, cancels another request,
then verifies eight-token recovery. `--compare PREVIOUS.json` requires every
completed request to retain its ids. The config's working directory and
environment are used; explicit process environment values take precedence.
For example, after initializing oneAPI:

```sh
python3 tools/sycl/serve_bench.py \
  --config ~/.local/share/strata-sycl/serve-config-iq3_s.json \
  --prompt writing=bench/results/2026-10-03-sycl-mtp-floor/strata-sycl-writing-tokens.txt \
  --prompt coding=bench/results/2026-10-03-sycl-mtp-floor/strata-sycl-profile-coding-tokens.txt \
  --output /tmp/strata-serve-bench.json --workers 4 --prefill 16 --spec-min-p 0.9
```

An exploratory 827-id prompt (826 prefetched tokens) with context 2,048,
1,649 fixed cache slots, adaptive swaps off and eight generated ids took
49.49 s at chunk 64 before these changes. The updated kernels with chunk
512 took 18.48 s, and chunk 1,024 took 14.48 s. All eight ids agreed.
These are single diagnostic runs with phase profiling, not repeated final
throughput estimates; some early baseline runs overlapped host compilation.

A separate transfer queue stalled while both GPU engines reported busy and
the main/staging threads waited. Completing ring-slot reuse waits on the
host allowed it to finish, but chunk 512 took 19.02 s. Streaming all experts
from chunks of 256 tokens took 16.53 s with that copy queue, or 16.97 s
with one queue, both slower than the routed-only 14.48 s run. The copy-queue
prototype was removed from production code; its patch and measurements are
retained with the performance summary. System driver settings were unchanged.


A second prototype packed sixteen native experts into host GU/down buffers,
then uploaded each group in two copies. Its initial profiled prompt improved
from 14.48 to 13.91 s. With profiling unset, a same-binary pair measured
14.14 and 14.02 s; short-prompt screens showed no improvement. The prototype
and its extra 42.6 MB of host buffers were removed. Its exact patch and
correctness checks are retained in the performance summary.

SYCL's routed prompt path now checks cancellation at a completed MoE layer
boundary, before its host stager starts. At that point the single compute
queue has finished prior work, so the caller can restore borrowed expert
slots safely. This keeps larger prompt chunks interruptible. Streamed-all
and multi-device paths keep their preceding cancellation boundaries.
`serve_bench.py --cancel-during-prefill` adds a STOP after 50 ms, then tests
ordinary generation cancellation and eight-token recovery as well.


Final whole-model measurements used native IQ3_S on the same B570 (10 GiB),
Ryzen 5 5600X and 125 GiB RAM, with oneAPI 2026.1.1 and default driver settings.
The table reports decode token/s medians from three alternating before/after
pairs, followed by three selected-preset processes. Each process ran two
128-token writing requests and two 128-token coding requests. All used context
512, 2,137 automatically sized compact expert slots, 64 adaptive replacements,
MTP window four and minimum draft probability 0.9. Other host workloads stayed
active, filesystem caches were not cleared, and phase profiling was unset.

| Request | Before: chunk 16, workers 4 | Updated: chunk 16, workers 5 | Selected: chunk 1,024, workers 5 |
| --- | ---: | ---: | ---: |
| Writing, first request | 13.14 | 13.66 | 13.15 |
| Writing, repeated request | 14.59 | 15.60 | 15.89 |
| Coding, first request | 16.26 | 17.12 | 17.21 |
| Coding, repeated request | 18.52 | 19.98 | 19.85 |

At the same chunk size, all completed output ids, expert-cache hit counts and
MTP acceptance counts matched the baseline in every trial. The repeated
writing and coding requests gained 6.9% and 7.9% decode throughput. The selected
larger chunk gained 8.9% and 7.1% on these repeated requests. Its first-request
prompt time fell from 4.49 to 3.52 s for writing and 5.76 to 3.79 s for coding.
The first-request outputs differ after changing the chunk size; the repeated
128-token outputs match. Batched FP16 computation changes rounding, so this
preset comparison is not a claim of bitwise equivalence or measured task quality.

With STOP scheduled 50 ms after GEN, median GEN-to-DONE time fell from 1.763 s
to 0.069 s at chunk 16; the selected chunk 1,024 took 0.142 s. These numbers
include the initial 50 ms and are not direct STOP-to-DONE measurements.
All nine processes passed prefill cancellation, cancellation after the first
output token and eight-token recovery, then exited normally.

For the 827-id long prompt, three alternating normal-execution pairs measured
826 prefetched tokens at a median 46.898 s before and 14.083 s after (3.33 times
faster). Time to first token was 47.229 versus 14.280 s. This comparison uses
context 2,048, 1,649 fixed cache slots, four CPU workers and adaptive swaps off;
it combines kernel changes with chunk 64 to 1,024. All eight generated ids
matched in all six runs. It does not measure the default context-512 workload.
The prompt fixture and each trial's timings are stored with the performance
summary above.

The selected local text configs (`serve-config.json` and
`serve-config-iq3_s.json` under `~/.local/share/strata-sycl/`) now use
`--prefill 1024 --pool-workers 5`. Their context limit remains 512. Original
configs are backed up under
`~/.local/state/strata-sycl/measurement-archive/2026-10-04-performance/config-before/`.

Final validation passed all 38 default CTest cases and the local HTTP checks
for model discovery, the web UI, OpenAI and Anthropic text requests, complete
stream equality and cancellation recovery. Seven image checks passed with the
SYCL encoder: repeat single-image answers, Anthropic image input, streaming,
text after an image, ordered red/blue images and a single image after two images.
Both servers exited with code zero and released their ports.
`serve-config-vision-sycl.json` also uses the selected chunk and worker count,
with its existing context 1,024 and adaptive swaps off. These image checks
validate behavior; they are not a vision-throughput comparison. Raw captures,
saved binaries and reproduction scripts are archived outside Git beside the
config backups. The JSON summary records binary hashes, flags, individual
trials, output ids and API results.

### Continued speed target (2026-10-04)

The ongoing target is 1,000 prompt tokens/s and 70 generated tokens/s on this
B570/5600X workstation. These targets have not been reached. The measured
baseline remains the preceding three-run results; aggregate serving throughput
will not be substituted for one request's generation rate.

`-DSTRATA_CPU_ARCH=znver3` builds the CPU expert library with
`-march=znver3 -mtune=znver3`. The default empty value retains the preceding
compiler settings. This option makes the binary specific to the selected CPU;
it is rejected with `STRATA_PORTABLE` or MSVC. GGML already used `-march=native`
on this machine, so the new option targets Strata's own CPU expert code.

The first diagnostic screens, using the preceding binary and 826 prefetched
tokens, measured 13.289 s with FP16 expert expansion and 19.867 s with the
existing fused native path (stream threshold 256). Both produced the same eight
output ids. These single, profiled runs identify candidates; they do not establish
a new default or a repeated throughput improvement.

Reference work extends beyond SYCL: [Marlin](https://github.com/IST-DASLab/marlin)
for weight layout, activation reuse and double buffering;
[KTransformers](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md)
for tiling around the CPU cache and instruction set;
[Fiddler](https://github.com/efeslab/fiddler) for CPU/GPU expert placement; and
[AITER](https://github.com/ROCm/aiter) for fused MoE kernels. These guide experiments,
not speed claims for this machine. In particular, AMX instructions cannot run on
this Ryzen; the transferable idea is the layout and scheduling method.
The [ongoing measurements](../bench/results/2026-10-04-sycl-speed-goal/run.json)
record the target, experimental conditions and reference links.

The first Zen 3 build retained all completed writing/coding output ids against
the preceding selected preset, passed prefill/decode cancellation and recovery,
and passed 38/38 CTest cases. Its single-run repeated-request rates were 16.05
and 20.10 token/s. More repetitions are needed to attribute any change to the
compiler flags; this is a validated tuning baseline, not a final speed claim.

The CPU/GPU responsibilities and CPU fallback policy remain based on upstream.
Kernel and data-layout experiments operate inside those existing paths. A
placement-policy change needs a measured reason. Microbenchmarks check numerical
agreement and screen candidates; adoption depends on repeated full inference
runs, including transfers, synchronization and memory use.

An opt-in prompt experiment, `STRATA_SYCL_MMQ_XMX=1`, uses Intel XMX integer
products for Q2_0, IQ3_XXS, IQ3_S, IQ2_S, IQ4_NL and IQ4_XS. It shares sixteen
weight rows across activation tiles while retaining GGUF weights and Q8_1
activations. `STRATA_SYCL_MMQ_XMX_TILE` accepts 1, 2, 4 or 8, with 4 as the
experimental default. The default reduction preserves the original virtual-lane
FP32 addition order. Boundary tests compare its output bits with the preceding
kernel, including partial tiles, routed row maps and multiple contributions per
virtual lane, as well as checking a separate dequantized reference.

`STRATA_SYCL_MMQ_XMX_EXACT=0` selects a diagnostic linear reduction. This earlier
variant passed numerical-tolerance checks, but a repeated writing request
changed generated ids starting at index 11. It is not an equivalent-output
throughput result and is not selected in local serving configurations. A
larger-state exact variant matched output bits but was slower in the component
screen; it was rejected. Kernel-by-kernel device linking was also tested and
reverted after first-use compilation affected full inference timings. The
experiment keeps the existing linker settings and remains disabled by default.

The exact XMX variant passed 39/39 CTest cases, including 31 prompt geometries
with the experiment enabled. A persistent real-model screen with tile 8 matched
all completed writing/coding token sequences against the Zen 3 baseline and
passed prefill/decode cancellation and eight-token recovery. Repeated-request
decode rates were 15.98 and 20.25 token/s in this single screen. First writing
prefill took 7.511 s versus 3.571 s in the prior baseline screen, so first-use
costs need separate evaluation. These results do not establish a throughput gain;
no serving configuration has adopted the experiment.

Three alternating normal-execution pairs then measured the 827-id long fixture
(826 prefetched tokens) using the same executable, context 2,048, 1,649 cache
slots, five CPU workers, chunk 1,024 and adaptive swaps off. With XMX disabled,
prefill times were 14.084, 14.074 and 14.743 s; with exact XMX tile 8 they were
9.705, 9.644 and 9.652 s. The medians are 58.65 versus 85.58 token/s (1.46 times
faster). All eight output ids matched in all six runs. Profiling was disabled
and compilation caches had been warmed by the preceding validation. This is a
long-prompt prefill result; it does not resolve the first-use latency issue or
establish a generation-speed gain. The target remains unmet.

### Compiler and runtime review (2026-10-04)

This review checks the local oneAPI 2026.1 compiler against Intel's compiler
manual, design documents and GPU optimization guide. Online design documents
can describe newer or proposed behavior; the local compiler invocation and
actual build remain the check for this workstation.

| Area | Finding and consequence for Strata |
| --- | --- |
| Host and device compilation | CPU `-march=znver3` affects the CPU expert library. GPU architecture selection is a separate step. Release already uses `-O3`; adding it again is not a new optimization. |
| AOT target | The new driver accepts `--offload-new-driver --offload-arch=bmg_g21`, but the local runtime probe below fails. `STRATA_SYCL_DEVICE_ARCH` therefore uses the old driver with `-fsycl-targets=spir64_gen` and OCLOC `-device`; empty retains JIT. A GPU-specific build is incompatible with `STRATA_PORTABLE`. |
| Backend options | `-Xsycl-target-backend` reaches IGC/OCLOC. Frontend options such as `-fgpu-inline-threshold` do not directly tune IGC. Inspecting only the C++ command is insufficient; the generated backend invocation matters. |
| Optimization levels | The documented JIT mapping sends `-O1`, `-O2` and `-O3` to the same Level Zero backend optimization level. Frontend optimizations can still differ. The design explicitly excludes AOT, so its mapping is not evidence about AOT output. |
| Code splitting | `auto` is the default; `per_source` and `per_kernel` change which kernels share an image. ESIMD splitting is separately enabled by default. More images can increase module creation and first-use costs. The preceding per-kernel full-inference experiment was reverted. |
| Device linking | Relocatable device code is enabled by default. Disabling it restricts cross-translation-unit device calls. The NoRDC design document describes the old offload model; its build-time benefits must not be assumed for the new driver. |
| GRF allocation | The initial XMX prompt kernel requested 256 GRFs. JIT runtime code converts that property to `-doubleGRF` for ESIMD. The first AOT build omitted it and failed on all six exact tile-8 instantiations. An indirectly addressed 8 KiB vector could not fit in the 128-GRF allocation. Explicit backend `-doubleGRF` passed this compilation stage. It affects other ESIMD images too, so their performance needs rechecking. |
| Floating point | Local `icpx -###` confirms that `-ffp-contract=off` alone still leaves unsafe-math, reciprocal and signed-zero transformations enabled. The exact IQ/XMX sources also use `-fno-fast-math`. That combination removes those frontend flags, but does not automatically request correctly rounded device division/square root. Preserve the measured numerical contract when testing flags. |
| Register diagnostics | The AOT log reports spills for Q4_K/Q5_K SPMD kernels, including prompt products. A spill warning is evidence of scratch traffic, not evidence that these kernels dominate this mixed-format model. Profile the executed kernels before changing their register mode. |
| ESIMD memory access | The optimization guide recommends contiguous block messages, suitable vector widths and avoiding indirect register indexing where practical. Our weight-row gathers are candidates for GPU-side layout changes. Include repacking and transfers in full-inference timing. Kernel-lambda forced inlining does not apply to ESIMD according to the compiler manual. |
| Caches | SYCL in-memory caching defaults on, its persistent cache defaults off, and the compute runtime has its own persistent compiler cache enabled by default. A fresh process is not a cold compilation measurement. Benchmark records now include explicit SYCL and NEO cache settings. Use separate cache directories for cold/warm comparisons without deleting the user's cache. |
| Level Zero adapter | Current documentation selects V2 for Xe2/Battlemage. Several `SYCL_PI_LEVEL_ZERO_*` variables apply only to the legacy adapter. Check the installed runtime before using older tuning recipes. `UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD` and `UR_L0_V2_FORCE_BATCHED` are diagnostic candidates, not adopted defaults. |
| Graphs and queues | Graph capture does not replay ordinary host work in a command-group callback. Host-task nodes can introduce synchronization. Strata already uses explicit contexts, in-order compute/transfer queues and graph replay; adding graph capture alone would not remove CPU expert work or transfer waits. |

Sources: [compiler manual](https://intel.github.io/llvm/UsersManual.html),
[offload driver](https://intel.github.io/llvm/design/OffloadDesign.html),
[compiler/runtime architecture](https://intel.github.io/llvm/design/CompilerAndRuntimeDesign.html),
[optimization-level propagation](https://intel.github.io/llvm/design/PropagateCompilerFlagsToRuntime.html),
[NoRDC design](https://intel.github.io/llvm/design/NonRelocatableDeviceCode.html),
[device linking](https://intel.github.io/llvm/design/SharedLibraries.html),
[GRF property](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_intel_grf_size.asciidoc),
[JIT option handling](https://github.com/intel/llvm/blob/sycl/sycl/source/detail/program_manager/program_manager.cpp),
[floating-point settings](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2026-0/floating-point-optimizations.html),
[ESIMD optimization](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/optimizing-explicit-simd-kernels.html),
[spill diagnostics](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/finding-kernels-with-register-spills.html),
[SYCL cache design](https://intel.github.io/llvm/design/KernelProgramCache.html),
[driver cache](https://github.com/intel/compute-runtime/blob/master/programmers-guide/COMPILER_CACHE.md),
[environment variables](https://intel.github.io/llvm/EnvironmentVariables.html), and
[graph guide](https://intel.github.io/llvm/syclgraph/SYCLGraphUsageGuide.html).

The new-driver AOT build with explicit `-doubleGRF` completed, but runtime
validation failed: the ordinary prompt-MMQ test passed and the XMX test failed
with `Sub-group size 1 is not supported on the device`. The generated properties lack `isEsimdImage`, so the runtime treats the ESIMD
subgroup requirement as an ordinary unsupported subgroup. The small
[`aot_esimd_probe.cpp`](../bench/results/2026-10-04-sycl-speed-goal/aot_esimd_probe.cpp)
reproduces this with the new driver in both JIT and AOT mode. The old driver
passes in both modes. New-driver split modes `off`, `per_source` and `per_kernel`,
and its in-process post-link alternative, did not fix it. This is evidence for
the installed compiler, not a claim that every version of the new driver fails.
The old-driver AOT build passed ordinary MMQ and exact-XMX MMQ tests
(2/2 in 1.04 s). The broader AOT suite has not been run for this candidate;
the full-inference comparison below determines the next experiment. The
working JIT executable and serving configs remain the baseline.

The initial AOT configuration added `-DSTRATA_SYCL_DEVICE_ARCH=bmg_g21`
to the existing Release/Zen 3 configuration, with
`-DSTRATA_SYCL_AOT_DOUBLE_GRF=ON` passing `-doubleGRF` at device link.
Without it, that version of the exact tile-8 XMX kernel failed to compile.
It applies large registers to all ESIMD images, so validation includes decode
kernels. The subsequent reduction rewrite below removes this requirement;
the option now defaults off and remains available for diagnostics.
The new-driver probe uses separate compile and link commands: passing the
backend option in a combined source-and-link command duplicated `-options` in
the OCLOC invocation on this compiler and failed for a separate reason.

To reproduce the small new-driver AOT failure outside the engine, run inside
an initialized oneAPI environment in a scratch directory:

```sh
icpx -fsycl -O3 --offload-new-driver --offload-arch=bmg_g21 \
  -c /path/to/aot_esimd_probe.cpp -o probe.o
icpx -fsycl -O3 --offload-new-driver --offload-arch=bmg_g21 \
  -Xsycl-target-backend=spir64_gen '-options -doubleGRF' probe.o -o probe
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./probe
```

For the passing old-driver comparison, replace the two new-driver switches
with `-fsycl-targets=spir64_gen` in both commands and replace the backend option
string with `'-device bmg_g21 -options -doubleGRF'`. The source and its expected
sixteen values of `3.0f` stay the same.

Eight persistent real-model runs then compared JIT and old-driver AOT with
exact XMX tile 8, the same context-512 text configuration and 128-token writing
and coding requests. Each executable started with independent empty SYCL and
NEO compiler-cache directories, with both persistent caches enabled. Three
alternating warm pairs reused those directories. All completed output ids
matched the preceding exact-XMX baseline; prefill cancellation, decode
cancellation and eight-token recovery passed in all eight processes.

| Measurement | JIT | AOT with all ESIMD images at 256 GRFs |
| --- | ---: | ---: |
| First writing prompt, empty compiler caches (one run each) | 19.857 s | 3.530 s |
| First writing prompt, warm-cache median (three runs each) | 3.408 s | 3.408 s |
| Repeated writing decode, warm-cache median | 15.92 token/s | 15.23 token/s |
| Repeated coding decode, warm-cache median | 20.16 token/s | 19.26 token/s |

This AOT configuration removes the observed first-use prompt compilation cost,
but reduces repeated writing/coding generation rates by 4.3%/4.5%. It is not
adopted in serving configs. The broad ESIMD register override is a possible
cause, not yet an isolated attribution. Reducing the XMX kernel's register
requirements is the next experiment. Startup also includes reading 46.84 GiB
of expert weights; the OS file cache was not cleared and read rates varied, so
startup differences are not attributed solely to compilation. Full trials,
flags, binary hashes, cache settings and output ids are in the ongoing JSON.

### Exact XMX reduction with 128 GRFs (2026-10-04)

The next kernel computes the same XOR-8/4/2/1 reduction in a statically expanded
tree. It completes each subtree before visiting the next, keeping four pending
vectors instead of an indirectly indexed 8 KiB array. Each cell's accumulation
order is preserved. An intermediate 4 KiB-array version passed output checks
but still spilled 1,536–2,688 bytes in the tile-8 AOT variants and was rejected
before performance adoption.

With the tree version, AOT device metadata reports 128 GRFs and no scratch
buffers for all 48 XMX instantiations. All 72 ESIMD kernels in the full engine
also use 128 GRFs without scratch buffers. The broad `-doubleGRF` override is
no longer required, and `STRATA_SYCL_AOT_DOUBLE_GRF` now defaults off.

Three alternating normal-execution pairs compared the preceding exact-XMX JIT
binary against the tree AOT binary on the same Arc B570/Ryzen 5600X workstation.
The 827-id fixture prefetched 826 tokens with context 2,048, 1,649 cache slots,
five CPU workers, chunk 1,024 and adaptive swaps off. Both used exact XMX tile 8
and reused the preceding experiment's compiler caches. Profiling was disabled;
no build or other benchmark ran concurrently.

| Long-prompt measurement | Previous exact-XMX JIT | Tree AOT, 128 GRFs |
| --- | ---: | ---: |
| Prefill times, three runs | 9.720, 9.615, 9.613 s | 8.971, 8.979, 8.978 s |
| Median prefill throughput | 85.90 token/s | 92.00 token/s |

All eight generated ids matched in all six runs. This comparison measures the
combined kernel and AOT change, not the contribution of either in isolation.

Three further alternating pairs used the persistent context-512 text
configuration, two 128-token writing requests and two coding requests per
process, with the same warmed compiler-cache directories. All completed ids,
prefill/decode cancellation and eight-token recovery passed in all six runs.

| Median measurement | Previous exact-XMX JIT | Tree AOT, 128 GRFs |
| --- | ---: | ---: |
| First writing prompt | 3.415 s | 3.485 s |
| Repeated writing prompt | 0.972 s | 0.992 s |
| First coding prompt | 3.674 s | 3.706 s |
| Repeated coding prompt | 0.948 s | 0.969 s |
| First writing decode | 13.43 token/s | 13.25 token/s |
| Repeated writing decode | 15.81 token/s | 15.89 token/s |
| First coding decode | 17.35 token/s | 17.37 token/s |
| Repeated coding decode | 19.73 token/s | 19.99 token/s |

The preceding broad-256-GRF AOT generation slowdown is absent in these paired
measurements. Short-prompt times are slightly higher, and these small decode
differences do not establish a general generation-speed gain. The measured
improvement is long-prompt prefill. Both speed targets remain unmet.

The final JIT and old-driver AOT builds both passed all 39 CTest cases (33.75 s
and 32.80 s respectively). Exact tiles 1, 2 and 4 also passed all 31 prompt
geometries in each build, in addition to the suite's tile-8 checks. These compare
bits with the preceding SPMD implementation and check an independent GGML
reference. The final AOT engine hash matches the executable used for the
performance comparisons.

An HTTP run also passed seven checks with the SYCL image encoder: repeated
single-image input, OpenAI and Anthropic responses, OpenAI streaming, text after
an image, two-image ordering and returning to a single image. The server exited
cleanly and closed its localhost port. This run used the normal
`SYCL_CACHE_PERSISTENT=0` setting. Explicitly enabling that setting for the
unchanged image encoder caused a startup segmentation fault; a standalone
comparison also passed with `0` and failed with `1`, before the Strata engine
started. The engine-only cache benchmarks above passed with it enabled. This
encoder limitation is recorded separately from the XMX results.

After these checks, the workstation's IQ3_S text and vision configurations use
`build-sycl-aot/strata` with `STRATA_SYCL_MMQ_XMX=1`,
`STRATA_SYCL_MMQ_XMX_TILE=8` and `STRATA_SYCL_MMQ_XMX_EXACT=1`. Previous configs
are archived. No persistent-cache override was added, and the Q2 configuration
was not changed. XMX remains opt-in in the source defaults. This becomes the
local tuning baseline; it does not meet the 1,000-prefill/70-generation targets.

### IQ3_S anonymous-arena THP recheck (2026-10-04)

The earlier Q2_0 THP result was rechecked with the current IQ3_S, Zen 3 and
128-GRF AOT baseline. The prototype requested `MADV_HUGEPAGE` before first touch
of the anonymous expert arena, respected `STRATA_NO_LARGEPAGES`, and changed no
system setting. At engine readiness, its three processes reported 48,928,768–
49,113,088 KiB of `AnonHugePages`, versus zero without advice, out of an arena
mapping of about 49,116,200 KiB. The inspection ran before timed requests.

Three alternating persistent-server pairs retained identical completed ids,
MTP acceptance and cache counts; cancellation and recovery passed. Median
results on the same B570/5600X were:

| Measurement | Ordinary pages | THP requested |
| --- | ---: | ---: |
| Startup | 21.52 s | 19.31 s |
| First writing decode | 13.43 token/s | 13.49 token/s |
| Repeated writing decode | 15.69 token/s | 15.94 token/s |
| First coding decode | 17.73 token/s | 17.52 token/s |
| Repeated coding decode | 19.80 token/s | 20.06 token/s |

Startup varied from 18.74 to 23.54 s across these runs; the OS file cache and
existing background workloads were not isolated. Three additional alternating
pairs on the 827-id long fixture (826 prefetched tokens, context 2,048,
1,649 cache slots, five workers, chunk 1,024 and adaptive swaps off) measured
median prefill times of 8.959 s without advice and 9.002 s with it, or 92.20
versus 91.76 token/s. All eight output ids matched in all six long runs.

Repeated generation improved slightly, but first coding and long prefill
regressed slightly. The prototype was reverted and serving configurations keep
the previous allocation behavior. These small mixed differences do not
establish a uniform improvement or a general conclusion about THP. The
[full trials and readiness mappings](../bench/results/2026-10-04-sycl-arena-thp-iq3_s/run.json),
[prototype patch](../bench/results/2026-10-04-sycl-arena-thp-iq3_s/prototype.patch)
and [benchmark snapshot](../bench/results/2026-10-04-sycl-arena-thp-iq3_s/serve_bench_snapshot.py)
preserve the experiment.

### CPU codebook and larger-prompt screens (2026-10-04)

Three CPU-only prototypes compared codebook access on real IQ3_XXS/IQ3_S gate
and up weights against the current Zen 3 build: AVX2 gather loads, packed SIMD
index calculation with scalar table loads, and that calculation combined with
a single-vector sign expansion. Each used eight experts, 640 rows, 2,560 columns
and 1/2/3/4/8-token groups, with five alternating paired timing samples. Every
output bit matched. None consistently improved the 2–4-token groups used by
the current speculative configuration. Some eight-token IQ3_S groups improved,
but that does not establish an engine gain. No prototype was integrated;
[raw timings, code patches and the CPU harness](../bench/results/2026-10-04-sycl-cpu-codebook/run.json)
are preserved. These warm component timings exclude host scheduling and data
transfers.

A separate real-model screen used a 4,008-id color-retrieval prompt with
context 8,192, 512 fixed cache slots, five workers, adaptive swaps off and exact
XMX tile 8. Chunk 1,024 prefetched 4,007 tokens in four chunks in 49.866 s
(80.36 token/s); chunk 4,096 took one chunk and 26.637 s (150.43 token/s).
Streamed expert counts fell from 86,957 to 24,064. All eight output ids matched,
including the expected first color token. This is one trial per setting, and
its longer input and smaller cache distinguish it from the preceding 827-id
baseline. No serving configuration changed. The
[fixture, flags, logs and scripts](../bench/results/2026-10-04-sycl-prefill-scale/run.json)
support further kernel work. The following experiment changes the GPU
scratch-weight layout while retaining the same arithmetic.


### Interleaved XMX scratch weights (2026-10-04)

`STRATA_SYCL_MMQ_XMX_PACK=1` enables an in-place rearrangement of disposable
prompt weights for exact XMX tile 8. A work-group reads sixteen complete rows
into local memory before overwriting them with interleaved half-words. Field
loads then read adjacent rows from neighboring addresses. The arrangement
keeps the compressed codes and the original floating-point reduction order;
it creates no additional global weight buffer. Persistent expert-cache storage
and CPU/GPU assignment are unchanged.

The caller must explicitly mark the weights writable and disposable and refill
them before the next product. The main prefill path does this only for freshly
gathered gate/up and down groups. Read-only weights, incomplete sixteen-row
tiles, unsupported types, and rows that exceed device local-memory limits keep
the original path. The flag is off by default.

Three alternating pairs on the Arc B570 10 GiB and Ryzen 5 5600X compared the
same AOT binary with the flag off and on. They used the preceding 4,008-id
fixture, 4,007 prefetched tokens, context 8,192, chunk 4,096, 512 fixed cache
slots, five CPU workers and adaptive swaps off. No profiling, compilation or
other GPU test overlapped these measurements.

| Median measurement | Original layout | Interleaved scratch |
| --- | ---: | ---: |
| Prefill time | 26,594.9 ms | 25,512.6 ms |
| Prefill rate | 150.67 token/s | 157.06 token/s |

The measured rate improved by 4.24%, including the rearrangement cost. All
six runs produced identical eight-token outputs with the expected first color
token. This result applies to this prompt and configuration; it does not
establish a decode-speed improvement.

The AOT image contains 54 XMX variants using 128 GRFs. The new packed IQ2_S
exact tile-8 variant has a 192-byte scratch spill; the other 53 have no scratch
buffers. The mixed IQ3_S model in this experiment has IQ2_S gate/up weights in
20 layers, IQ3_XXS in17, IQ3_S in10 and IQ4_XS in1, so that spill affects
a substantial part of this workload. The [run record and scripts](../bench/results/2026-10-04-sycl-xmx-packed/run.json)
preserve the flags, hashes, timings and output ids.

AOT and JIT each passed all 40 CTests. The MMQ tests cover 50 geometries with
an independent GGML oracle and bitwise comparison against the old SPMD path.
They also check the exact rearranged weight bytes, padded expert strides,
read-only matrices, incomplete row tiles, local-memory limits, unsupported
types, mapped output rows and guard bytes. After source formatting, all 519 GPU
text sections in the final AOT binary matched the measured binary.

A separate 827-id screen (826 prefetched, context 2,048, cache 1,649, chunk
1,024) took 9,143.0 ms without packing and 8,964.7 ms with it. Timing-enabled
runs were nearly equal, 9,351.3 and 9,347.1 ms. These are individual screens;
they do not establish a short-prompt gain. The profiled gate/up bucket fell
by 97 ms and the down bucket grew by 88 ms. These event intervals include
queue waits and host gaps, so they are not isolated kernel durations.

Normal persistent serving with each layout passed the same 128-token writing
and coding requests twice, cancellation during prefill and decode, and
eight-token recovery. Completed output ids matched the preceding exact-XMX
fixture. No decode-speed gain is claimed, and local serving configs still
leave packing disabled while more selective use is measured.

A fixed minimum-column prototype packed the 2,560-column gate/up product and
left the 640-column down product unchanged. It passed all 50 geometries in
four MMQ test modes on AOT and JIT. One normal screen improved the 826-token
prefill from 8,966.2 to 8,865.6 ms, but the 4,007-token screen worsened from
25,525.1 to 25,685.7 ms. The fixed-column rule was removed; the record preserves
its patch and timings. Routed-token reuse also matters for the cost tradeoff.


### Zen 3 IQ3_S loop expansion screen (2026-10-04)

Clang's IQ3_S AVX2 code on this Ryzen 5 5600X kept many decoded values live
across a fully expanded four-step loop. A prototype retained that loop while
allowing its two halves to expand. Warm component tests improved IQ3_S rates
for two to four tokens by about 8–17%, with identical output bits. Disabling
both loops or adding two integer accumulators did not give the same result.
An explicit unroll count for other formats also slowed IQ3_XXS, so the engine
prototype used a dedicated IQ3_S helper only for Clang targeting Zen 3.

The actual engine CPU object matched the old code for one through eight
tokens. AOT and JIT each passed the three MMQ tests. Six alternating normal
serving runs then compared the old and candidate AOT binaries with the same
context-512 configuration, five workers, adaptive swaps 64 and XMX packing off.
Every completed output id matched; cancellation and recovery also passed.

| Median decode rate | Old CPU kernel | Retained IQ3_S loop |
| --- | ---: | ---: |
| First writing request | 13.317 token/s | 13.502 token/s |
| Repeat writing request | 15.910 token/s | 15.825 token/s |
| First coding request | 17.452 token/s | 17.417 token/s |
| Repeat coding request | 19.743 token/s | 20.006 token/s |

The change did not establish a uniform engine gain. It was removed, and the
configured AOT binary was restored. These results illustrate why warm,
repeated-expert component gains do not by themselves justify an engine change.
The [record](../bench/results/2026-10-04-sycl-cpu-loop-unroll/run.json) preserves
all candidate patches, raw component samples and complete serving results.

The native expert manifest contains20 IQ2_S gate/up layers,17 IQ3_XXS,10
IQ3_S and1 IQ4_XS; down uses IQ4_NL in39 layers and Q2_0 in9. The CPU loop
prototype therefore changes10 of48 layers, and only the rows dispatched to
the multi-token AVX2 kernel. The earlier packed-XMX register record also needs
this distinction: its IQ2_S scratch spill applies to20 layers, not an unused
format. Further tuning uses this measured model composition.
