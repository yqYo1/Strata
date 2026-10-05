# Compact scratch and layer-major prompts on the Arc B570

These measurements use the same Intel Arc B570, native IQ3_S model,
oneAPI 2026.1, precise floating point and selected compute runtime as the
[parent study](../README.md). The host has 128 GiB of RAM. Native expert
weights stay in RAM; all K/V pages stay in VRAM. The long comparisons use
one fixed 65,538-token K8/V8 context, cache 128, attention batch 128/layout
4, the tuned selector and original key-head GDN arithmetic. They use
direct PLE reads with 16 I/O threads. PLE preload is disabled.

## What changes

`STRATA_PREFILL_COMPACT=1` reuses storage after its final consumer. The
FP16 expert path gathers and computes one expert's rows at a time, keeping
the original GEMM dimensions, dequantization, FP16 conversions and combine
order. `=2` also puts the HC inputs into the shared child workspace and
retains the original full-chunk PLE workspace. The scalar attention and
convolution/recurrence scratch reuse follows the existing in-order queue.
The original layout remains the default. MMQ, fused expert paths, XMX
attention and NaN diagnostics retain their original layout.

With compact scratch, `STRATA_PREFILL_LAYER_MAJOR=1` processes all chunks
through one layer before moving to the next. It uploads all 512 quantized
experts once per layer into a 1,363,148,800-byte cache. It stores FP32
residual rows between layers, preserving each layer's original chunk
shapes and token order. The option currently requires one GPU, the full
model, the FP16 expert path and all K/V pages resident in VRAM. It rejects
unsupported configurations. A one-chunk prompt uses the original traversal;
`=2` forces the layer traversal for correctness diagnostics.

`STRATA_PREFILL_LAYER_MAJOR_R_GPU=N` keeps a complete prefix of chunks in
VRAM and the remaining residual rows in RAM. Requests round down to a
chunk boundary, or clamp to the whole input. The default is zero. These
are ordinary FP32 copies: they do not change any values. Decode's expert
cache and callbacks are restored when prefill ends. Intermediate progress
reports do not save a whole-model checkpoint; the final chunk can save it.

## Complete output and state checks

The [nine compact/host checks](compact-and-host-validation/summary.json)
compare the original layout, compact HC scratch and layer traversal at
257 tokens/chunk 128, 4,096/chunk 4,096, and 8,087/chunk 4,096. They compare
all 248,320 finite first logits, every FP32 residual row and all saved
persistent state bytes, including authoritative K/V, scales, pooled
indexer values, recurrence/convolution state and PLE history. All bytes
and generated IDs match. Both the compact layout and layer traversal use
the same original shapes, including odd final chunks.

The [eight GPU-residual checks](gpu-residual-validation/summary.json)
repeat the original controls and test partially and entirely resident
residual rows. All complete output, residual and persistent-state bytes
match the original. The mixed 8,087-token case also uses ring 16; the
other controls use ring 8. All 30 registered JIT tests pass. The frozen
binary and source hashes, original comparison hashes, commands and memory
samples are in the records. Timing in these diagnostic runs includes
state downloads and is not a production throughput estimate.

The [normal MTP check](gpu-residual-long/normal-mtp.json) and
[long MTP check](gpu-residual-long/long-mtp.json) each exercise four
requests, including checkpoint reuse and restoration. IDs and logprobs
match the previously accepted original, and the available complete first
heads match. The long case forces layer traversal and keeps all residual
rows in VRAM. It exercises model state across prompt segments.

The current-source B570 `bmg-g21` AOT build succeeds. Its
[build record](aot-validation/build.json) and
[30 registered tests](aot-validation/registered-tests.json) are saved.
All 30 current-source AOT GPU tests pass in 93.84 seconds after host
restart. The [six AOT model cases](aot-validation/initial/summary.json)
preserve all head, residual and persistent-state bytes against both the
original controls and the corresponding JIT cases. The [normal and long AOT MTP checks](aot-validation/confirmation/) each
preserve four requests and checkpoint restoration. The 32K/8K-chunk CLI
allocation fails while a separate embedding server occupies about 1.73
GiB of the GPU; its failure record is preserved. That confirmation and
repeated normal-wall measurements remain pending an equivalent GPU
memory condition.

## Long-input measurements

The [host-residual records](host-residual-long/controller.py) use
4,096-token chunks at 16K, 32K and 64K. The
[GPU-residual records](gpu-residual-long/controller.py) keep the first
32,768 rows in VRAM at 32K and 64K. Each complete finite first head and
output ID matches its original K8/V8 control with the same chunk size.
These are single transfer-timed observations, not repeated medians.
GB below means decimal bytes.

| Input tokens | Traversal, residual storage | Prefill seconds | Tokens/s | Quantized expert GB | Host/device residual GB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 32,768 | Original | 115.251 | 284.32 | 363.564 | 0 |
| 32,768 | Layer-major, RAM | 112.551 | 291.14 | 50.292 | 126.165 |
| 32,768 | Layer-major, all VRAM | 86.299 | 379.70 | 50.292 | 0 |
| 65,536 | Original | 235.714 | 278.03 | 739.6 | 0 |
| 65,536 | Layer-major, RAM | 198.799 | 329.66 | 50.292 | 252.329 |
| 65,536 | Layer-major, first 32K in VRAM | 185.232 | 353.81 | 50.292 | 126.165 |

VRAM-to-VRAM residual copies are recorded separately and do not count as
PCIe traffic. With 32K rows resident, these copies total 126.165 GB and
about 0.765 seconds at either input length. Quantized expert DMA takes
about 8.1 seconds. Serial layer-load wall time is about 8.3 seconds and
includes CPU staging; it is not pure DMA time. Host residual DMA at 64K
takes about 20.0 seconds. The reported prefill wall time includes PLE
gathers, weight loading and all residual transfers.

The [1 Hz process samples](gpu-residual-long/memory-summary.json) observe
about 9.037 GiB of resident VRAM in the 4K-chunk, 32K-residual runs.
Peak process RAM is 47.288 GiB at 32K and 48.511 GiB at 64K. No process
swap is observed. Samples can miss transient peaks and describe the
engine process, not total card usage.

The [8K-chunk comparison](8k-chunk-long/controller.py) repeats the 32K
and 64K controls with the same 65,538-position K8/V8 context. All four
complete finite first heads and output IDs match within each original
8K-chunk comparison. They are also single transfer-timed observations:

| Input tokens | Original seconds / tokens/s | Layer-major seconds / tokens/s |
| ---: | ---: | ---: |
| 32,768 | 79.509 / 412.13 | 74.477 / 439.98 |
| 65,536 | 157.089 / 417.19 | 157.157 / 417.01 |

The layer-major runs keep the first 32K residual rows in VRAM. At 32K,
expert bytes fall from 190.110 to 50.292 GB. At 64K, combined weight and
host residual transfers fall from 382.987 to 176.457 GB, but wall time
does not improve. Reducing PCIe traffic alone does not establish an
overall speed gain. Sampled process VRAM reaches about 9.853 GiB at 32K;
the original scratch uses 3,353,103,360 bytes and compact HC scratch
uses 1,895,413,760 bytes at this chunk size. RAM and VRAM samples are in
the [memory record](8k-chunk-long/memory-summary.json).

## Standalone RAM/VRAM bandwidth

The [standalone copy probe](pcie-bandwidth/README.md) completes 35 cases,
five rounds each, with complete final-copy byte checks. At native expert
sizes, CPU staging plus H2D reaches 6.125–6.192 GB/s; direct host-USM H2D
reaches 6.288–6.365 GB/s and 6.466 GB/s at 1 GiB. This supports the earlier
in-engine rate of about 6.2 GB/s. DMA overlaps other prompt work, so the
copy rate alone does not establish the critical-path transfer wait.

## Phase intervals and the remaining SSD wait

The [32K profiled record](gpu-residual-long/gpu32768-k8v8-32768-c4096-profile/run.json)
emits one aggregated layer-major report. Its phase markers increase wall
time to 97.711 seconds, versus 86.299 seconds in the separate transfer
run. It is a bottleneck diagnostic. GPU-timeline intervals include host
submission gaps and waits between event markers; they are not isolated
kernel execution times.

The record attributes about 26.97 seconds to PLE read waiting, 17.37
seconds to dequantization intervals, 9.42 seconds to attention intervals
and 8.27 seconds to layer loads. PLE gathers run once, at their target
layer, for each input chunk. They are not repeated at all 48 layers.
Removing repeated expert uploads does not remove this SSD work.

The [grouped oneMKL probe](grouped-gemm-rejected/run.json) tests the same
FP16 expert shapes with variable row counts. Two-expert down groups keep
all output bits but show little timing benefit. The first four-expert
down group changes 14,891 of 28,160 output bits, with maximum absolute
error 0.00000762939453. The probe stops at that failure. Grouped GEMM is
not integrated into the engine.

## Actual allocation at a 256K context

The [capacity record](gpu-residual-long/capacity-summary.json) requests
262,146 K8/V8 positions with all K/V pages resident, dense model weights,
the normal decode cache, compact prefill scratch and the actual full
512-expert layer cache. Chunks 4,096 and 2,048 fail to allocate the layer
cache; their logs are preserved. Chunk 1,024 succeeds, loads all 48
layers and produces a complete finite first head. Its compact scratch is
490,183,168 bytes. Sampled process VRAM reaches 9.867 GiB and RAM reaches
47.315 GiB, with no observed process swap.

This run processes two input tokens to isolate actual allocation. It
does not establish full 256K prompt throughput or full-prefix numerical
parity. The [256K fixture manifest](256k-fixture/fixture.json) records a
262,145-token prefix of the same extended source stream as the
[64K fixture](long-fixture/fixture.json), preserving its complete prefix.

## Production PLE table in RAM

The [RAM PLE proof](ram-ple-proof/README.md) exercises the upstream
`--ple-io ram` option, retaining every output, residual and state byte at
8,087 tokens/chunk 4,096. The table is loaded but not locked under this
session's memlock limit. Its startup load takes 42.3 seconds and sampled
engine RAM reaches 74.104 GiB without process swap. Long timing comparisons
remain pending. The profiling helper now accepts the production RAM mode
and records its actual locking outcome and startup load time separately.

The [32K RAM diagnostic with the embedding server resident](ram-ple-shared-gpu/README.md)
uses 4K chunks and a 4K GPU residual prefix to fit the current free VRAM.
Both complete finite heads and IDs match the original 4K control. One
normal-wall run takes 79.063 seconds (414.46 tokens/s), with a warm
1.5-second table startup separately recorded. The profile's PLE waiting
is about 20 ms, while host residual DMA takes 17.460 seconds and the
dequantization phase interval is 17.239 seconds. This condition differs
from the earlier 8K-chunk, 32K-GPU-prefix runs and is not a matched speed
comparison. All sampled process swap remains zero.
