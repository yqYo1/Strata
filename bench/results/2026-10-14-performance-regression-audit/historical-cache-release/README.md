# Decode expert-cache release during layer-major prefill

Machine: Intel Arc B570, 10 GiB VRAM; Ryzen 5 5600X; 128 GiB RAM.
Model: native IQ3_S Qwen3.8 Flash Next. oneAPI 2026.1, precise math,
compute runtime 26.35.39758.10. The embedding service stays stopped.

This experiment lends the decode **expert weight cache** to layer-major
prefill, then restores it before decode. The residency table and slot layout
stay in CPU memory. KV and recurrent state are not part of this weight cache.
No setting below is enabled by default.

```
STRATA_PREFILL_COMPACT=2
STRATA_PREFILL_LAYER_MAJOR=1
STRATA_PREFILL_LAYER_MAJOR_R_GPU=65536 # request this many residual rows in VRAM
STRATA_PREFILL_RELEASE_CACHE=1
STRATA_PREFILL_CACHE_ALLOC=vmm       # vmm or rebuild
STRATA_PREFILL_CACHE_RESTORE=ram     # ram or snapshot
STRATA_PREFILL_CACHE_RELEASE_FRAC=1  # 0..1; rebuild requires 1
```

Use `--no-prefill-borrow`, one GPU, and no helper expert caches. RAM restoration
requires the complete resident expert arena, so a file-tier configuration
cannot silently add disk reads to the RAM comparison.

* `vmm` releases physical segments while retaining the virtual address and the
  existing graphs. Segment size is 64 MiB; retained bytes round up. On this B570
  the last segment also needs padding: the runtime reports 64 KiB minimum
  granularity, but a 5 MiB physical allocation fails while 6 MiB and 64 MiB
  allocations succeed. The physical-size checks preserve these observations.
* `rebuild` frees and reallocates ordinary device memory. The server refreshes
  cache pointers and recreates every Verifier window graph. The CLI constructs
  its Verifier after prefill, so CLI restoration timings do not measure
  recreation of an existing decode graph.
* `ram` restores occupied slot payloads directly from immutable resident RAM
  blobs. `snapshot` downloads the released tail into a new RAM buffer before
  releasing VRAM and uploads it after prefill.

The enclosing prefill timer includes suspension, restoration and graph
recreation. The lifecycle line reports physical bytes released, logical bytes
restored, source, suspension/restoration times, graph time (included in restore),
and whether the cache address stayed the same.

## Evidence and scope

`vm-probe` contains 15 standalone checks: 64/320/1536 MiB, five rounds each.
Every restored byte and a captured XOR graph replay matched. Median complete
snapshot/release/map/restore times were about 22/109/631 ms respectively.
This is not a benchmark of actual model graph recreation.

`capacity-cache1` preserves an invalid one-slot/per-layer configuration, which
fails before prefill; it is not a VRAM capacity result. `capacity-cache48`
preserves a 64K/4K/all-GPU-residual allocation failure with 48 cache slots.
`initial-allocation-failure` and `allocation-diagnosis` preserve the failed
unpadded segmented implementation.

`assembled-ram-proof` has eight finite 257-token/128-chunk model runs. Full
248,320-float finite heads, every FP32 residual row, and all dumped persistent
state bytes match the original same-cache implementation. Reallocation also
passed with a changed address. Its RAM method first assembled a second RAM
image; it is retained as an intermediate implementation, not the current RAM
path. A concurrent build occurred during these diagnostic runs; their single
timings are not performance medians.

The current direct-RAM implementation also passed the same eight full-head,
all-residual-row and persistent-state comparisons (`direct-ram-proof`). All 30
registered GPU tests passed: the first run passed 29, and the PLE test passed
when rerun with its required GGUF fixture path (`tests`). `build-manifest.json`
identifies the frozen binary and source.

## Actual server graph recreation

Six policies, four requests each, passed normal MTP generation, repeated
requests and checkpoint restoration (`serve`). Each policy produced the same
complete finite first head, all generated IDs and printed logprobs for all
requests, and the same final dumped persistent state bytes.

Each process has two fresh 1,024-token prefill segments, with 128-token chunks
and a 128-slot expert cache. The second fresh segment follows decode, so the
rebuild case really replaces graphs that have already executed. The table is
that **single second-segment observation**, not a repeated-process median.
It includes preparation, release, allocation, weight restoration and any graph
recreation. It does not include the rest of prefill's compute time.

| Cache policy | Physical VRAM released | Lifecycle time |
| --- | ---: | ---: |
| Full snapshot to RAM, fixed address | 384 MiB | 278.334 ms |
| Full direct RAM restore, fixed address | 384 MiB | 53.514 ms |
| Half physical release, snapshot to RAM | 192 MiB | 113.237 ms |
| Half physical release, direct RAM restore | 192 MiB | 24.676 ms |
| Ordinary reallocation, direct RAM restore, recreate graphs | 325 MiB | 153.111 ms |

The full logical slot-storage extent is 325 MiB. Fixed-address physical storage is padded
to 384 MiB. Consequently half physical release restores 133 MiB of logical
slot storage, not half of 325 MiB. Graph recreation took 102.940 ms in the second
fresh segment, included in 153.111 ms. First-time graph creation took 588.045 ms
in this process; do not treat it as the warm recreation cost.

Here, ordinary RAM restoration plus graph recreation beats snapshotting to RAM
at a fixed address. Keeping the address and restoring directly from the existing
RAM arena has the lowest full-release lifecycle time in these observations.
These costs alone do not decide which cache policy gives the fastest prefill.
## 64K capacity and prefill observations

`long-context` records nine successful or refused configurations on the same
frozen JIT binary. Each has 65,536 input tokens, 4K chunks, a fixed 65,538-position
K8/V8 context, and 128 decode cache slots. Transfer diagnostics are enabled.
These are single observations, not repeated timing medians. The CLI initializes
its Verifier after prefill; the graph recreation costs above come from the
separate server runs.

| Decode cache policy | Residual rows in VRAM | RAM residual traffic | Prefill seconds | token/s |
| --- | ---: | ---: | ---: | ---: |
| Retained, previous 32K-prefix setting | 32,768 | 126.165 GB | 140.996 | 464.81 |
| Retained, largest fitting 4K-step prefix | 57,344 | 31.541 GB | 126.097 | 519.73 |
| Half physical release, direct RAM restore | 61,440 | 15.771 GB | 123.065 | 532.53 |
| Full physical release, direct RAM restore | 65,536 | 0 | 121.418 | 539.76 |
| Full physical release, snapshot restore | 65,536 | 0 | 120.848 | 542.30 |
| Ordinary full release and RAM restore | 65,536 | 0 | 120.887 | 542.13 |

Retaining the cache with 60K or 64K residual rows, and half release with 64K
rows, failed the free-VRAM capacity check. Every successful run has the same
complete finite head and generated IDs. Expert-weight traffic remains 50.292 GB
in each run. RAM residual DMA time falls from about 5.0 seconds for the largest
retained-cache prefix to zero with full release. Sampled peak VRAM is below
10 GiB and process swap is zero; 1 Hz samples can miss short peaks.

Full release reduces the observed time by 3.7% versus the largest fitting
retained-cache prefix, rather than the larger improvement against the old 32K
prefix. The three full-release timings differ by less than 0.5%; they do not
establish a prefill-speed winner among restoration mechanisms.

## Adaptive cache correctness

RAM restoration must use the driver's current residency table. Adaptive swaps
update that table rather than the cache object's original admission map.
The implementation now follows the current table. `STRATA_PREFILL_CACHE_VERIFY=1`
adds diagnostic downloads that compare every occupied released-tail payload
against RAM before release and after restoration; snapshot mode checks its
entire restored tail. These diagnostics add copies and RAM allocations, so
their timings must not be used for performance comparisons.

The timing binary above used `--adapt-swaps 0`. The corrected restoration passed
five policies and 20 normal-MTP requests of 16 generated tokens, with four
adaptive swaps every window (`adaptive-check`). Complete finite first heads,
all IDs/logprobs for all requests, and final dumped state bytes match the
retained-cache control. RAM restoration checks all occupied released-tail
weight bytes before and after copying, including requests with experts that
differ from the original admission map (56 changed entries in the full-RAM
case). The full cache occupies 325 MiB of logical slot storage; occupied expert
weights occupy 268.60 MiB. RAM restore copies those weights and clears unused
slot padding on-device; the snapshot method copies the whole slot extent.
