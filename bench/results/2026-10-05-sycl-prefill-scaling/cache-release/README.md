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
when rerun with its required GGUF fixture path (`tests`). Server graph recreation
and long-context capacity are under validation. `build-manifest.json` identifies the frozen binary and source.
Results will be added after finite controllers finish; there is no measured
winning cache policy yet.
