# Optional expanded-FP16 layer cache: CPU preparation only

The 2K unitrace run recorded 50,030 GU and 50,030 down dequant launches,
4.62848 seconds total, about 60% of its kernel duration. This private
prototype reuses unchanged FP16 expert outputs across chunks within one
layer-major request. It does not change quantization formulas, wrappers,
GEMM dimensions, activation operands, phase waits or decode callbacks.

STRATA_PREFILL_FP16_CACHE_EXPERTS defaults to zero. Each admitted expert
needs 9,830,400 bytes. The candidate limits admission by free VRAM minus
256 MiB and the device maximum allocation size. No available budget, a
single chunk, unsupported dimensions or failed allocation uses the original
scratch path. At the observed full-context free budget of about 51 MiB,
the arithmetic budget would select zero slots; GPU fallback is untested.

The allocation, dequant producers and GEMM consumers use the same in-order
queue. An expert is marked ready only after both producers are enqueued;
layer changes invalidate all tags. The owner waits before freeing USM with
the allocation queue/context, and frees it before main/MTP restoration.
On early return, the existing Restore guard waits and restores the old
cache pointer before the new owner is destroyed. If a wait fails, its
destructor retains memory instead of freeing potentially in-flight USM.
These are source-review invariants, not execution proofs. See the
[SYCL queue reference](https://github.khronos.org/SYCL_Reference/iface/queue.html)
and [USM lifetime reference](https://github.khronos.org/SYCL_Reference/iface/usm_allocations.html).

The [offline build](build/record.json) passed in 28.279 seconds. Exactly
one prefill archive member was replaced; other archive members and all
production link inputs remained byte-identical. The private binary SHA-256
is 9d9085d4…d658110. Production source and executable were not changed.
No GPU model run, output parity, clean speed measurement, repeated restore
or 256K fallback gate has run. The first GPU check must capture Level Zero/
UR logs and compare cache-zero and positive-capacity multi-chunk outputs.
