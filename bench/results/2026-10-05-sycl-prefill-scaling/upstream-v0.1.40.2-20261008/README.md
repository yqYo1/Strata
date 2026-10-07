# Upstream v0.1.40.2 and the B570 SYCL port, 2026-10-08

The integration targets upstream tag `v0.1.40.2`, commit
`e8ca9afd03d839d4f8dbbe82dffce7f8a3bafd7a` (242 commits after the previously
integrated `82f46a8c`). The fork's starting head is `186409e8`.
The unmodified upstream tag is built in an independent worktree. Both Strata
and the shared ggml checkout are clean; ggml is `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
The upstream and integration binaries still print `0.1.40-sycl`, as upstream's
SYCL CMake version does; the commit and executable hashes identify this release.

Hardware: Intel Arc B570 10 GiB, Ryzen 5 5600X, 128 GiB RAM. Kernel
7.0.0-38-generic, NEO 26.31.39395.14, oneAPI 2026.1.1, Level Zero V2.
Direct submission and persistent SYCL cache are disabled. Copy offload is
explicitly disabled, as in the preceding lifetime controls. Embedding remains
stopped and disabled. No reset, rebind, reboot, driver or service change was made.

## Short-input functional preflight

Model: Qwen3.8-Flash-Next IQ3_S, the local native pack and original two GGUF
shards. Eight separate processes cover unmodified upstream defaults, integration
defaults and prior tuning configurations. They use a 37-token prompt, up to64
greedy output tokens, context128, chunk32, normal MTP4 and five CPU workers.
All produce 64 finite-logprob tokens and exit normally without a new xe fault.

These short inputs are functional preflight only. Per-layer loading, JIT and
graph preparation dominate their request times. The raw timing records remain
in `results.csv` and `comparison.json` for provenance; they are excluded from
throughput comparisons and tuning decisions. Meaningful comparisons require
at least32,768 input tokens and the same prompt, chunk, context and KV settings.
The first32K logged checks and subsequent separate clean timings are underway.

All short arms request `--expert-cache 600 --expert-cache-per-layer`. Upstream
expands that budget into768 mixed-size slots (1498 MiB); the integration retains
600 explicit-count slots (1523 MiB). This changes expert placement, so even a
long-input comparison must report the actual counts and memory. Unmodified
upstream uses `--pcie-frac 0` because it warns against direct reads of pageable
experts on xe without --stream-experts. The local positive-fraction arm uses
its existing queue-copy path. Prior tuning flags are COMPACT=2, LAYER_MAJOR=2,
RELEASE_DRAFT=1, DRAFT_VERIFY=1 and EXPERT_WAIT_BATCH=0. These short checks do not
establish their benefit or cost for long prompts.

## Integration checks and defined accesses

The production build passes. Host checks pass: 562 server tests with eight
skips, 30 setup scripts, five standalone CPU tests, 168 CPU-pool bitwise
comparisons, Q8_K byte parity on 16 rows at four sizes, eight lifetime/queue
host test suites, and the new ASan/UBSan cache-copy bounds test. Fixtures were
updated for upstream's structures, arity and four-part version; the setup hash
fixture now also mocks upstream's added HTTP HEAD probe.

Three logged normal-MTP controls complete first/repeat/other/restored requests:
the bounded-cache build, the final float-storage build with prior tuning flags,
and final default prefill with API logging alone. All twelve requests match
the historical control's token IDs, protocol logprobs and every one of 248,320
first-head floats. The unmodified upstream completes four diagnostic requests
with finite logprobs and matching IDs. Its serve path does not implement the
whole-head dump, so whole-head parity is not claimed for that arm.

The merge keeps host completion boundaries, atomic publication, pageable-arena
queue copies, queue/storage ownership and graph retirement guards. Unsupported
shared queue forks, fused one-token history commit and unvalidated batched
pipeline modes are rejected or kept off. Explicit per-layer cache counts keep
the existing placement; auto sizing retains upstream's layer-size packing.
Copies validate the actual backed slot capacity before forming a device pointer
or enqueueing. New MoE float4 arithmetic reads/writes the underlying float
array through float pointers, retaining its ten ordered FMAs and shared addition.
53 reviewed independent launches use ordinary work groups. Persistent/global
coordination kernels were not stripped blindly. The allocation alias check now
refuses an unverified eighth retry and checks its result allocation for null;
its sparse word check is not a proof of every byte in an allocation.

## Failures retained

- The first merge candidate enters an unsupported fused GDN-history commit.
  SYCL now retains separate commit graphs and rejects an explicit request for
  that unimplemented path before GPU initialization.
- Three subsequent first-request comparisons have matching IDs and different
  logits. They still expand the cache to768; disabling the new decode batch,
  CPU quantizer or K/V append paths did not establish matched placement. Once
  the same 600 placements are restored, complete heads match. Those earlier
  comparisons are not evidence of an arithmetic regression.
- An incomplete local explicit-count fix falls through into profile-order
  sizes with per-layer admission: 770 slots, not600. A 2,662,400-byte copy at
  offset1,594,547,200 exceeds its 1,596,723,200-byte arena by486,400 bytes.
  The driver classifies the out-of-range destination as host memory and CPU
  SIGSEGV occurs in NEO staging memcpy. The corrected branch and the new
  per-slot bounds checks prevent this invalid enqueue. The CPU stack is saved.
- The first unmodified diagnostic controller requests an unsupported serve
  logits file after a successful four-token answer and raises FileNotFoundError.
  The protocol-only retry passes four requests. This is a controller failure.
- Final default prefill with extra `STRATA_PREFILL_SYNC=1` waits stops at layer46
  of token0's chunk, last completed mark20330/dequant. Its 60-second watchdog
  raises SIGABRT; owned cleanup removes both processes. API logging without
  those extra waits completes all four exact comparisons. This narrows the
  conditions of that process wait but does not identify its resource or prove
  the additional waits are its root cause. The verify shutdown's different
  queues finishing in0ms do not prove prefill completed.

Fresh logged H2D/kernel/D2H checks after the failed jobs pass on the same boot.
No new xe fault is recorded in these integration/model/timing jobs. This does
not establish that long requests or every optimization avoid a GPU fault.

## Evidence and limits

`comparison.json` and `results.csv` retain all eight short preflight rows,
excluded from performance comparisons. `checks/`
contains executed request records and raw protocol; `pure-build/` proves the
unmodified source/build; `integration/` contains test and source/binary receipts;
`failures/` contains the two CPU/watchdog snapshots; `controllers/` contains
executed controller versions. Diagnostic multi-hundred-MiB API logs and head
binaries remain in the private state directory; their paths and hashes in the
records identify them. The earliest compiler log was overwritten during the
first resolution attempt; `integration/first-build-log-limitation.txt` records
this loss rather than claiming a complete initial compiler archive.

This tag has not completed the full262144-cell normal-MTP serving, repeated
restoration, clipped tail, refusal and later-valid-request gates. The earlier
2048/long-prefill waits with zero cooperative flags remain unresolved. These
short checks do not establish PP1000/TG70, the maximum context or a driver fix.
