# Strata synchronization and initialization audit, 2026-10-06

This audit finds reproducible defects in the project and closes their entry
points. It does not attribute the first machine-wide xe stall to any one of
them. Failures of other programs after that stall do not clear Strata: those
programs inherited an already unhealthy device. The earlier definite borrowed
queue UAF and actual engine crash remain recorded in
[queue ownership](../queue-ownership/README.md).

After the successful short default checks below, the same frozen engine's
forced layer-major draft-weight lease crashes during its repeat request.
Strata's CPU SIGSEGV in `GraphicsAllocation::prepareHostPtrForResidency` precedes
a BCS fault by 4.288931 s. The
[actual failure and exact symbolization](draft-lease-runtime-crash/README.md)
are retained. The
[graph-retirement candidate](graph-retirement-candidate/README.md) has build
and CPU validation only. GPU testing stopped after the fault; the lease and
machine-wide recovery are not marked healthy.

The [October 7 residency audit](residency-lifetime/README.md) traces tagged NEO
raw residency references through physical destruction and replay. Four CPU
mechanism cases support retiring and recreating lists, without identifying the
actual crashing object. A separate extracted MTP prefill control finds a local
host upload source freed on capture failure; seven candidate cases pass after
completing that upload and propagating asynchronous errors. The interactive
SYCL cache-resize feature is rejected before device selection. The updated
engine builds and passes actual preflight. Its first small health prerequisite
fails on that same boot with internal migration queue timeouts and 37 recorded
resets, so no model comparison or recovery was attempted on that boot.

After an external reboot, the [current-candidate model comparisons](post-reboot-retirement/README.md)
complete all 16 requests and ordinary process exits without new xe faults,
with direct submission disabled. Retained weights pass full-head equality;
releasing weights and recreating graphs fails repeat-result equality even with
prompt/conversation reuse disabled. The lease remains experimental and off by
default; GPU execution success is distinct from mathematical validation.

The GPU is the same Arc B570 10 GiB, Ryzen 5 5600X, 128 GiB RAM, kernel
7.0.0-38-generic, stock NEO 26.31.39395.14 and oneAPI 2026.1.1. Boot ID is
`ef28c8b7-46f0-4806-9326-36e3158bceb7` for the earlier tests below. Those positive GPU tests use
persistent SYCL cache off, Level Zero V2, default direct submission and default
copy offload. No reset, rebind, display shutdown, kernel update or firmware
flash is performed during these tests.

## Retired waits did not establish readiness

Three production kernel fragments returned after 20,000 polls while the flag
was still unsatisfied: `wait_flag_ge`, `wait_flag_ge_or` and `doorbell_wait`.
There was no failure state propagated to every following graph node. An
acquire fence after an unsuccessful poll does not publish the CPU's plan or
expert rows. The following plan/copy consumers could therefore read while the
CPU was still writing, or use stale plan pointers. The old `_or` wait also
checked its skip flag only before entering its loop.

The [CPU control](shutdown-spin/record.json) extracts the real old kernel
fragments from commit `60fc342` and proves all three unsatisfied returns.
It uses CPU atomic/fence stand-ins and does not submit a bad graph to the GPU.
The candidate API functions reject all three legacy wait calls before touching
a queue. Verification rejects `STRATA_VERIFY_NO_HOST` and
`STRATA_SYCL_HOST_BOUNDARY=0`; default host completion boundaries remain in use.
The monolithic token graph is disabled and its caller falls back to the normal
per-layer graphs. Current native-pack decode did not enter that monolithic
capture path, even though its resident-hit table log said "token graph hit
path". This defect is not established as the first B570 stall's active path.

The value-publishing doorbell also used a plain volatile store while the host
reader used `std::atomic_ref`. It now uses the existing system-scope release
store, and its work-group barrier includes global memory before thread 0
publishes the payload. The normal incrementing publisher already included a
global-memory barrier. The value publisher's verifier caller is currently
HIP-conditional, so this repair is not attributed to default B570 execution.
SYCL's [host atomic rules](https://github.khronos.org/SYCL_Reference/iface/interaction-with-host-code.html)
specify interoperability through the same atomic object; a volatile store is
not that atomic operation. Its
[forward-progress rules](https://github.khronos.org/SYCL_Reference/iface/atomic_ref.html)
also do not guarantee general blocking work-item coordination will progress.

## Completion before CLI reclamation

The CLI success footer lacked an explicit completion check before freeing its
raw arena and buffers. The serve drain covered only the current device, even
though another device's copy queue can still use primary allocations. CLI,
serve and the GPU-stage success path now check every device's queues before
raw reclamation. An error returns nonzero through the existing immediate-exit
boundary and skips unsafe frees.

Actual old CLI footer fragments with deferred primary and secondary queue
consumers reproduce two ASan heap-use-after-free controls. Both candidate
consumers complete first; an injected secondary failure exits 1 with no frees.
These are CPU queue stand-ins demonstrating the missing ordering guarantee,
not a claim that the old real CLI always had pending consumers at exit.
SYCL [USM deallocation](https://github.khronos.org/SYCL_Reference/iface/usm_allocations.html)
does not wait automatically, and freeing storage still used by queued work is
undefined behavior.

## Device creation before main

With an intentionally nonexistent GPU selector, the first actual candidate
aborted before its configuration guard. GDB identifies
`dpct::blas::descriptor::_saved_queue_ptr` selecting the device in static
initialization: [first stack](preflight-invalid-selector-gdb.txt).
Making that queue lazy exposed a second independent pre-main constructor in
`native_mmvq.dp.cpp`: [second stack](preflight-invalid-selector-lazy-gdb.txt).
The retained IQ4 and S2 codebooks had been heap-allocated in global initializers
to avoid late runtime destruction; that still initialized the device manager
before main.

The saved BLAS queue now selects its default only when requested, propagating
selection exceptions. Codebook constructors are also lazy, retaining their
storage across captured graph calls. The CPU descriptor/table controls prove
old pre-main construction and candidate initialization only on use. The final
real executable now rejects all three tested legacy selectors with exit 2 and
prints help with exit 0 even when no requested GPU exists:
[actual preflight](host-boundaries-preflight-final.json). The shell launcher
rejects legacy settings before any Docker command. This prevents a help or
invalid-mode request from entering an unhealthy driver; it does not make an
unhealthy driver recover by itself.

## Codebook object type and alignment

The IQ4 helper read an `int8_t` codebook through a `uint32_t*`, without a
word-alignment contract. It now copies 16 bytes into four real word objects;
the byte permutations, table values, integer dot products and floating-point
operation order remain the same. The production lookup and actual DPCT byte
permutation are extracted into the
[CPU checks](device-tables/record.json). The old aligned control matches
524,288 reference bytes; its unaligned control reports a UBSan misaligned
load. The candidate matches 2,097,152 reference bytes across four byte offsets.
This does not establish that the historical GPU codebook was misaligned.

Migration helpers preserve the disabled waits, atomic publisher, lazy BLAS
queue and lazy codebooks. Wait/table helper transformations are checked on
old production sources and are idempotent. Complete SYCLomatic regeneration
and review of all other hand fixes remain separate work.

## Real model checks

The engine used for these successful short checks has SHA-256
`9a2da2252d50b0a594f891490816fbf8f2f6bb385a972c0bbf599fdc07cbd74f`.
[Normal MTP](model-final/record.json) completes four 37-token prompt requests,
including repeated and restored conversation checkpoints, with normal exit 0.
All 16 generated IDs and printed logprobs match the retained normal-MTP
report. No new xe fault appears. This checks displayed probabilities, not
every internal state byte or every full-vocabulary logit.

An intermediate real candidate also passed those four requests:
[intermediate record](model-intermediate/record.json). Its missing-device
preflight still exposed the global initialization bugs above; the successful
short inference did not establish safe initialization.

The first additional CLI harness expected `--dump-logits` to write data in
native MTP decode. That existing path does not write this file; the actual CLI
completed four tokens, ordinary exit and queue drain without a new xe fault.
The [failed harness receipt](cli-empty-dump-control/record.json) is retained.
The [corrected harness](cli-and-storage/record.json) uses
`STRATA_DUMP_FIRST_LOGITS` to read the entire first verify head. The CLI generates
the same four IDs, exits 0 with the completion message and checks all 248,320
head floats finite. Its head SHA-256 is
`3d6a3bff1c62225d7de5bcc9895dad206c3eb686d856cb566039b5ea690943b8`.
It is not described as a final-token head measurement or byte equality against
a retained complete head.

## Actual memory mapping and retained graph replay

The same corrected controller next runs the real ExpertCache storage probe:
[nine rounds](cli-and-storage/cache-lease.stdout). Its payloads are 3,145,748,
707,788,800 and 223,227,900 bytes, each released and restored three times.
Every round checks physical release, unchanged virtual address, all restored
payload bytes and output from the previously captured graph. All nine pass
and ordinary exit is 0. Neither the CLI nor storage run adds an xe fault.
The probe SHA-256 is
`1def5a1400f1214382b758317e14ca7c63a6dad3339f51771ef6d866fff6bacd`;
its current library build/link receipt is in [storage-build](storage-build/record-final.json).

These checks follow the
[virtual-memory extension](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_virtual_mem.asciidoc)
contract: the device advertises support, reservations/mappings use their
respective granularities and one context, each complete mapped range is
unmapped separately, and physical storage is destroyed only after unmapping.
Queue consumers complete before unmapping; a graph is submitted again only
after all its addresses are backed and restored. A mapped accessible pointer
can be used where device USM is accepted. Passing this short storage test does
not prove model state, cancellation cleanup or full-context capacity.

The first model-level draft-lease smoke left default short-prompt selection
active. Its four requests completed and matched the baseline but no prefill
lease ran, so its lease gate is correctly false:
[unreached-path receipt](draft-lease-unreached/record.json). A second harness
confused its own context-checker's `--serve-chunk` option with an engine option;
the actual engine rejected it before startup, with no new fault:
[bad-option receipt](draft-lease-bad-option/record.json). The subsequent smoke
uses the engine's `--short-read 0` to force compact layer-major prefill. It
completes its first request and three physical release/restore pairs, then
crashes on the repeat as recorded above. It is not a passing lease test.

The remaining gates are normal MTP memory-lease/cancellation/state checks,
actual 262,144 context consumption, repeated long runs and performance. Short
successes do not prove the first full-device stall's cause or a zero fault rate.
The direct-submission semaphore association remains a teardown candidate in
[the runtime audit](../direct-submission/README.md). The newer failure supplies
the missing observation that a CPU fault can precede the repeated BCS address;
it does not identify the invalid residency allocation or the original global
stall's exact trigger.
