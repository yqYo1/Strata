# SYCL usage and failure audit, 2026-10-06

This records confirmed source-level defects and their fixes. It does not
establish the cause of the first xe fault or prove that the modified engine
is correct on a healthy GPU. The independent stock 64 KiB probe lost the
device during H2D before any integer kernel; see
[the recorded control](../health-stock-after-loss-v2-no-copy.json).
That failed-state control cannot identify which earlier workload triggered
the original loss. The user declined saving that later dump.

## Confirmed defects addressed

- The printing-only DPCT async handler consumed failures, allowing callers
  to continue after `wait_and_throw`. It now logs all errors and propagates
  the first. Serve shutdown returns a failure status when queue completion
  fails instead of swallowing it and exiting successfully.
- DPCT's error macro discarded the boolean from `ext_oneapi_empty`. Two
  legacy handshake loops now distinguish a completed queue from pending work.
- GPU fills followed by volatile CPU polls did not provide the required
  synchronization for Stager/PLE reuse. In-order queue host tasks now publish
  CPU atomic acknowledgements; shared ownership retains their lifetime.
  The prefill queue is checked for in-order execution.
- Concurrent mapped handshakes now pair CPU `std::atomic_ref` operations
  with system-scope acquire/release device atomics, checking the required
  host-USM atomic/order/scope capabilities before legacy graph capture.
  Monotonic flag publication preserves a cancellation sentinel. Deferred
  copy callbacks capture immutable flag/value pairs.
- The default verifier keeps the GPU ring in device USM and updates its
  CPU diagnostic sequence after event completion. Shared trace slots use
  a mutex and progress snapshots use CPU atomics. Migrated GPU timestamps
  were always zero; their concurrent mapped writes are disabled.
- SYCL USM allocation can return null without throwing. The migrated
  exception-only checks now reject null before dependent operations, using
  the existing error or pinned/pageable fallback path. There are 322 checked
  allocation results across engine and validation sources.
- Virtual reservation size is checked against the context granularity,
  separately from physical device granularity. ExpertCache cleanup preserves
  the primary allocation/map error and its destructor logs cleanup failures.

The relevant contracts are in the [SYCL 2020 specification](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html)
(USM, host tasks, async errors and section 4.15.5 host atomics) and Intel's
[virtual memory extension](https://raw.githubusercontent.com/intel/llvm/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_virtual_mem.asciidoc).

## Actual validation

`all-build.json` identifies the source snapshot. Ninja compiled all 45
configured modified C++ translation units and linked the engine successfully.
Four additional sources passed manual syntax checks. One unused legacy
`ggml_cuda_host.dp.cpp` failed because `common.cuh` is unavailable; the HEAD
control fails the same way and that source is not in the configured engine.
This is a recorded limitation, not an additional passing translation unit.
The factor-0 engine SHA-256 is
`1e37f0a910828046d9ad4701f44428e98a78f0056841a9d6e36376ca0d03347e`.
No modified engine was executed on the GPU.

The CPU-only helper test passes ASan/UBSan: 4,096 payload handshakes,
concurrent monotonic publication preserving the cancellation sentinel,
2,048 two-slot reuse checks, deferred callback lifetime, null allocation
rejection, async error propagation and shutdown status. It simulates a
queue's deferred tasks; it does not validate a real SYCL scheduler or GPU.
`host-completion-validation.json` and the output logs preserve that scope.
CPU native-pool exact-output/performance/sanitizer results are recorded
[separately](../../cpu-pool-scheduling/README.md).

The recorded root recovery attempt at 08:08 UTC was refused before any
reset because Xorg PID 4330 held the B570 open. Disconnected display
connectors do not imply there are no GPU clients. See
`recovery-refused-xorg.json`; recovery remains unverified.

## Remaining investigation and validation

Stager and PLE polls still need a cancellation/error-aware failure path when
a DMA never completes. Legacy bounded GPU waits may fall through after their
spin limit without proving that host payload is published. Raw diagnostic
verifier pointers need a lifetime audit against concurrent destruction.
Error handling during graph/memory restoration and partial unmap failures
needs further review. These fixes do not constitute an exhaustive UB proof.

On a recovered GPU, validate with fresh kernel-log boundaries and an
independent stock probe before Strata: source/adapter validation, exact
whole-head/residual/state comparisons, cancellation and checkpoint reuse,
full 262,144-context capacity including normal MTP, and paired prefill/TG
measurements. CPU tests and compilation cannot replace those checks.
