# MMQ public entry points and scratch lifetime

The `strata_sycl_mmq` component library implements the original
`strata::prefill::mmq` symbols from `include/strata/prefill/moe_mmq.hpp`, using the
ported quantizer, products and surrounding stages. The original header and
upstream host sources remain unchanged. The library is currently built by the
isolated validation project, not linked into a complete SYCL engine.

## Connection and device admission

The component's opaque stream argument is a non-null pointer to an in-order
`sycl::queue`. `mmq_adapter.hpp` declares that contract. A queue copy compares
as the same queue; the address of its C++ wrapper is not its identity. Empty
operations preserve the original early returns before looking up a stream.
Invalid gather groups preserve the original false/no-launch contract, including
range and alignment rejection before stream lookup.

`built()` reports that this component is compiled. `supported()` admits exactly
the nine default weight formats; optional Q5_0/Q4_K/Q5_K/Q5_1 products are still
unimplemented. `fits()` checks every visible GPU, as upstream does. For this
Intel tile it requires:

- Intel GPU, matrix and FP16 support, and device USM allocations;
- the 128-thread workgroup, 16/32-wide subgroups used by the product/quantizer,
  and the relevant work-item limits;
- at least 16,128 bytes of local memory for the product's explicit local arrays;
- an advertised signed-int8 × signed-int8 → int32 matrix shape of 8×16×32;
- for AOT, the device architecture selected when building this library.

This preserves the purpose of `fits` (select fallback when the available tile
cannot run), not CUDA's architecture-specific tile search. Its positive result
is a resource/capability check; it does not prove every driver/device combination
works. Only B570, Level Zero, JIT and `bmg_g21` AOT have been executed here.

The adapter checks 64-bit `Product` dimensions before converting them to the
component's int32 geometry. Generic `matrix_bytes()` links the pinned GGML base
library's actual type table, including types which MMQ cannot multiply. It does
not replace that table with the nine-format product subset. Invalid dimensions
and allocation overflow throw; the valid-input byte formula is upstream's.

## Scratch and ordering

The source reference is Strata's `CachingPool` in `ggml_cuda_host.cu`, combined
with GGML's `ggml_cuda_pool_alloc` and `launch_mul_mat_q`. Upstream retains scratch
allocations after their host allocation handles leave scope and reuses them.

Each context here retains a queue copy and a device scratch allocation per queue.
The current planner needs either no scratch or
`compute_units * 64 * 16 * sizeof(float)` bytes; B570 therefore retains 655,360
bytes for each queue which needs fixup. A fixed queue/device does not grow this
allocation. Capability checks and compute-unit queries run once when admitting
a queue, rather than at every product. The normal `run()` path performs no host
wait, device query or new scratch allocation after the first use.

In-order queue execution protects reuse between successive products. A context
mutex protects the host submission of each main/fixup pair so two host callers
cannot interleave pairs using the same scratch. Different queues have separate
scratch; their device work has no dependency introduced by this adapter.

Context teardown waits for each retained queue before freeing scratch. This is
needed because SYCL `free` has no CUDA `cudaFree` implicit synchronization. It also
keeps the underlying queue alive if its caller-side wrapper has already gone out
of scope. Teardown reports an asynchronous error and terminates rather than
throwing from a destructor. Callers can observe errors sooner with their queue's
`wait_and_throw`. Caller-owned weights, activations, IDs, bounds and outputs must
remain valid until their queued work completes.

## Evidence

Intel Arc B570, Linux, oneAPI 2026.1.1, Level Zero, 2026-10-04:

- The previous ten connected stage fixtures also pass through the public API.
  They cover all nine default native formats plus the actual Strata Q2 packed
  geometry: 1,410,304 byte comparisons and 135,046 numerical comparisons.
  The test facade alone adds completion barriers for event-based assertions;
  production entry points do not. The same independent references and numerical
  limits described in [mmq-stages.md](mmq-stages.md) apply.
- The context test validates the nine-format admission set and 30 generic matrix
  size cases, including non-MMQ F32/F16/BF16, Q4_0/Q4_K/Q5_K/Q5_1 and IQ1_M.
- Twenty-seven products with distinct input scales check 2,565 outputs exactly,
  plus output row guards. They exercise two queues, a copied queue wrapper,
  repeated scratch reuse, twelve submissions from two host threads sharing a
  queue, context destruction with queued work, and a queue wrapper destroyed
  before the context.
- A host task intentionally blocks preceding work on a warmed queue. `run()` must
  return while that task remains blocked (two-second observation bound). The test
  always releases the task before cleanup. This detects an accidental host wait;
  it is not a throughput or submission-latency benchmark.
- Null and unordered streams, unsupported products and overflowing dimensions
  are rejected. Empty operations and invalid gather groups return before stream
  lookup, matching upstream's no-work paths.

The exact-output fixture uses powers-of-two input scales. An initial fixture
incorrectly assumed any integer scale would survive the original `1 / (127 /
amax)` expression exactly: scale 7 did not, producing a 0.0000610352 output
difference. The fixture was corrected, retaining the upstream quantizer's
expression. This is not evidence for changing that expression to `amax / 127`.

Source/binary hashes and JIT/AOT logs are in
[mmq-context-results.json](mmq-context-results.json). The existing quantizer,
product/fixup-order and direct-stage checks continue to pass.

## Remaining integration

There is no implicit default/null queue resolver yet. CUDA default-stream ordering,
events across streams, capture/replay, allocator/arena ownership and full runtime
error behavior must be ported before routing the unchanged engine through these
symbols. The AOT enum mapping currently targets Intel GPU architecture names.
Optional MMQ formats, non-MMQ fallback implementations and other inference
families remain open. These component checks are not CUDA device parity,
full-model state/logit agreement or a PP/TG speed result.
