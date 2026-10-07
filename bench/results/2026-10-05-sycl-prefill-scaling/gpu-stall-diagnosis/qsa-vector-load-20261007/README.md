# Typed vector loads for FP16 and INT8 K/V helpers

The original eight-value FP16 helper read a `uint16_t` array through a
`sycl::uint4` pointer, then read that local vector through a `sycl::half2`
array pointer. The INT8 helper similarly read through `sycl::uint2` and a
pointer into its representation. The candidate uses typed `vec::load` for
both arrays and `vec::as<half8>` with four pairwise swizzles for FP16. The
four half-pair conversions and INT8 scale/multiply arithmetic remain the
same. These follow the
[SYCL vector member rules](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html)
and avoid the unrelated typed reads addressed by
[C++ type-accessibility rules](https://eel.is/c++draft/basic.lval#11).

On 2026-10-07, CPU and B570 fixtures each passed 6,291,456 comparisons with
zero mismatches. Both sweep all 65,536 half/scale bit patterns, both K and V,
rotated signed INT8 edge codes and every aligned eight-element row slice,
including the final slice. They compare original and candidate helpers
against an independent host IEEE binary16 reference. Non-NaN results must
match float bits exactly; NaNs must match classification. NaN payloads are
not checked. The GPU fixture uses a BDF-selected in-order queue, waits before
host buffers are reused and logs all Level Zero/UR calls and parameter checks.
It exited normally without forced cleanup or new xe faults.

The exact engine candidate also completed four real normal-MTP requests:
first, repeated, different and restored. All IDs, printed logprobs and every
complete finite 248,320-float head matched the released-weight control.
Six verified decode release/restore pairs completed, and both owned children
exited without force or new xe fault/reset messages. The case used context
128, chunk 32, compact mode 2, layer-major mode 2, five CPU workers and
phase-sync waits. Direct submission, persistent cache and V2 copy offload
were disabled. The diagnostic lasted 177.01 seconds; this is not throughput.

The private build replaced exactly one kernel archive member, kept the other
members and production link inputs unchanged, and linked a separate engine
`619c830b…`. Source hashes are `f81d6121…` before and `8158c41c…` after.
The subsequent normal production build reproduced the same executable hash;
its source matches the tested candidate. Build receipts, exact
original/candidate sources, CPU/GPU fixtures and execution records are saved.

The model check's complete private API log is 1,078,290,262 bytes. Its digest,
API entry/result counts, unsupported graph-query counts, phase counts and
final 64 KiB are archived. These results cover the two helpers and short
output parity. Full-cell CLI and normal-MTP serving gates, other query/Q4
pointer casts, throughput and any causal link to prior stalls remain
unproven. The preceding full-prefill failure used the original engine and
is recorded separately in the
[phase-sync capacity evidence](../phase-sync-capacity-20261007/README.md).
