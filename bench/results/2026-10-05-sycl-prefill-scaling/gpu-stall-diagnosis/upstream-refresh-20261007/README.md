# Upstream refresh checked on 2026-10-07

Upstream Niko1221/Strata at 82f46a8c8f475f001ad76d92f58f4a4f8ffb0253 is
integrated on the existing SYCL work branch. Upstream rewrote its history:
old base 6f32ec07 and new-history a1641e9f have the exact same whole-tree
SHA 27b0e86f. An unchanged-tree bridge records this equivalence before a
normal three-way merge. Five conflicts were resolved with full file-tail
checks, retaining local graph retirement, atomic host doorbells and source
lifetime protection. No branch history was force pushed.

An isolated Release build with oneAPI 2026.1.1 and the existing GCC IQ2_S
object option passes. Shared API compatibility adds default contiguous
strides, optional host-registration waits and upstream CPU file-source
methods. Unsupported new CUDA fused/padded modes reject explicit requests.
Upstream CUDA elastic K/V remains unavailable; the existing Level Zero
expert-cache mapping is retained. SYCL device arithmetic, launch geometry
and per-phase waits are unchanged. The SYCL engine retains its upstream
0.1.39-sycl version; this receipt does not assert every CUDA 0.1.40 feature
has a SYCL implementation.

Host validation: 452 serve tests pass, with eight skips; all 24 setup scripts
pass (the unsloth test mocks were updated for the new minimum engine);
message-boundary, exchange-storage, draft-policy and suffix-drafter existing
standalone CPU tests pass. Initial build and test failures are retained
alongside their passing replacement receipts.

On B570 10 GiB / Ryzen 5600X / 128 GiB RAM, the updated executable
ceff2d8a…680c0bd0 runs through owned GDB with flushed Level Zero/UR logs and
validation. Four context-128 normal-MTP requests match the frozen pre-update
IDs, every printed logprob and every byte of all 248,320 first-head floats.
Six MTP release/restore pairs complete. A separate context-4096, int8 KV,
2048-token/two-1024-chunk normal-MTP request matches the earlier exact-head
control and completes one release/restore pair. Both exit normally; no
owned processes survive and no new xe fault is recorded. Diagnostic times
are not clean throughput figures.

The repeated 262144-cell CLI/serve, clipped-tail, overflow-refusal and
later-valid gates remain incomplete. Previous full-capacity failures and
eager/lazy whole-head differences are not resolved by this update. No
reset, service change or package change was performed for these GPU checks.
Private whole heads and large diagnostic logs are hashed; bounded tails,
controllers, build/source hashes and terminal receipts are preserved here.
