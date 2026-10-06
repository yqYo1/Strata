# Ordinary expert-cache reclamation, 2026-10-06

The SYCL port's ordinary `ExpertCache::close()` freed its USM payload without
waiting for pending cache fills or readers. The segmented branch and ordinary
`shrink()` already waited. Khronos's [USM reference](https://github.khronos.org/SYCL_Reference/iface/usm_allocations.html#sycl-free)
requires all commands using an allocation to finish before `sycl::free`.
Free does not perform that wait. Copy queues and compute queues can both use
the cache, so close now drains every registered queue on its device. If that
wait throws, it retains the payload instead of freeing it.

The test extracts the actual production close method and compiles it with CPU
queue/USM stubs under GCC AddressSanitizer and UndefinedBehaviorSanitizer. It
links no SYCL or Level Zero runtime and executes no GPU work. Both old-code
controls, deferred read and deferred write, report heap-use-after-free. All
five candidate cases pass: read, write, two queues, failed drain and repeated
close. This demonstrates the reclamation defect when consumers remain pending;
it does not demonstrate that a particular inference caller left them pending.

Reproduce from the repository root, with a new output directory:

```sh
python3 sycl/tools/test_expert_cache_close_host.py --output /tmp/strata-cache-close-check
```

[record.json](record.json) pins the old control to `fde2eed5524034a94b30c5fb8e870ccebddbea90`.
[evidence-manifest.json](evidence-manifest.json) records the source, harness,
logs, build and frozen engine hashes. Public logs remove trailing whitespace;
raw log hashes are preserved. The production `strata` target rebuilt and
linked successfully with oneAPI 2026.1. Its engine SHA-256 is
`d70286a29da5c3e15c350e76ac493efd66bf576b9f7dd2b6ed29966a29e6118d`.
The CPU kernel archive is unchanged. This candidate has not run on the GPU.

The original GPU hang trigger is still unresolved. The cache fix changes
resource lifetime only; it is not evidence of GPU recovery, arithmetic parity,
full-context correctness or a speed improvement.
