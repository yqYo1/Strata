# PR #6 engine restoration for isolated review

This directory restores the engine source closure selected by
`d0c5dda0dc2e8d82458c4b0ff31c4cef44c40b32:sycl/CMakeLists.txt`, including its
historical root fallback files and headers. The implementation is
`f263d87d44aa671d056c33e6c6b6b0635fc25b8c`; engine files are identical at those
revisions. This is an independent validation route. The normal root build and
`feature/sycl` are unchanged.

The first restoration commit preserves every historical project file exactly.
It is an unsafe source reproducibility anchor, not a validated executable.
Later commits hold separately attributable safety fixes. Main must review the
whole patch before building or executing either version. No build or test has
been run by the worker; >600 tok/s has not been remeasured.

The new outer CMake wrapper changes only isolation/dependency provisioning:

- It requires Main's explicit reviewed-build flag.
- It delegates to the unchanged historical `sycl/CMakeLists.txt` and resolver.
- It requires external historical DPCT/ggml-common headers, checking all 49 file
  SHA256s from `historical-closure.json`. No dependency tree is copied here.
- It requires an existing offline llama.cpp checkout at the historical pin
  `3cf03257f219afbe7334045ff7c6a06ac68c627d`. Main must verify its HEAD and diff.
- It disables the historical GPU parity target group because this restores the
  engine target `strata`, not the unrelated legacy fixture suite.

`sycl/tools/migrate.sh` and `fixups.py` document the original generation route;
regeneration is neither required nor appropriate for reproducing these blobs.
`sycl/tools/prefill_profile.py` is the final PR helper. It starts native programs;
do not run it until Main schedules execution. Its default cwd is historical to
this nested snapshot; pass an explicit cwd for a scheduled reproduction.

See `../../bench/results/2026-10-16-pr6-faithful-restore/report.ja.md` and
`manifest.json` for provenance, differences, bugs and Main-owned execution gates.
`historical-closure.json` inventories all copied files, their original Git blob
IDs and SHA256s, resolved translation units and external headers. Runtime sources
under `sycl/src/kernels/cuda/*.dp.cpp` are the historical SYCL migration output;
no CUDA/HIP implementation is built or newly imported.
