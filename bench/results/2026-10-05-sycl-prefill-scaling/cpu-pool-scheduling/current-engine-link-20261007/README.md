# Current task-factor-9 engine relink, 2026-10-07

The current reference and candidate CPU archives exactly match the previously
measured and byte-validated production archives. Only `pool.cpp.o` is replaced
by `native-pool-task-factor.cpp.o`; every other archive member is identical.
The current engine, prefill, verifier, device kernels, core, ggml and program
objects are reused unchanged. This includes the latest host-source, queue and
recorded-graph lifetime repairs, rather than reusing the older whole engine.

The resulting candidate SHA-256 is
`64385aa3fd3da16122081d53bb51be004e62597a2f9360af77dcfdc079f9f7b7`.
The frozen reference remains
`79a4b363d33f66f41d910be6274e609b4eb73f62afb0dc49bca72bf2a538d223`.
This is a build and archive-identity check. Neither GPU arithmetic parity nor
whole-engine throughput improvement is established by this relink.

`current-tasks9-link-system-tools/record.json` records the successful link,
all reused archive/program hashes and clean dynamic-library resolution.
The original `current-tasks9-link/` failure is retained: compiler helpers
inherited Nix tools through PATH while the loader selected the host libc,
causing a GLIBC_PRIVATE symbol error before GPU execution. Restricting this
compiler invocation's PATH to the system tools fixes the link; no package or
global environment was changed. CMake's default task factor remains 0.

The earlier CPU measurements and complete-output/sanitizer checks are in the
[parent report](../README.md). Executables and archives remain private; this
directory stores the exact controllers, receipts and archive-member digests.
