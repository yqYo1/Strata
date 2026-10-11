# Opt-in prefill embedding / PLE buffer reuse

The experimental flag shares one FP32 plane after embedding broadcast completes on the same admitted in-order device/context queue. Default allocations and numerical operations are unchanged.

Requested savings are 80 MiB plus 256 bytes at T8192 and 100 MiB plus 256 bytes at T10240. Physical savings, larger-chunk fit and performance have not been measured.

Main built free and contrib-llvm and their host address/undefined sanitizer variants from the qualified source-built LLVM toolchain. Each mode ran 14 CPU tests successfully and skipped one AVX-512 feature test. Mandatory lints introduced no findings. One earlier cast diagnostic and one earlier host ELF parser failure are preserved with their original results.

GPU alias parity, full-vocabulary continuation comparisons, actual allocation admission, full 262144-cell state validation and 64K timing remain open. Existing CPU context tests do not prove physical GPU endpoint correctness. The prepared planner helper implicitly initializes Runtime and was not launched as a CPU-only test.

See main-results-and-corrections.json, byte-exact receipt archives, research/implementation reports and deletion-manifest.json. The upstream parameter-audit TODO remains open for independent official settings selection and held-out validation.
