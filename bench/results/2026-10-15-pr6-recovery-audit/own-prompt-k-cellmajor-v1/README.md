# Own prompt K cell-major v1 — source proposal only

Pinned base: `3d873f9ca050db105122b8b18c8a8e1275aecdde`. `manifest.json` pins every original and proposed source byte hash. Original files remain reachable in that Git commit; `files/` contains complete proposed files and `source.diff` is against that same commit. `prepare.py` reconstructs those bytes from the immutable commit without consulting changing worktree source. Only this proposal directory was written. No product edit, build, test execution, GPU, runtime, process/service/environment change, cleanup or Git mutation occurred. All tests are UNRUN.

Implementation origin is this fork's own `src/kernels/sycl/prompt_attention.cpp` and R725. Reviewed project skill/AGENTS, artifact policy, current research index, R725, R732, Main correction731–732 and R734. XeStrata source was not opened or used. No external implementation or CUDA/HIP work is involved.

Main's newly closed fine observation motivates a bounded experiment: actual78453/C8192/FP32/R32768, matrix_chunks120/fallback0, SUM_T941424/cap2051; selected768 exact, native/server0/fault0. Main reports QSA104926ms, attention69909.3ms, hostsubmit9.42802ms, selection24399.6ms, KVstage937.713ms and wrapper251647ms. These nonexclusive observer intervals do not isolate K gather or pure kernel time. The trial is a hypothesis, with no performance, adoption or quality claim. Main's R731–732 corrections supersede R732's workload arithmetic; no old operation/byte count is reused here.

## Product change and dispatch

Only exact environment value `STRATA_SYCL_PROMPT_K_CELL_MAJOR=1` enables the candidate; unset, empty, 0, 10 or other values use the old route. Set all environment values before native exec and leave them immutable. Existing Direct selection semantics remain `STRATA_SYCL_PROMPT_DIRECT` starting with 1. Candidate dispatch requires Mode1 INT8 K8V8 AND Direct1. Modes0/2/3 and Direct0 ignore this switch. Native=true exists only in Mode0 and never selects this candidate; the static assertion enforces Mode1/Direct1/Native=false.

There is exactly ONE additional launch/device-kernel specialization: `<1,true,false,true>`. Nine old specializations remain, now with a default false final template parameter (Mode0 has 3, other modes have 2 each). The old branch is compiled with `if constexpr`, retaining its original gather statement and declared local resources; there is no runtime device branch. Template symbol names necessarily change; identical compiled code/resource allocation is a verification gate, not an assertion about compiler output.

The new traversal maps `c=i/256, dim=i%256` and stores `kv[dim*32+c]`; the original maps `dim=i/32,c=i%32` and stores `kv[i]`. Both enumerate all8192 logical destinations once with the same exact INT8-to-half conversion and invalid-row zero. Row resolution, scales, V gather, matrix arithmetic, barriers, CH32/WG128, pool/tile layouts and placement remain byte-identical. All lanes execute both existing surrounding barriers uniformly. No new local storage or barrier is declared. Bounds remain the existing contract: valid nonnegative selected IDs fit the page table, valid physical pages fit pools; no wider loads or new ID acceptance is introduced. The strided transpose stores can increase SLM conflicts and regress performance.

## Actual configured SYCL fixture

`cmake/sycl_backend.cmake:77–85` explicitly compiles `src/kernels/sycl/prompt_attention.cpp` into `strata_kernels`. Added manual `EXCLUDE_FROM_ALL` target `sycl_prompt_k_cellmajor_test` links that library and compiles new `tests/sycl/prompt_k_cellmajor.cpp` with `STRATA_TEST_PROMPT_MATRIX=1`, strict FP flags, calling `qsa_prompt_attn_batch` directly. No automatic GPU test is added. The root CUDA parity executable is not used or assumed active.

New fixture is based solely on the own-source SYCL `tests/sycl/decode_attention.cpp` at the pinned commit. That old fixture writes process environment after runtime initialization and between arms; all those blocks are removed in the new fixture. It performs no setenv/unsetenv at all. Never use the old fixture's environment-mutating arms for this comparison.

Coverage: all256 dimensions/32 slots, reversed page indirection, negative IDs/pages, partial final tile (cap2051), counts13/67/2051, all masked/zero/invalid widths, K8V8 plus nonselection modes0/2/3, batch versus single exact output, FP64 whole-attention reference with existing1e-5 tolerance, finite outputs, output/scratch tail canaries. Independent CPU logical-tile mapping checks every destination has one writer and old/new exact half bits for counts0/1/13/31/32 and page indirection/negative rows on both KV heads. This is a source-level logical proof; actual candidate device correctness is checked through whole-attention parity and separate-arm exact dumps, not a claim to inspect device SLM directly. The drain guard precedes allocation owner destruction even when validation throws; uploads/downloads wait, main also drains on exit/error. As with original tests, tail canaries do not establish absence of every possible read out of bounds; valid-ID/pool allocation contract remains required.

Main future execution (NOT run here): build manual target using existing pinned SYCL configuration/toolchain in an isolated owned build. Run independent supervised processes, fixed environment before exec:

```
env STRATA_SYCL_PROMPT_DIRECT=1 STRATA_SYCL_PROMPT_K_CELL_MAJOR=0 <build>/sycl_prompt_k_cellmajor_test --dump <owned>/off.bin
env STRATA_SYCL_PROMPT_DIRECT=1 STRATA_SYCL_PROMPT_K_CELL_MAJOR=1 <build>/sycl_prompt_k_cellmajor_test --dump <owned>/on.bin
cmp <owned>/off.bin <owned>/on.bin
```

Also repeat OFF/ON for Direct0 to prove nonselection. Run native minimum values0/1 in separate processes for modes0 (covered within each full fixture) to confirm the old native route remains correct; no in-process environment writes. Dump includes all cases in deterministic order plus tail guards. Retain hashes and compact exact-comparison receipt, then retire dumps after review. Maximum full dump is under2MiB per arm.

For bounded component timing append a repeat count1..20 to `--dump PATH`; this chooses only fixed mode1/page4/nq3/cap2051/counts13,67,2051. One untimed completed launch precedes a host steady-clock interval around the repeated submissions and a final drain. Output equality is checked after the timed drain; oracle/copies/allocations are outside that interval. The interval includes submission and queue service, not pure kernel duration. No event per expert or profiler instrumentation is added. Main chooses serial qualifying runs, independent processes, timeout/log budgets, process ownership and fault/journal boundaries; compare identical shapes/repetition counts and record all samples. This narrow shape is not a C8192 throughput proxy. No speed claim until matched whole-path repeats.

Before integration Main must review complete diff, verify base hashes and apply/rebase explicitly preserving concurrent product edits, compile/link, run component and separate-arm parity, inspect resources/lowering if feasible (R734 optional offline AOT; not a JIT resource claim), then matched actual78453 model and clean repeated whole-prefill. Existing task-quality, actual RESTORE and physical262144 gates remain. No current >600 or1000 tok/s milestone is established.
