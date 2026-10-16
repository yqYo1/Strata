# Own prompt K cell-major v2 — source proposal only

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

V2 leaves v1 immutable. Product kernel and CMake proposed bytes are exactly v1: kernel SHA-256 `a22ae0834ead98e0b129b3f31a15cb06db99fe855e96b03c10f216df713f07eb`; CMake SHA-256 `2544f5a426f3565ade25f3860c0f6317d1770fdc9e225cb0439ff5218ede14d2`. Only the fixture performance path and drain handling change. Normal invocation and `--dump PATH` still run the complete v1 formats/pages/scaled/oracle/masked/invalid-width matrix, batch/single equality and original shape rejection cases. No test was executed.

Explicit bounded benchmark syntax (Main future execution only):

```
env STRATA_SYCL_PROMPT_DIRECT=1 STRATA_SYCL_PROMPT_K_CELL_MAJOR=0 <build>/sycl_prompt_k_cellmajor_test --bench 256 3 --dump <owned>/off-bench.bin
env STRATA_SYCL_PROMPT_DIRECT=1 STRATA_SYCL_PROMPT_K_CELL_MAJOR=1 <build>/sycl_prompt_k_cellmajor_test --bench 256 3 --dump <owned>/on-bench.bin
cmp <owned>/off-bench.bin <owned>/on-bench.bin
```

`--bench` accepts exactly nq128 or256 and iterations1..20, with mandatory output dump. It fixes Mode1, page4, cap2051; query0/1 counts are13/67, all remaining queries count2051. This launches256 or512 workgroups, rather than v1's6, retaining the actual matrix API and existing selection/data construction. No in-process environment mutation occurs. One completed untimed warmup and output download precede host steady-clock timing around repeated submissions and a final checked drain. Allocation/copies/dumping/oracle are outside the interval. Repeated output must exactly match warmup. Timing includes host submission and queue service; it is not pure kernel time or C8192 model throughput.

Benchmark mode skips the nq independent single-query launches and limits scalar FP64 comparison to explicit query indices0,1,2,last, all24 heads/all256 dimensions, using the existing1e-5 tolerance. All outputs receive finite checks and output/scratch tail-canary checks. All output bytes plus output tail canaries are dumped; Main must compare separate OFF/ON dumps exactly. Stdout labels the limited scalar oracle and external exact comparison requirement. Benchmark mode does not repeat the normal matrix's all-masked/zero/invalid-width cases; those remain intact under normal invocation. A passing limited oracle does not establish full scalar reference parity for all128/256 queries. Dump maximum at nq256 is6291520bytes per arm; record hashes/comparison then retire once reviewed. Main owns qualifying runs, immutable arm environments, timeout/log budgets, faults and process ownership; use same shape/repetition count for comparisons.

The local Drain guard now aborts on a failed runtime wait rather than continuing owner unwinding when retirement is unknown. Existing own-source `Allocation::release` calls `checked_cleanup`: failed completion check invokes `cleanup_fail_stop`, which exits before `sycl::free`; release failure also stops. Thus normal drain errors remain failures and never authorize a free. The main error handler remains nonzero; local guard protects outstanding run allocations before owners unwind. No runtime/allocator implementation is changed.

Before integration Main must review complete diff, verify base hashes and apply/rebase explicitly preserving concurrent product edits, compile/link, run component and separate-arm parity, inspect resources/lowering if feasible (R734 optional offline AOT; not a JIT resource claim), then matched actual78453 model and clean repeated whole-prefill. Existing task-quality, actual RESTORE and physical262144 gates remain. No current >600 or1000 tok/s milestone is established.
