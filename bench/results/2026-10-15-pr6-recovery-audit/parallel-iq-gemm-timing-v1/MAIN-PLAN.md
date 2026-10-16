# Standalone component timing proposal

Status: implementation proposal only. Build, CPU tests, GPU tests, measurement,
fault checks and external capture comparison: **NOT_RUN**. No speed claim or
product adoption. Main owns execution; the existing live GPU comparison was not
touched. No product files, branches, Git state, services or builds were changed.
The project strata-upstream-sync skill and artifact retention policy apply.

Compile `timing.cpp` separately with Main's pinned custom clang 22 / oneMKL
2026.1 stack and current baseline archive recipe. Compile/link the proposed
`parallel-iq-dequant-v1/proposed/src/kernels/sycl/iq.cpp` object, placing its
include directory before the repository include directory. Link the retained
baseline archives after that object so their original IQ object is not selected;
verify the actual link map/symbol provenance (avoid duplicate IQ definitions).
Use existing baseline compile flags, including `-fsycl`,
`-fsycl-id-queries-range=size_t`, `-fno-fast-math`, `-ffp-contract=off`, C++20,
and the baseline MKL BLAS link flags. Pin actual compiler, runtime, archives,
object, source closure, binary, command and effective environment in Main's
build receipt. This proposal does not guess archive locations or replace the
working baseline build configuration.

The executable takes:

```
timing 0|1 TYPE gu|down NE dequant|gemm REPEATS TRIALS WARMUP [NEW_CAPTURE]
```

Set `STRATA_SYCL_PARALLEL_IQ_DEQUANT` to exactly the corresponding `0` or `1`
in the native child environment **before exec**. The argument and environment
are checked before queue/provider construction. There is no environment mutation.
Both arms use the same executable, packed inputs, shapes, public materialization
entrypoints and FP16-to-FP32 row-major oneMKL GEMM. GU is 640 rows by K=2560,
materialized as alternating gate/up rows, N=1280 (types 18,22,21,23). Down is
N=2560 by K=640 (types 20,42). NE is 4,16,64,128,512. Separate processes select
`dequant` or `gemm`; the latter rematerializes weights on **every** repeat.
No packed XMX implementation or altered arithmetic is included.

Main should run one small supervised smoke case first, with clean logging and
fresh fault boundaries. A suggested starting batch is REPEATS=5, TRIALS=5,
WARMUP=2; the accepted bounded ranges are repeats 1..100, trials 3..30, warmup
1..10. Main may adjust batch count before the comparison, then fix it across
both arms and independent processes. Do not change counts based on a favorable
arm. For each type/NE/mode use at least three independent processes per arm,
alternating first arm. Main's serial controller owns the exclusive measurement
lock, immutable env, process identity/start time, deadline, raw log budget,
reaping and fresh driver/kernel fault receipt. Do not run this concurrently with
GPU work, builds, tests or cleanup. Start with a finite deadline large enough
for untimed scalar host decode and oneMKL initialization.

Each trial records the host submit interval separately from completed batch
host wall through checked queue completion. These are batch measurements,
including host submission/driver costs, and **are not exclusive kernel times**.
Divide completed batch wall by repeats for per-operation comparison; retain all
raw trials and process order. No profiling queue or API trace is enabled by this
fixture. Record the inherited environment to exclude external trace overhead.
Queue is in-order and the same scratch is intentionally rewritten in serial;
there is one public dequant call per repeat and one GEMM call in combined mode.
OneMKL internal kernel counts are not asserted. NE does not change dequant-only
work; dequant NE entries are duplicate workload identities, not distinct speedups.

Before timing the entire finite half output is compared bit-for-bit against the
existing scalar CPU decoder with repository RNE half conversion. Combined mode
computes an untimed full GEMM reference from that CPU-decoded weight buffer
using exactly the same oneMKL FP16/FP32 parameters as current
`src/prefill/sycl/gemm.cpp`: nontrans/trans, alpha=1, beta=0, ldX=ldW=K, ldY=N.
Every batch verifies all dense halves, finite outputs, exact full GEMM reference
parity, unchanged full packed weights/X and all seven allocation guards.
It checks once again after all trials. The CPU decoder is independent of the
proposed element-parallel decode helper. Combined parity establishes same GEMM
inputs/operation, not independent validation of oneMKL itself. Signed float
zero differences pass numerical equality and remain visible in output bit
fingerprints/external comparison. Max absolute output error is recorded.

JSON on stdout is published only after checked final drain, allocation release,
and queue retirement. The async handler maintains a sticky fault and first
exception; all exceptions fail-stop with exit 86, without implicit USM/queue
unwinding, freeing unknown-completion storage or retrying. Main must treat any
missing complete JSON, abnormal exit or fresh fault as failure. Heap Runtime
is intentionally leaked until process exit, even on normal completion, after
explicit retirement; it cannot be implicitly destroyed.

Optional NEW_CAPTURE stores only dense-half output for dequant mode or float
GEMM output for combined mode, bounded to 32 MiB per case (actual maxima are
6.25 MiB and 5 MiB). Use only for a named full OFF/ON comparator; the same case
in independent arms can be compared byte-for-byte. Compact JSON includes FNV1a64
input/output fingerprints (not cryptographic identity). Main records SHA256 for
any retained capture plus full-buffer comparison scope/results. Once verified,
retire successful captures with a path/size/hash/reason manifest; retain compact
JSON and the external comparator receipt. No capture is required for clean timing.

Source pins describe the proposal's inputs, not a built executable. Main must
refresh the actual build closure if a pinned source changes before compilation.
