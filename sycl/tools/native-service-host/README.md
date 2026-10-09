# Standalone native expert service host

Root built this target with the pinned compiler and ran a bounded actual-weight CPU smoke under the shared measurement lock. The recorded checks and limits are below. Root owns all subsequent builds and runs. It does not include the parent SYCL CMake project. The target compiles the production role plan, layout, wrappers and pool, including the SYCL-side IQ AVX2 forwarding source, with CPU-only pinned ggml and Threads.

The executable takes three positional arguments:

```
native_service_host PACK_DIR PRIMARY_GGUF COHORT_TSV
```

The owner must freeze the immutable actual pack, primary/shard identities, native metadata and cohort before running. The TSV begins with literal `layer<TAB>expert`, then one decimal pair per line; 1–96 unique pairs, layers 0–47 and experts 0–511, at most 8192 bytes. Select these pairs from actual route evidence; the program does not select a cohort or read route logs. It accepts H=2560, FF=640, GU18/21/22/23, Down20/42. NativeRolePlan validates all 48 layers' three role extents before copying selected blobs into owned buffers (aggregate at most 1 GiB). Mappings close before service timing. Root must prevent shard mutation/truncation throughout assembly and bind SHA256 identities externally; emitted FNV checksums are not cryptographic provenance.

Root-only proposed build recipe (not executed):

```
cmake -S "$WROLE/sycl/tools/native-service-host" -B "$HOST_BUILD" \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DSTRATA_ROOT="$WROLE" \
  -DGGML_SOURCE_DIR=/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned/ggml
cmake --build "$HOST_BUILD" --target native_service_host --parallel 6
"$HOST_BUILD/native_service_host" "$PACK_DIR" "$PRIMARY_GGUF" "$COHORT_TSV" \
  > "$PRIVATE_RUN/samples.csv" 2> "$PRIVATE_RUN/pool-diag.txt"
```

Use the pinned oneAPI 2026.1 IntelLLVM environment. Configure checks ggml HEAD=3cf03257f219afbe7334045ff7c6a06ac68c627d. The owner additionally verifies clean ggml sources, exact compiled source hashes, flags, compiler identity and binary hash. Strata TUs have no global march; ISA flags are per TU. Q2 AVX-VNNI is the compile-probe result, IQ wrapper AVX-VNNI is zero, IQ2S GCC aliases are absent, and precision is `-fp-model=precise`. GGML_NATIVE applies its own host flags. Root must audit the actual link/import closure and use the R74 runtime no-device tripwire before calling a source-only CPU closure a runtime qualification. No engine/GPU callback is registered; the harness checks the nullable release callback remains null.

All host calls and sequential tasks=0/tasks=6 pool lifetimes run under production host affinity pinning, with RAII restoration. Each pool has five pinned workers and host participation. TASK_PLAN gives 18/18 default GU/Down row tasks, or 6/6 explicit tasks. PLACEMENT contains planned worker CPU IDs; observed host CPU is recorded. This does not independently verify each worker's OS affinity; root must retain an actual thread-placement audit. No artificial ISA feature environment is allowed. Relevant dispatch environment strings are recorded; root must freeze them before starting the process.

Each expert/NT1 or NT2 cell gets three warmups and five retained repeats, separately for each pool lifetime. FP32 input is deterministic, finite and distinct per token; the formula is emitted. Input quantization is timed separately. Direct GU, FF quantization, Down and outer duration are separate samples. Pool GU/Q/Down accumulator deltas and actual outer duration are separate fields. Allocation, assembly, GGML initialization, independent references, canary checks, byte comparisons, checksum and printing sit outside timed intervals. No cache flushing, page reset or preload is performed; assembled selected blobs are naturally touched and this is a small hot-cohort measurement.

Before recorded repeats, production input quantization must exactly match meaningful pinned GGML `from_float` bytes, with untouched tail guards. Down20 FF quantization gets the same independent GGML byte check. Down42 uses full ActQ and directly compares meaningful fields with the selected production AVX2/AVX512 quantizer; that check is a route/representation check, not an independent arithmetic oracle. ActQ/FF/output guards and finite/full-row writes are checked. Each direct repeat matches its initial output; each pool result matches direct output byte for byte, independently for both tokens. Input quantization is reverified after the pool call.

NT1 GU and Down20 must match independent per-row GGML `vec_dot` results bitwise, including the same SiLU composition. NT2 GU and Down20, and Q2 Down, expose maximum absolute differences without a new tolerance. The Down characterization uses the actual direct FF vector quantized into GGML's activation type; for Q2 this differs from ActQ and is explicitly a different route. No NT1/NT2 cross-route equality is demanded. Direct/pool parity is a partition correctness check; it is not an independent numerical reference. This fixture does not certify NT2 or Q2 independent-arithmetic equivalence.

CSV includes META, ENV, EXTENT, BLOB, PLACEMENT, TASK_PLAN, ROUTE, SAMPLE and RESULT rows. SAMPLE includes all five individual repeats and phase durations, formats, NT, tasks, workers, output checksum and reference differences. stderr contains a named DIAG marker and production `pool.diag` after every call, outside its timed interval. RESULT/pass means the defined finite/quantizer/NT1/reference/partition/repeat gates passed, not performance adoption. Failures return nonzero. A pool's existing stall watchdog can abort; the owner must separately record signals and closure.

Root admission and supervision remain mandatory: a new private output directory, shared serial lock, finite wall/CPU/RSS/address-space/descriptor limits, at least 1 GiB selected-blob allowance plus pool/ggml overhead and full-shard virtual mappings, bounded stdout/stderr (proposed 8 MiB each), exit status, faults and cleanup ownership. The finite loop is at most 384 cells, 3072 pool calls and 1920 recorded samples. Root should first run invalid TSV/geometry cases with small synthetic role-plan fixtures, review compile commands and linkage, then use frozen actual payloads. Do not run a model or GPU engine as a substitute.

This is not end-to-end inference, R46 calibration (no 25-round/minimum-cohort/holdout qualification), actual decode activations, weighted/shared-expert evaluation, mixed callback replay, worker tail measurement, DRAM traffic, GPU overlap, or a performance/adoption result. No existing failed C/D/r5 status or physical/full lifecycle qualification changes.

On Linux, after the first completed warmup of each pool lifetime, the executable also emits `/proc/self/task/*/status` CPU-allowed lists, bounded to 128 threads and less than 16 KiB per status. Those are OS affinity observations by thread ID, not a mapping of worker index to thread ID or a trace of CPU residency. Root must reconcile them with the planned host and worker CPUs. Other systems report that snapshot unsupported; the Linux recipe is the primary profile.


## Recorded root verification, 2026-10-10

The standalone production CPU build passed in 25.325s, with 37 compile rules
including 11 Strata/harness translation units. Strata flags are precise and
per-TU, Q2 AVX-VNNI is 1 and the IQ wrapper is 0. Direct imports are the ordinary
host C/C++ libraries. This build does not link the parent SYCL engine.

A traced actual-weight smoke passed in 17.345s on 14 routed expert IDs across
seven GU/Down format pairs. It retained 280 individual NT1/NT2, tasks0/tasks6
samples. NT1 GU and Down20 independently match GGML exactly; the complete
controlled direct/pool outputs and repeats also match exactly. NT2 and Q2
independent arithmetic remain delta characterization, without a new tolerance.
The process exited normally, with no forced cleanup or surviving owned process.
For that exact binary, environment and lifetime, strace/loader observation
found no GPU device open/ioctl or GPU runtime library load. This is a scoped
runtime observation, not API interposition or a universal binary guarantee.

The samples include syscall tracing and a small hot cohort. They cannot
select a task policy, supply CPU service weights, or demonstrate inference
performance. R46 requires separate streaming cohorts and repeated whole rounds.
Source, root commands, exact inputs/extents, samples and receipts are committed
in `bench/results/2026-10-10-native-service-host/`. R77's alleged missing input
initialization was a false source reading and is explicitly withdrawn by R78;
no implementation change was made for it.
