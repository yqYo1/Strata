BF16 residual boundary storage proposal v1, 2026-10-15

Source-only proposal against the actual `feature/sycl` worktree. No root source
was edited. No compiler, build, test (including host tests), GPU, process probe,
Git mutation, environment mutation or cleanup was run. Main owns integration and
all serial validation after its live baseline closes. `source.diff` is the
integration payload; `source-manifest.json` pins each actual root base and
proposed file SHA-256. The new fixture source is included both in the patch and
as `prefill_bf16_storage.cpp` for direct review; no existing full sources are
duplicated. `artifact-sha256.json` pins the proposal and its read-only inputs.

This is an optional storage format, not an adoption decision. FP32 remains the
default and reference; FP16 remains separately selectable. Exact case-sensitive
`STRATA_PREFILL_LAYER_MAJOR_R_DTYPE=fp32|fp16|bf16` is accepted. BF16 changes only
the inter-layer residual store: one RNE pack after each non-final layer, exact
FP32 expansion before the next layer. Final-layer callbacks stay FP32. Weights,
KV, recurrent state, arithmetic, operator types and activation hi/lo conversion
are unchanged. In particular the existing SYCL activation `to_bf16` is not used
or changed: boundary-only `residual_pack_bf16` and `residual_unpack_bf16` call the
shared `bf16_from_f32` and `f32_from_bf16` helpers.

The configured SYCL resolver in `cmake/sycl_backend.cmake` directly compiles
`src/prefill/prefill.cpp` and `src/prefill/sycl/postops.cpp`; `prefill.cpp` includes
`src/prefill/layer_major.inc`. This patch targets that route only. CUDA/HIP
implementation or validation is outside scope.

Both two-byte formats reserve the same checked staging allocation, share generic
`two_byte_stage` owner/member names, and use the same single in-order compute
queue and outer checked lease. Full-device store packs directly into GPU rows;
host/mixed tails reuse the single device chunk for H2D -> unpack -> existing
model work -> pack -> D2H. Explicit dtype dispatch prevents a two-byte store
falling through to an FP32 copy. The store callback checks its drain even when
copy submission reports an API error. Existing fail-stop retirement and
owner/Restore destruction order are preserved. Phase audit remains FP32-only.

BF16 uses RNE, preserves signed zero and infinity, preserves BF16 subnormals,
rounds tiny magnitudes to signed zero, can round finite maximum-range inputs to
signed infinity, and quiets NaNs while preserving sign and retained payload.
Restore is a bit shift, including signaling-NaN encodings. Planner products,
prefix/tail offsets, staging/admission budget and pointer-span checks cover BF16
as two bytes. At 262144 planned rows, width 10240, the store is 5368709120 bytes
and an 8192-row host-tail stage is 167772160 bytes. These are layout arithmetic;
they do not establish physical 262144-context admission or correctness.

The optional boundary audit now takes an explicit dtype and emits
`boundary_audit_v2`, `storage=bf16`, matching converter metadata, packed-bit and
restored-input phase names. Raw selected output rows stay FP32 with the same
order and budgets. FP16/FP32 phase semantics stay available. No historical
checker, receipt, FP16 proposal or result is edited. The separate
`consume_closed_fixture.py` consumes BF16 fixtures and refuses quality/speed/
physical-context qualification.

Main's later serial validation, not executed here:

1. Recheck base hashes before integrating `source.diff`; stop on drift rather
   than apply it over concurrent changes. Build the configured SYCL engine and
   target `sycl_prefill_bf16_storage_test` with its actual pinned toolchain.
2. Run the extended `prefill_layer_major_plan_test`, the existing exhaustive
   shared BF16 CPU bit helper test, existing FP16/postops regressions and the new
   conversion mode. The host planner adds BF16 strict parser/name/invalid-enum
   cases and repeats two-byte full/mixed/host, partial-tail, over-budget and
   checked-overflow cases without changing FP16 numerical expectations.
3. Run the fixture four times in separate owned processes, serially:
   `sycl_prefill_bf16_storage_test conversion`, `... fullgpu`, `... fullhost`,
   `... mixedstore`. Conversion uses 35 literal pack cases, all 65536 BF16 restore
   codes, and 8 negative-geometry/overlap/unordered-queue rejection cases.
   It includes ties-even, both signed zeros/infinities, finite overflow,
   subnormal/underflow edges and signed signaling/quiet NaN payloads. All three
   placements use 19 rows / chunk 4 / width 10240 / 48 layers, with a 3-row final partial
   chunk, 235 packs and 235 restores. The mixed prefix is 8 rows; fullhost is 0.
   Each scalar is checked against a literal bit oracle. This synthetic fixture
   injects a fresh layer result and checks the previous restore: it is storage
   mechanics coverage, not a model or a quality simulation.
4. For each store mode set `STRATA_PREFILL_BOUNDARY_ROWS=0,3,4,7,8,15,16,18` and
   a new absolute `STRATA_PREFILL_BOUNDARY_PATH` in that run's private directory
   before exec. Leave `STRATA_PREFILL_BOUNDARY_ROWS_ABSOLUTE` absent. Runtime,
   driver, profiler and other experiment variables also belong in that native
   child environment before exec. Do not switch arms with setenv/unsetenv.
   The audit exclusively creates its log and `.rows.f32` companion. Its raw
   source rows total 15728640 bytes per placement; retain them only until the
   new consumer's exact placement comparison is recorded and verified.
5. Record Main's owned-run closure, fresh driver fault receipt and compile/
   source/binary identities; consume the closed artifacts with
   `python3 consume_closed_fixture.py --closure /absolute/closure.json --output
   /absolute/new-bf16-mechanics-receipt.json`. The output uses exclusive create.
   No execution, supervision or cleanup is implemented by this consumer.
6. Run an actual small model BF16 boundary arm under fullGPU/host/mixed controls
   and selected-row boundary audit across all 47 non-final boundaries. Compare
   BF16 placement against BF16 placement; compare dtype impact separately
   against a matched fixed-prefix FP32 control. New BF16-specific receipts must
   identify dtype, conversion APIs and all byte fields. Historical FP16 receipts
   remain byte-identical. The old checker accepts only FP32/FP16 and is not the
   BF16 consumer. The new consumer here is for synthetic component fixtures,
   not a replacement for Main's actual-model controller or quality evaluation.
7. Then evaluate natural histories, first/middle/final selected full-vocabulary
   KL and teacher-forced target NLL, paired task/retrieval outcomes at intended
   32K/64K lengths, and a physically admitted 262144-context correctness case.
   Numeric margins from research are proposals, not preapproved acceptance
   criteria. Run independent clean end-to-end prefill speed samples with
   matched placement/configuration and separate decode checks. No BF16 quality,
   performance, RESTORE or physical-maximum qualification exists yet. 600/1000
   tok/s remain goals, not measured BF16 outcomes.

The closure JSON is supplied by Main's existing supervision, using absolute
artifact paths and SHA-256 values. Required top-level fields:

```json
{
  "source_manifest_sha256": "SHA of this source-manifest.json",
  "source_diff_sha256": "SHA of source.diff",
  "compiled_source_sha256": {"each root path in source-manifest.json": "actual compiled proposed SHA"},
  "binary": {"path": "/absolute/binary", "sha256": "SHA"},
  "hardware": "actual device/host", "compiler": "actual compiler/version",
  "runtime": "actual SYCL/Level Zero versions", "driver": "actual driver version",
  "build_command": ["actual", "command"], "build_exit_code": 0,
  "build_log": {"path": "/absolute/build-evidence", "sha256": "SHA"},
  "runs": [
    {
      "mode": "conversion",
      "stdout": {"path": "/absolute/stdout", "sha256": "SHA"},
      "exit_code": 0, "owned_process_closed": true, "fresh_fault_count": 0,
      "fault_receipt": {"path": "/absolute/fault-closure", "sha256": "SHA"},
      "command": ["actual-binary", "conversion"],
      "effective_environment": {"actual_runtime_variable": "actual_value"},
      "environment_set_before_exec": true, "boot_id": "actual boot ID",
      "owned_pid": 123, "owned_start_time": "actual", "started": "actual",
      "ended": "actual", "deadline": "actual", "log_budget_bytes": 67108864
    }
  ]
}
```

Supply exactly four run records: conversion/fullgpu/fullhost/mixedstore. Store
records also need `audit` and `audit_raw` objects (`path`, `sha256`) and their
effective environment above. The consumer validates exact byte equality across
placement source rows, every packed/input selected-row memcmp, phase counts,
placement, explicit BF16 dtype/oracles and checked closure metadata. It trusts
Main's recorded compile/supervision facts and fault receipt; it does not discover
or inspect processes or infer those facts from current runtime state.

Research origin: R569/R573/R661/R662 and FINAL R674 contract
`research-bf16-prefill-storage-contract-round674-v1.txt`, SHA-256
`c3fbcaf0d4563e9c1843cc8ef4f8f2612df3afcb7a0e2c4556892189342937a7`.
These support investigating separate BF16/FP16 recurrent storage arms, not a
quality claim from post-hoc final-tensor casts or the author's FP8 result.

Retention: no run artifacts were created here. Preserve this compact proposal,
new fixture/consumer source and identities for Main's next idle review. After
future runs close, retain individual outcomes and exact comparison summaries;
retire successful raw audit rows/log duplicates only under the measurement lock
with a path/hash/reason manifest after the consumer has verified them. Failed
receipts stay failed and representative unresolved diagnostics remain retained.
