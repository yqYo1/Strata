Source-only proposal to restrict the independent SYCL selector experiments to
dedicated layer-major long prefill. All compilation, functional, GPU, model,
fault and performance qualification is **UNRUN**. Main owns review, application,
qualification, retention cleanup and Git integration. No active source, runtime
environment, services, or Git state were changed by this package.

The base is the four actual working files captured at
`a276e658f11fd4419d60d28c5d477364a56f52bf`. `manifest.json` records each byte
count and SHA256. `prepare.py` recreates `files/` and `source.diff` using only
those pinned `base/` bytes, rejecting a base hash mismatch. It does not read
live source, apply the patch, run qualification, or access the GPU. Review the
complete diff and compare live bytes against the base manifest before applying;
do not replace changed live files from these snapshots.

The proposal changes four files:

- `src/kernels/sycl/qsa_select.cpp`: both selector entrypoints share the existing
  geometry, capacity and span validation. `qsa_block_topk` directly submits
  `submit_topk<0>` and reads neither experiment flag. `_ref` continues to call
  that original entrypoint. `qsa_block_topk_prefill` alone parses the existing
  default-off radix4/cyclic flags and dispatches Mapping0/1/2. The complete
  `submit_topk` template definition, lambda and specializations are unchanged.
- `include/strata/kernels/qsa_select.hpp`: declares the dedicated entrypoint
  only under `STRATA_ENABLE_SYCL` and documents its environment contract.
- `src/prefill/prefill.cpp`: both `Prefill::run_impl` selector sites (production
  and optional FP16-key comparison) use the dedicated entrypoint only when
  `m.layer_major_active`, under the SYCL preprocessor guard. Conventional short
  prefill uses the original API. `src/core/layer.cpp` decode and
  `src/core/verify.cpp` remain original-API callers. Other backends retain their
  original calls. CMake is unchanged; the existing SYCL runtime target exports
  `STRATA_ENABLE_SYCL=1` in `cmake/sycl_backend.cmake`.
- `tests/sycl/qsa_select_radix.cpp`: the existing 17 batches / 674 normal queries
  and 256-query benchmark call the dedicated API. Every batch also validates
  original and `_ref` outputs against the full CPU oracle, including guards,
  tail sentinels, range, ordering and input immutability, outside the timed
  interval. The public API result is also compared with the dedicated result.
  `_ref` has no active-block argument, so its oracle explicitly includes the
  otherwise valid row rejected only by the smaller active bound. Host geometry
  and span rejection checks exercise both original and dedicated APIs. No
  process-environment writes were introduced.

Environment flags retain their existing exact-`1` interpretation. Both enabled
is an explicit `invalid_argument` from the dedicated API. Set both arm values
before each native child exec; keep process environment immutable after runtime
or provider initialization. Run separate owned processes for `(0,0)`, `(1,0)`,
and `(0,1)`, using the existing supervisor, fault checks and serial experiment
lock. A separate `(1,1)` rejection check is pending. All old receipts and their
statuses remain unchanged. Previous model runs with the old global dispatch
remain mixed prefill/decode diagnostics; this package establishes no new result
and makes no inference or performance adoption claim.

The fixture CLI remains `--dump PATH` or
`--bench varied|equal REPEATS --dump PATH`. Only dedicated API submission plus
drain is timed. Original/reference validation occurs after that interval, so it
does not warm their Mapping0 kernel before a candidate benchmark. Normal output
still contains exactly 17 records and 5,450,408 bytes; benchmark output still
contains one record and 2,100,360 bytes. These sizes are source-derived, not
measured. There are no extra raw original/reference dumps, new capture files or
retired artifacts. The existing selected-output dump remains the single exact
cross-arm comparison payload. Main should review retained fixtures after its
comparison consumer closes, under `docs/ARTIFACT_RETENTION.md`.

Source origin is the current personal fork and independent host-dispatch/test
work. No XeStrata implementation or protected contract record was read, copied,
adapted, or used. No new upstream check is claimed.
