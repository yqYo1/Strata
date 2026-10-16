# Own dense initial-prefill NLL package v3 — engine/file format v2, all validation UNRUN

Owner: Main, serialized provenance/full-patch review, integration/build/tests/GPU qualification. Consumer: paired FP32 repeats and FP16 residual-storage quality at the actual 78,453-token history, then a separately designed physical-context gate. This followup only prepared new package metadata/source snapshots and compared bytes; it did not execute prepare.py, apply/check the patch, build, run tests, start model/runtime/GPU work, change environment/services, clean artifacts or mutate Git. Original v2 and active product/report files remain untouched. No performance, numerical, admission, quality, fault-health or physical-context qualification is claimed.

The base is actual `f27189d626f4a428982d25bb07f441b52345784c` plus the fully hashed actual source snapshot in `manifest.json`. `base/` preserves all 333 current files under `src/`, `include/`, `cmake/` and root `CMakeLists.txt` (5,189,667 bytes), plus a byte-identical copy of root `LICENSE`. It is the necessary immutable patch/rebase consumer, not a progress backup. `files/` contains only the same six proposed changed/new files as v2. `source.diff` is generated against these exact current base bytes. No active product file was edited. Do not replace live files from this snapshot; Main must review/apply the patch after checking the actual base, or explicitly rebase it.

This package reuses the same agent-authored v2 implementation. Its substantive design references include the current own-production selected-logit writer/callback (`include/strata/program/prefill_logits.hpp` and `src/program/generate.cpp`), existing GR/native head interfaces, final-prefill callbacks and checked drain lease. It is **not a clean-room implementation or independent of that own-production design**. The selected-logit header was introduced by own fork commit `f5903fdcddc2e86fd88b75d2a04ac276c7f62a1c`; the existing drain helper by `332cc37242fb9456dbb3dcee7437945fd1605a6d`. `provenance.json` records current hashes, introduction/last-change metadata and exact design dependencies, including upstream foundations with later fork changes. Local Git metadata and Main's own-source identification are recorded with their limits; this is not blanket historical derivation clearance.

No external implementation/template, XeStrata implementation, or old NLL v1 implementation was consulted or reused. No removed helper or ledger was revived. V2 inspected only old v1 filenames and an initial compact part of its artifact manifest, not its source diff, fixture, generator, helper or base/proposed mirrors. This followup did not open v1 or XeStrata source or derived proposals. The original v2 stays byte-identical and retained for Main review.

V2's five newly authored files mistakenly reused the project-only skill's SPDX metadata convention: `SPDX-FileCopyrightText: 2026 yayoi` and `SPDX-License-Identifier: LGPL-3.0-or-later`. Root `LICENSE` had not been checked at that time. The skill supplied those metadata strings, **no implementation or implementation template**. Actual root `LICENSE` is MIT, SHA256 `949b9b0aada7476006469e58ee4ffb0e0f68c2d9137e90a356757b61a09d37f0`; its full notice is preserved in `base/LICENSE`. Under Main's explicit followup and current AGENTS, v3 changes only those five license identifiers to MIT and preserves the copyright metadata verbatim. That author metadata came from the prior convention, not an independent determination of copyright ownership. Correcting a license identifier is not asserted to remove or resolve a hidden derivation.

The normalized comparison in `provenance.json` covers all six proposed source/test/tool files, `prepare.py`, and the complete patch. Replacing only the old SPDX identifier in v2 makes them byte-identical to v3. `src/program/generate.cpp` and `cmake/sycl_backend.cmake` are byte-identical proposals between v2/v3; their current actual base files are also byte-identical to v2's base, so no exact-base context adjustment was necessary. Engine/raw/recipe/closure/comparator schemas remain **v2**. No feature, optimization or code fix was added.

The current selector scope fix is preserved. Three files changed within the 333-file snapshot since v2: `src/kernels/sycl/qsa_select.cpp`, `include/strata/kernels/qsa_select.hpp`, and `src/prefill/prefill.cpp`. V3 captures their current f27189d6 bytes and has no patch hunks for them. The fourth scope file, `tests/sycl/qsa_select_radix.cpp`, lies outside that production-source inventory; its current hash is separately pinned in `provenance.json` and the patch leaves it untouched. Original v2 patch SHA256 is `c4febed46b321c782ec0ea08f55347c6846a713c83baf1b5c8cfa87f0c022079`; original v2 manifest SHA256 is `fb7d55fd3a2b3c0786c8728ffe4e4323816a5f7cd8e564355a097f3b60767d74`.

The configured SYCL resolver is `CMakeLists.txt` -> `cmake/sycl_backend.cmake` -> target `strata`, whose entrypoint is actual `src/program/generate.cpp`; it links `strata_prefill` built from `src/prefill/prefill.cpp`. This proposal adds a small host reduction translation unit to that target and a host-only fixture target. It does not import a root CUDA generator or claim CUDA/HIP validation.

## Exact hook and bounded cost

`STRATA_PREFILL_NLL=<exclusive-output-path>` is absent by default. `STRATA_PREFILL_NLL_RECIPE_SHA256` must be the lowercase SHA256 of the external immutable pre-launch recipe JSON. The two variables are read once at startup. An orphan digest, empty output, unsupported backend/route or malformed digest is rejected before GPU preflight or pending submissions. NLL supports explicit 2..262144 token IDs, a native pack, `--native`, `--stream-token`, positive `--prefill`, standalone single GPU, and the entire initial prefill. It rejects serve, vision, split/peer/remote/GPU-stages, graph-only/GPU-floor, partial `--prefill-until`, coupled sampling, `STRATA_LOGPOS`, enabled head release and enabled phase audit. Existing selected-logit-only behavior and default inference remain available.

The new callback composes with existing selected-logit/MTP callbacks at the standalone `Prefill::on_chunk` hook. In layer-major mode, the existing engine forwards only the final model layer's FP32 residual rows in increasing chunk order; non-final FP16 pack/store/restore behavior is unchanged. Each row is projected using the existing production GR/head path, by borrowing a local `BlockBuffers` view of that residual. The persistent residual/session/KV/recurrent state is not replaced. The same already allocated single-row `d_logits`, native head activation scratch and block GR scratch are reused.

This is deliberately a **bounded sequential diagnostic**. `NativeHead::run` is the production single-row API; there is no newly qualified production batched head method to use here. No full-vocabulary history matrix or additional device allocation is introduced. One host FP32 vocabulary row is allocated before prefill, capped at 262144 vocabulary entries (1 MiB; the current 248320-vocabulary row is 993280 bytes). Every head/readback submission has the existing checked drain lease: partial submission, early return and exception drain before borrowed residual/host storage can unwind; uncertain drain fail-stops with owners live. Each successful row drains before CPU math or I/O and before the next row or previous selected/MTP callback can reuse scratch. The host row lives until after the callback and `Prefill::run` unwind on an error. Native dense/canonical bank, MTP expert, head, graph retirement, restore and final callback boundaries retain their existing ordering.

At 78,453 input IDs this runs **78,452 full head projections** and transfers 77,924,802,560 bytes of vocabulary logits cumulatively. A 78,452×248320 FP32 matrix would occupy that entire amount at once; this proposal never allocates it. It rereads production head weights once per row and adds one checked synchronization and a CPU full-vocabulary scan/reduction per row. That can be very slow. `callback_ms` includes these head submissions, waits, readbacks, CPU reduction and row writes; it excludes the previously installed selected/MTP callback, startup, and the rest of prefill. The ordinary reported PP interval also contains this added work. These runs are quality diagnostics and are **excluded from every clean performance cohort**. No speed claim follows.

The strict CPU FP64 reduction rejects any non-finite FP32 logit, computes max-shifted exponentials with compensated summation, and evaluates `(max-target)+log(sum)` to avoid cancellation of a common offset. Its translation unit alone gets `-fno-fast-math -ffp-contract=off`; normal generator/kernel arithmetic flags are unchanged. The host fixtures compare against an independently written CPU FP64 online log-add-exp recurrence and analytic uniform-distribution/extreme-range examples. This checks loss reduction, not independent truth of the model's head operators.

## Output and failure contract

The raw file has a 64-byte little-endian header followed by exactly `n_input-1` records of `{uint32 target_id; IEEE754 binary64 nll}`. Position is implicit record index `p`; target must equal the exact input ID at `p+1`. Header token0 plus every target reconstruct all input IDs without a lossy hash. Header fields are magic `STNLLV2\n`, version 2, header bytes 64, input count, vocabulary count, completed rows, token0, head type, native-head flag, and 24 zero reserved bytes. The completed row count stays zero until the complete outer prefill/native-bank/MTP restore returns successfully.

Every callback must be contiguous and its rows fully consumed before the next callback; append rejects missing/duplicate/out-of-order positions, wrong targets and non-finite logits/losses. Completion requires all rows. Raw creation and partial metadata creation are exclusive. Raw and metadata flush/close must succeed; only then a no-clobber hard link publishes `<path>.meta`. A `.meta.partial` file (even with a footer) or raw file without final `.meta` is incomplete. Failed writes, close/flush/seek or publication leave no newly published success receipt. An existing final `.meta` is never overwritten. No diagnostic failure is silently converted into success or normal inference. External supervision must also require healthy process closure before accepting a file pair.

Maximum raw size is `64+12*262143 = 3145780` bytes. Metadata is below 4096 bytes; their sum is strictly below 4 MiB, including the exact token identity sequence. At current input the raw file is 941488 bytes. No logits history, tensor dump or session capture is retained by NLL. Optional selected-logit output has its own existing budget and must be budgeted separately by Main; it is not needed for the first dense gate.

`dense_prefill_nll_host_test` covers independent/analytic FP64 references; equal large-offset and extreme finite FP32 values; wide 248320-vocabulary reduction; NaN/Inf rejection; exact little-endian records; full 262144-input representation and strict raw budget; multi-chunk and partial-final-chunk coverage; skipped/overlapping/unconsumed/repeated ranges; wrong gold IDs; incomplete/double finish; exclusive-create/publication collision; and injected header/record/count/footer write, seek, flush and close failures. All tests are **UNRUN**. No new GPU fixture is claimed to have executed; Main must qualify the actual integration and callback coexistence on hardware.

## External receipts and comparison

`tools/compare_dense_prefill_nll.py` is a new source-only comparator, not a launcher. Each independent arm directory must contain `recipe.json`, `closed.json`, `nll.bin`, and `nll.bin.meta`. Main's supervisor must record real evidence; field assertions in JSON cannot manufacture process ownership, loaded-library identity, healthy driver state or actual plan admission. Never use dummy closure values to turn a run into a pass. The engine binds its output to the pre-launch recipe SHA256; closure hashes bind that recipe and the exact raw/final metadata bytes.

Required pre-launch recipe schema (actual values replace the descriptive values):

```json
{
  "schema": "strata-dense-prefill-nll-recipe-v2",
  "source": {"commit": "40 lowercase hex", "snapshot_sha256": "64 lowercase hex", "binary_sha256": "64 lowercase hex"},
  "model": {"manifest_sha256": "SHA256 of manifest pinning all model/pack/PLE/MTP/embedding inputs"},
  "input": {"count": 78453, "file_sha256": "exact token-file byte SHA256", "ids_le_u32_sha256": "SHA256 of all input IDs packed LE uint32", "template_sha256": "template/tokenization identity SHA256"},
  "runtime": {"hardware": "actual hardware", "compiler": "actual compiler", "runtime": "actual loaded runtime/libraries", "driver": "actual loaded driver"},
  "command": ["absolute newly built strata", "--pack", "actual pack", "--native", "actual shard", "--stream-token", "--tokens-file", "actual current token file", "--prefill", "8192", "--max-context", "actual context", "other unchanged actual arguments"],
  "environment": {"STRATA_PREFILL_NLL": "/absolute/arm-directory/nll.bin", "STRATA_PREFILL_LAYER_MAJOR": "1", "STRATA_PREFILL_LAYER_MAJOR_R_DTYPE": "fp32", "STRATA_PREFILL_LAYER_MAJOR_R_GPU": "32768", "STRATA_PREFILL_RELEASE_HEADS": "0", "STRATA_PREFILL_PHASE_AUDIT": "0"},
  "settings": {"dtype": "fp32", "residual_gpu_rows": 32768, "chunk": 8192, "context": 0, "kv": "actual unchanged KV setting", "mtp_manifest_sha256": "actual MTP manifest SHA256"}
}
```

The actual `environment` must include all effective child experiment/provider/driver/profiler settings, not only the example subset. Set these in the native child before exec and keep them immutable. The derived `STRATA_PREFILL_NLL_RECIPE_SHA256` is the only extra field added after hashing; do not include it in its own recipe. Precreate/hash one recipe per run with a unique absolute output. Command, all model/input identities, runtime, effective environment and settings must otherwise match across arms except output path, dtype and the explicitly declared placement axis. Do not put credentials or the prohibited private post-reboot record into receipts.

Required post-close external schema:

```json
{
  "schema": "strata-dense-prefill-nll-closed-v2",
  "recipe_sha256": "actual recipe bytes SHA256",
  "raw_sha256": "actual nll.bin SHA256",
  "metadata_sha256": "actual final nll.bin.meta SHA256",
  "process": {"boot_id": "actual boot", "pid": 1, "start_ticks": 1, "deadline_utc": "actual finite deadline", "exit_code": 0, "owned_absent": true},
  "faults": {"fresh_boundaries": true, "count": 0, "receipt_sha256": "fresh kernel-journal boundary/fault evidence SHA256"},
  "observed": {"n_input": 78453, "rows": 78452, "dtype": "fp32", "chunk": 8192, "context": 0, "residual_gpu_rows": 32768, "residual_host_bytes": 0, "residual_staging_bytes": 0, "head_release": 0}
}
```

Every observed field must come from the actual emitted plan/request and supervised closure; the zero context/host-byte examples are placeholders, never admissible current receipts. The comparator rejects a plan differing from the requested prefix/chunk/context, lost coverage, any gold-ID mismatch, incomplete or oversized output, missing identity, mismatched output hashes, unhealthy/faulted closure, repeated process identity and differences outside declared axes. It requires at least two independent FP32 controls and one FP16 candidate. It reports all-row mean/delta distribution, FP32 pair repeat spread, FP16 versus each control, early/middle/late thirds, chunk-edge rows and the actual mixed-placement boundary.

The comparator intentionally emits `COMPARED; QUALITY_ACCEPTANCE_UNSET`: no numerical acceptance margin has yet been established. Preregister the quality margin against FP32 repeatability before inspecting FP16 outcomes. Add fixed scored task/retrieval outcomes later. Identical generated token sequences are not an acceptance rule.

## Proposed Main commands — none executed, not ready against today's binary

When Main's exclusive work boundary permits source/build/host tests, review provenance, the complete patch and preserved features. These remain proposed commands and must not run concurrently with other GPU work:

```sh
python3 bench/results/2026-10-15-pr6-recovery-audit/own-dense-prefill-nll-v3/prepare.py --check-root "$PWD" --output /tmp/strata-own-nll-v3-review
# Review source.diff; integrate only after exact-base review or explicit rebase.
git apply --check bench/results/2026-10-15-pr6-recovery-audit/own-dense-prefill-nll-v3/source.diff
# Main owns application and source integration; prepare.py never patches live files.
```

Main then reconfigures the reviewed configured SYCL build, builds `strata` and `dense_prefill_nll_host_test`, runs the host fixture serially, and reviews compiled flags/source/binary hashes before any model run. For example, **only after applying/reviewing the proposal**, with `BUILD` set to Main's actual configured SYCL build:

```sh
cmake -S . -B "$BUILD" -DSTRATA_SYCL_TESTS=ON
cmake --build "$BUILD" --target strata dense_prefill_nll_host_test -j2
ctest --test-dir "$BUILD" -R '^dense_prefill_nll_host$' --output-on-failure
```

Native CLI arguments below exist in the current source. The NLL environment feature will exist only in the newly built integrated binary. Replace all paths and settings with the frozen current-input recipe and launch through Main's bounded supervisor, with a finite deadline, owned-process receipt, fresh fault boundaries and the <4 MiB NLL artifact budget. This sketch omits the supervisor and is not a standalone ready runner:

```sh
env STRATA_PREFILL_NLL="$RUN/nll.bin" \
    STRATA_PREFILL_NLL_RECIPE_SHA256="$RECIPE_SHA256" \
    STRATA_PREFILL_LAYER_MAJOR=1 \
    STRATA_PREFILL_LAYER_MAJOR_R_DTYPE=fp32 \
    STRATA_PREFILL_LAYER_MAJOR_R_GPU=32768 \
    STRATA_PREFILL_RELEASE_HEADS=0 STRATA_PREFILL_PHASE_AUDIT=0 \
    "$BUILD/strata" --pack "$PACK" --native "$SHARD1" --ple-gguf "$PLE" \
    --mtp "$MTP" --spec 4 --stream-token --no-token-graph \
    --tokens-file "$CURRENT_TOKENS" --prefill 8192 --max-context "$CONTEXT" --max-new 1
```

Retain the remaining current arm settings/flags verbatim in the recipe and child environment: native/canonical dense bank, Direct, prompt mapping, selected selector and all other model/runtime controls. First run two fresh FP32 controls with the exact 78,453 current input IDs. Next FP16 with the same actual 32768 GPU-row prefix/tail; read the plan, including FP16 staging. Compare the completed independent receipts:

```sh
python3 tools/compare_dense_prefill_nll.py --tokens-file "$CURRENT_TOKENS" \
  --control "$FP32_A" --control "$FP32_B" --candidate "$FP16_MATCHED" \
  --placement matched --output "$MATCHED_COMPARISON"
```

Only after matched quality/health review, separately request FP16 `STRATA_PREFILL_LAYER_MAJOR_R_GPU=78452`. Require actual `residual_gpu_rows=78452`, `residual_host_bytes=0`, `residual_staging_bytes=0`; reject fallback/rejection or any other placement. Use `--placement all-gpu` for that independent placement experiment. Its diagnostic wall time is not a clean prefill throughput sample or a dtype-only comparison.

The existing `check_selected_prefill_logits.py long` is historical and **not runnable unchanged as a current-source runner**: its `current-diagnostics-build-v1.json` pins 30 stale files, in addition to its old 34,603-token/19-position fixture. Do not use it as the current gate. Any sparse KL/top-two/top-one followup needs fresh source/binary/current-input receipts of its own.

262144-input representation in the host writer is not a physical-262144-context correctness or RESTORE result. The existing CLI reserves positive `--max-new` slots (`context >= input + max_new`); a physical 262144 context with `--max-new 1` therefore has at most 262143 input IDs. Design and record the full physical boundary/RESTORE consumer separately without silently increasing its physical context. This proposal does not change that CLI contract or claim admission at either boundary.

At the next review Main must decide provenance and review the complete patch before any application/build/tests/GPU action, then integrate accepted source/tests/tooling and compact true outcomes. The original v2 remains untouched until Main replaces its review consumer under retention policy. No runtime artifacts were created or retired here; no deletion manifest is needed. No new Niko upstream check is claimed by this source-only proposal. Byte/hash normalization is source metadata evidence, not a build, unit test or numerical/functional result.
