# Native16 natural-long sparse precision screen proposal

R802 is an unrun, source-only proposal for Main. The new Python files are original
project code under the repository `LICENSE`. The production reference is immutable
commit `31ad125b` (full ID and inspected-file hashes are in `manifest.json`). The one
explicit working-tree reference is the frozen natural-input preparer, whose SHA256
is also in the manifest. No production source, old verifier, historical receipt or
other worker file is changed. No proposal, fixture, build, GPU job, profiler or
process inspection was run by this worker.

`run_screen.py` launches three fresh native CLI processes in the fixed order
`fp32-a`, `fp32-b`, `bf16`. They use the same qualified v2/v3 binary, exact token
file and base argv/environment. They differ in the two precision controls, private
cache directory and sparse output path. FP32 remains the production default.
Each arm starts from an identical private copy of `data/expert-profile.bin`; the
external file is checked before/after every arm. A private token-file copy is also
checked. There is no server, automatic retry, process adoption, environment mutation
after exec, new GPU implementation, dense NLL observer or output-text equality gate.

The controller takes the existing exclusive owned-measurement lock, SHA-verifies
the qualified `direct_owner.py`, and uses its bounded pidfd/session ownership for
every native and journal process. Native limits are 1800 seconds, 21600 CPU seconds,
88 GiB session RSS, 64 MiB per file and 64 MiB aggregate owner text. The finite job
budget is 384 MiB: three sparse tensors plus the text allowance, local profile,
cache and metadata need margin above 256 MiB. The separate tensor budget remains
256 MiB. There are no signals to foreign processes and no cleanup in this proposal.

Prerequisites and scope

- Main reviews this complete proposal, its provenance and pins, and installs the
  additive files in the actual tree before executing anything.
- Main closes the currently active GPU cohort first and supplies its successful,
  normal, healthy owned receipt with its SHA256 via `--closed-cohort PATH SHA256`.
  This argument is repeatable when several cohorts need closure proof. The lock
  and recorded prior-owner identities are checked; Main retains responsibility for
  excluding unrelated GPU workloads, including a foreign GPU service.
- Main runs the separately prepared natural-input recipe and pins its completed
  `own-native16-natural-input-v1.json`. That receipt must show thinking ON,
  exact official tokenizer/template roundtrip, project-document source hashes,
  no repeated padding, and an actual input **greater than 32768** and below 35000
  tokens. The screen never invokes input preparation or alters conditioning IDs.
- Build, all68 typed-GEMM, model mechanics and regression receipts must be normally
  closed. For v2 only, the exact pinned closed separate-queue failure stays FAIL;
  the separate `regression-followup2` receipt admits only the qualified single-queue
  route. No original gate or receipt is relaxed or rewritten. A v3 failure cannot
  silently reuse the v2 exception.
- The fixed diagnostic argv uses physical max-context 262144, KV-resident 32768,
  prefill chunk 8192, residual GPU rows 32768, max-new 1 and `--no-token-graph`.
  The actual placement banner must match and all48 layers must complete exactly
  `ceil((input_count - 1)/8192)` chunks. Actual prompt IDs and normal generated
  count are checked from the standalone CLI output. These settings do not prove
  physical 262144-boundary correctness.
- The effective environment is the previously qualified model-smoke environment,
  including same-queue 1, `STRATA_TRACE=1` and UR parameter validation with warning
  logging. Values are frozen in a fresh dictionary before each exec; the controller
  never inherits its own process environment. Known runtime/admission settings are
  checked exactly. Timing from these runs is diagnostic and ineligible as a clean
  speed sample.

Source identity is the build receipt's complete `src/`, `include/`, `cmake/`,
`tests/` and root `CMakeLists.txt` hash set, with compiler, binary and runtime-library
SHA256 pins. Advancing commits containing reports alone do not invalidate that
freeze. Model shards retain the established bytes/mtime identity check; this is
explicitly recorded as a stat identity, **not a full-shard cryptographic hash**.
Pack/MTP metadata, tokenizer/template/input sources, external profile and actual
input IDs are SHA-pinned. Qualified toolchain paths may contain a historical
directory name; no source from that other engine is read or adopted here.

Sparse capture and independent comparison

Only `STRATA_PREFILL_LOGITS` and its 64 positions are enabled. Positions are exactly
`255 + 512*i`, `i=0..63`, ending at 32511. Every stored row has all 248320 FP32
vocabulary logits. Each data file is exactly **63569928 bytes**, below 64 MiB;
all three total **190709784 bytes**. Metadata is at most 8192 bytes per arm. The
existing writer's published `.meta`, exact native-endian header, complete footer,
batched count, ordered positions and next-token IDs must match exactly. A
`.meta.partial` alone is incomplete; its expected hard-link coexistence with the
published `.meta` is allowed. All vocabulary values must be finite, including
values unrelated to the target or top1. Each arm is parsed fully before the next
launch, then all three are reopened by the independent comparator.

The sparse native format contains no recipe digest. Its binding comes from the
exclusive fresh arm directory, checked actual Owner argv/environment and log
fingerprints, recipe hash, payload/metadata hashes and common source/input/model
binding hash. This is recorded honestly rather than implying an engine-signed
recipe field. The top-level receipt records the real controller PID/start identity,
Python binary hash, exact `sys.orig_argv`, controller/comparator/manifest hashes and
direct native child identity. Every source, input, binary, runtime and external
profile is rechecked before/after capture. Fresh kernel journal windows, unchanged
boot, no driver dump, all normal module-closed commands and zero fault matches
are mandatory. A failed run retains a failed immutable receipt and fresh root.

`compare_screen.py` decodes FP32 rows into independent Python floats and computes
target NLL with max-shifted logsumexp and `math.fsum`. It records the selected top1
token and tie count, choosing the lowest token ID on an exact tie. It keeps each
of the 64 position/target results and reports:

- `fp32-a` versus `fp32-b` repeat noise, separately from BF16 versus **each** control;
- signed/absolute target-NLL deltas, mean delta and `exp(mean_delta)` as a sampled
  PPL ratio, with explicit overflow representation;
- maximum absolute logit difference, top1 agreement and each arm's sampled NLL;
- by default, full-distribution `KL(Pcontrol || Pcandidate)` and L1, computed with
  shifted FP64 log probabilities and `math.fsum`, without probability floors.

Distribution work is bounded to three pairs per row and constant row memory. Main
may choose `--skip-distributions` explicitly if the additional host cost is not
useful; the omission is recorded. A very small signed negative KL from FP64
rounding is preserved, not hidden by clamping. No exact-text condition, numeric
acceptance tolerance, confidence interval or automatic adoption decision exists.
`passed=true` means capture/closure/parser/comparison validity only; all quality,
clean-speed, RESTORE and physical-context qualification flags remain false.

The unresolved same-source 78443-token FP32 dense-NLL differences are not a BF16
tolerance. This new sparse two-control screen can observe noise on its own input;
it cannot resolve those earlier differences merely by succeeding. One ordered
English/code/document corpus and 64 correlated positions are exploratory evidence,
not general or task-quality proof. Matched BF16 clean speed is not inferred from
this proposal, the synthetic model smoke or the single reported FP32 speed sample.

Main's execution interface

After the GPU cohort closes, Main can review/run the four host cases first:

```sh
python3 -B bench/results/2026-10-15-pr6-recovery-audit/own-native16-quality-screen-v1/hostfixture.py \
  --root /home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-own-native16-quality-host-root-v1
```

The host fixture leaves a compact result in that new root. Its four groups cover
an independently hand-computable uniform distribution with ties and additive
logit shift; zero/wrong headers, truncation and trailing bytes; wrong/duplicate
metadata positions/targets and incomplete publication; and NaN/positive/negative
Inf in a non-target vocabulary entry. It imports no backend and does not claim
full model/controller qualification. No fixture has been run by this worker.

Controller usage is:

```text
python3 -B .../own-native16-quality-screen-v1/run_screen.py
  --version v2
  --run-id v1
  --input-sha256 <Main-verified completed natural-input receipt SHA256>
  --closed-cohort <closed current GPU cohort receipt> <Main-verified SHA256>
  --retention-review-utc <explicit UTC timestamp within the next 48 hours>
```

Angle-bracket values are required inputs for Main, not runnable placeholder values.
The default completed-input path is fixed in the controller. The fresh output root
is `main-own-native16-quality-screen-root-v2-v1` under the private state directory;
the immutable receipt is `A/own-native16-quality-screen-v2-v1.json`. Any rerun,
including a failed one, needs a new `vN` run ID. No failed root is resumed.

v2 reference hashes are fixed in the manifest. To use v3, Main must additionally
provide `--gate-pins PATH --gate-pins-sha256 SHA256`. That reviewed JSON must map
the exact v3 build/GEMM/model/original-regression basenames to completed receipt
hashes. The v3 original regression must pass. The controller still checks all
source/binary/owner/fault/model gates and uses the qualified smoke environment;
the pin file cannot waive any gate. No v3 execution is represented by this proposal.

For an independent later host recomparison, use `compare_screen.py --receipt PATH
--receipt-sha256 SHA256 --output NEW_PATH`. It verifies normal closure and exact
capture/recipe/input hashes and requires the named current input sources. It writes
a new result even for a rejected malformed receipt; the original remains unchanged.
It does not inspect live processes or initialize a runtime. Recomparison requires
that the named captures and current input fixtures are still retained.

Retention and handoff

Main is the raw-data owner. The named current consumer is
`compare_screen.py:compare_record` plus Main's numerical review. Before launch Main
supplies a finite UTC review deadline within 48 hours; review also occurs at screen
closure or supersession, whichever comes first. Keep compact per-position results,
source/command/environment/receipt pins, correctness/fault outcomes and decisions
in Git. Keep tensors outside Git only while that comparison or an explicitly named
numerical failure still consumes them. At closure Main reviews successful full
text/journal and tensor retention under the measurement lock and records any later
deletions in a path/hash/reason manifest. This proposal performs no deletion and
does not modify original statuses or remove required fixtures.

Main still needs to review source, run the proposed host cases, install/commit the
proposal, close the GPU cohort, execute input preparation, pin all actual receipts,
and decide whether to execute this diagnostic. No numerical or performance result
is claimed by this source delivery.
