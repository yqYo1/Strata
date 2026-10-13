<!-- SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Fixed-history capture receipt

[日本語](FIXED_HISTORY_CAPTURE_RECEIPT.ja.md)

`tools/fixed_history_capture_receipt.py sidecar.json` reads evidence from a completed
standalone native-pack `--spec-follow --dump-logits` capture and prints receipt JSON
accepted by `tools/fixed_history_logits.py`. It does not launch processes, operate
the GPU, read model tensors, or measure speed. Save the receipt beside the sidecar
so evidence pins retain their directory base. Point each comparison manifest arm's
receipt pin at the result; common and capture/binary/source/runtime pins must match.

No real production producer is registered for this standalone schema yet. The
separate SERVE GEN1 schema registers Main's reviewed retained K row; that does
not change the standalone rejection boundary below. Even a consistent standalone fixture returns
`production_capture_qualified: false`. This is a deliberate rejection boundary:
hand-entered `passed: true`, an owner source hash, and a command recipe do not
prove execution authenticity. A separate reviewed change must audit and register
the real controller and collectors. `adapter_schema_validated: true` means only
schema and pin consistency. Changing a producer name or evidence_kind to production
does not cross this boundary.

## Input schema

The sidecar schema is `strata.fixed-history-production-sidecar.v1`. Every field
below is required; unlisted fields are rejected. Pins contain only `{path, sha256}`;
SHA256 is 64 lowercase hex characters and relative paths use the sidecar directory.
Duplicate JSON keys and nonfinite JSON numbers are rejected. Each JSON is limited
to 16 MiB and each log to 64 MiB.

- `schema`, nonempty `producer`, `evidence_kind` (`production` or `synthetic-fixture`),
  nonempty `consumer` and `retire_after`.
- `common`: the existing comparison contract's `P N vocab max_context endian stride
  position_convention policy prompt_ids follow_ids prompt_ids_sha256
  follow_ids_sha256 tokenizer model`. Positions use `P-1+i predicts follow[i]`, stride 1.
  Policy is `bos: prompt-explicit`, `eos: score-all-stop-off`, `mask: all-follow`.
  tokenizer is a pin; model pins the small identity manifest below, never tensors.
- `budget`: positive integer `max_vocab max_rows max_capture_bytes`. Capture size is
  `8+4*vocab*N` bytes. Adapter limits are 1 GiB and vocabulary 2097152.
- `owner_source`, `owner_receipt`: pins. Owner source SHA is
  `a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f`.
  The receipt is the saved real owner's object containing its commands array;
  `command_index` is the nonnegative index of this command. Preserve additional
  original owner receipt fields.
- `artifacts`: pins for `capture binary source runtime generator verifier prompt follow
  input_manifest stdout stderr provider_evidence`. Source/runtime are small
  source/build/runtime record envelopes; generator/verifier are actual sources and
  binary is the executable. Model-manifest scope differs from source scope.
- `environment`, `identities`, `fault`, `close`: JSON pins described below.

Environment is the **entire effective environment** string map passed by the owner
to subprocess. It must exactly match the owner receipt, with STRATA_VERIFY_SEGMENTED=1
and STRATA_VERIFY_DEVICE_PLAN=0. LD_PRELOAD and probe/bench overrides are rejected.
Other environment values and build/runtime envelope semantics require future
producer review.

Identities contains `schema: strata.capture-identities.v1` and
`command_sha256 environment_sha256 artifacts model_files pack_path native_path
profile_path boot_before boot_after started_utc finished_utc`.
Command SHA hashes the argv array serialized as UTF-8 compact JSON with separators
`,` and `:`. Environment SHA is the environment pin hash. Times and argv/environment
bind to the owner receipt. Artifacts has every sidecar artifact name with
`{pre_sha256, post_sha256}`. Capture/stdout/stderr pre hashes are null (absent before
launch); all other pre hashes equal post. Post hashes equal artifact pins.
Model_files is a nonempty array of `{path, bytes, pre_sha256, post_sha256}` with
unique absolute paths, positive integer bytes and identical hashes. It must include
native/profile paths. Model manifest is
`{schema: strata.model-identity.v1, files: model_files}`. The real producer must
inventory every tensor/shard, pack metadata and profile. The adapter binds pre/post
identity evidence without rereading huge tensors.

Provider_evidence contains `schema: strata.capture-provider.v1` and
`provider generator_sha256 verifier_sha256 binary_sha256 runtime_sha256 source_sha256
native_pack device_plan segmented clip_scope`. Provider is
`standalone-native-segmented-spec-follow-dump-logits`; native_pack/segmented are true,
device_plan is integer 0, clip_scope is
`full-vocabulary-all-follow-no-position-clipping`. Hashes match artifact pins.
This separates source/native-pack/runtime scope from logits position-selection scope.

Fault contains `schema: strata.capture-fault-health.v1` and
`boot_before boot_after before_utc after_utc command_sha256 kernel_complete health_complete
records health_records kernel_evidence health_evidence`. It requires the same boot
and a UTC interval containing the command, with at most 60 seconds before launch
and after completion. Complete flags are true; records and
health_records are empty arrays matching the JSON arrays pinned by kernel_evidence
and health_evidence. Real collectors must classify all kernel faults and device
health for the interval, rejecting collection failure, missing cursors, boot changes
and faults. Empty arrays and self-declared complete flags do not prove collector
authenticity.

Close contains `schema: strata.capture-close.v1` and
`command_sha256 capture_sha256 bytes closed_after_utc close_error forced canceled
 deadline_hit budget_hit`. Close_error is null and the last four flags are false;
time follows command completion and bytes/hash match the actual capture.

## Validation and remaining real standalone producer work

Owner observation/reap/session-empty/session-anchor must be true, ownership observed,
survivors/errors/cleanup empty arrays and exit integer 0. PID/start-ticks/session
must match registered owner identity. Missing values are never filled with success.

Argv begins with binary generate, with explicit tokens-file/spec-follow/dump-logits,
max-new N, max-context, logits-stride 1, spec >=2, native/pack/profile/cache/prefill,
suffix-draft 0, lookup-chain 0 and adapt-every 0. N must be >=2. Options outside the
supported allowlist, duplicates, serve/MTP/EOS/early-stop/prefill-until are rejected.
Add options only through reviewed changes. Full prompt/follow token files and input
manifest are checked against common.

The actual segmented-init stderr line, complete prompt/output IDs in stdout,
N decode tokens, logits-dumped completion line, and owner log hashes/sizes are checked.
Capture is an 8-byte int32 vocab/rows header followed by row-major full FP32;
exact N/bytes and every finite value are required. Positions and targets are derived
from history rather than trusting supplied success positions. All pins are rehashed
after validation. Fixtures substitute a mock owner hash only inside tests.

The real controller must preserve owner receipts and collect command pre/post hashes,
absent outputs, complete model-file inventory, actual effective environment, same
boot, fresh fault/health intervals and close observations. Review source-to-binary
build binding, runtime resolution, collector completeness and tamper resistance
before registering a producer. Schema checks do not replace this independent audit.

Production capture closure/route provenance, logits numerical comparison, persistent
GDN/QSA/PLE state RESTORE comparison (not implemented), and held-out task quality
(not performed) are distinct outcomes. Quality_qualified remains false, with no
automatic adoption. Exact generated prose equality is not a quality gate. Use only
unchanged upstream bench.sh for speed comparisons. Retain raw captures only for
the declared consumer, byte budget and retire_after period.

Schema fixture command:
`python3 -m unittest discover -s tools -p test_fixed_history_capture_receipt.py`.
Fixture success does not establish GPU capture, production qualification or quality.

Actual closed SERVE GEN1 rows use a separate reviewed one-row schema; see
[SERVE first-row adapter](SERVE_FIRST_ROW_CAPTURE_RECEIPT.md) and
[日本語](SERVE_FIRST_ROW_CAPTURE_RECEIPT.ja.md). Standalone semantics remain unchanged.
