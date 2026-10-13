# Importing actual SERVE GEN1 first rows

This adapter imports one closed row qualified on hardware by Main through
`tools/fixed_history_capture_receipt.py`. `tools/fixed_history_logits.py` uses a
separate comparison schema. Existing standalone follow-dump semantics, admission
requiring at least two rows, and its empty `REVIEWED_PRODUCERS` remain unchanged.
[日本語](SERVE_FIRST_ROW_CAPTURE_RECEIPT.ja.md)

The only registration is `phase-gated-gen1-b4f7b5ec-32k`. Original source revision
`6035c4f92bf87362d7e55051ae1375b9befba89e` plus five pinned changed files is now
producer commit `b4f7b5ec1d7583de52f0e0dbcf181d461591cf17`. The retained
`request-2.bin` represents fresh GEN1 with 32768 prompt tokens, position 32767,
one IEEE FP32 little-endian row of 248320 logits, exactly 993288 bytes, SHA256
`d3903d0a0eefc77dbb3493ed413b007104eb365944e67df641684f00d9297c54`.
It imports the original B570 10 GiB / Ryzen 5600X / 128 GiB RAM observation;
it is neither a new hardware measurement nor a quality result.

## Contract

- `strata.serve-first-row-sidecar.v1` requires producer, receipt, publication,
  retention, capture, fixture, source_root, source_commit, request_index,
  model_files, consumer and retire_after. See `tools/serve_first_row_example.json`.
  Pins contain only path and SHA256.
- The registry JSON hash is pinned in adapter source. Each entry binds Main's
  reviewed actual receipt/publication/retirement hashes, commit/tree, hashes of
  283 src/include/cmake/build entries, five source pins, model/provider/binary
  identities, retained row and request index. There is no runtime registration
  option. Candidate P's layer-major producer is unregistered. Adding a producer
  requires actual qualification followed by explicit Main source review and
  registry/hash changes.
- Validate the original immutable receipt's PASS/complete/inactive status,
  normal QUIT without fallback signals, owner exit 0/reap/empty session, READY
  PID/start, complete before/after health and kernel observations, same boot,
  fault filter and counts. Embedded stdout/stderr match original owner log
  hashes. Retired successful raw logs are not required.
- Bind actual raw spec=4, effective T=1, prefill capacity 8192,
  phase/head/scratch/gated, source_row=0, reuse=0 and disabled MTP/lookup/batch.
  COPIED must precede COMPLETE, with one emitted token and a completed DONE.
- Hash the entire 65536-token conditioning fixture, SHA256
  `5284d8fa53d1ef28f6ba916a629faf901c890bc0c669bf986ee04ed72aa2c3ff`.
  Reconstruct the actual `GEN 1 temperature=0 ...` request-packet hash from
  its prefix. The comparison target is frozen fixture[P]. Generated text
  equality and self argmax never decide quality acceptance.
- The native format has no magic. Validate its LE int32 `(vocab, 1)` header,
  exact byte count, all finite vocabulary values and hash. Fixture and row each
  have a 2 MiB limit; JSON has a 16 MiB limit. Read one row and no model tensors.
- Do not require retired request-1 or `.partial` aliases. The selected row must
  not be in retirement deletions and must have a current consumer/review point.
  Retirement does not alter original status. Rehash validated files on return.

## Main's serialized validation

At implementation handoff, these checks are **UNEXECUTED**:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_serve_first_row_capture_receipt.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_fixed_history_capture_receipt.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_fixed_history_logits.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/fixed_history_capture_receipt.py tools/serve_first_row_example.json
PYTHONDONTWRITEBYTECODE=1 python3 tools/fixed_history_logits.py tools/serve_first_row_self_comparison.json
```

CPU fixture source checks bad header/endian/shape/truncated/nonfinite rows,
request position/packet, model/provenance, active/unreaped owners,
fault/missing coverage/boot, consumer, retired capture, unregistered producer
and incorrect authoritative hashes. Main separately imports the actual retained
row under its exclusive slot.

A `strata.serve-first-row-comparison.v1` manifest contains baseline/candidate
sidecar path/hash pins, consumer and retire_after. It reimports both arms and
requires equal entire conditioning identity, numeric argv and effective
environment. Currently a K-versus-K self comparison can verify plumbing only.
A separate candidate still needs actual qualification. Zero self-comparison
deltas are not candidate quality evidence.

Results contain one-row target log probability/NLL, rank, top1/top2 margin and
bidirectional KL. They establish no corpus quality, zero ability loss, 256K
physical extent, RESTORE, speed or adoption. Quality/performance/full-context/
restore flags stay false.

## Current pairing limitation

The implementation requires equality of the entire numeric argv and effective
environment, excluding the capture directory. This contracts the current
K-versus-K check; it does not already admit a future K-versus-P layer-major
comparison. For example, a source-defined implicit default of 700 and an explicit
700 are not a numerical setting change when their effective values are equal.
The current literal equality check cannot normalize that equivalence and may
reject it.

A future candidate comparison needs Main's source-reviewed treatment
normalization and authorized-delta contract. Resolve defaults from each arm's
pinned source and actual effective evidence. Manage LM and RGPU differences as
explicitly controlled treatments defined before comparison, while preserving
conditioning, model/provider and other numerical controls. Arbitrary caller
exclusions of argv/environment fields cannot substitute for that review.

This change implements neither that future contract nor P registration or a
K-versus-P comparison. P's actual model route remains unqualified and no retained
P row exists.

## Main's execution evidence

The UNEXECUTED statement above records the initial source handoff. Subsequently,
Main's CPU v1 receipt recorded passing fixtures and an actual-import rejection,
`qualified effective route`; that failure remains preserved. The cause was the
incorrect name `STRATA_PHASE_PREFILL_GATED_ALIAS`; the actual name is
`STRATA_PREFILL_GATED_ALIAS`. Main corrected it and added a route/wrong-name
regression using the real reviewed receipt. Main's v2 receipt,
`main-serve-first-row-adapter-cpu-root-v2/record.json`, records PASS/complete/inactive
for three CPU suites, actual K-row import and K-versus-K comparison. The author
of this documentation update did not execute those checks. The v2 pass does not
qualify P or establish quality.
