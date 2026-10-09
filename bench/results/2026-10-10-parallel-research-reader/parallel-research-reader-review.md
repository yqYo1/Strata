# Parallel research and root validation, 2026-10-10 JST

Parallel source work continued while root executed the current checks. The
optimization goal remains open. No model/GPU run, reset, service change or new
performance claim occurred at this boundary. The original C full-repeat
numerical rejection remains unchanged.

## Roles and actual overlap

Two Luna agents completed rounds 24–29 with distinct scopes, twelve read-only
reports. They independently reviewed coverage and bounded I/O, investigated
cross-engine QSA and Zen3 candidates, then reviewed new source and qualification
designs. Each assignment shared the prior report directory, registry v23 and
committed root decisions. Root did not intervene before their reports returned.

Sol 6.1 implemented the reader hardening in a separate worktree while root tested
the frozen original reader. After that source returned, root supplied the actual
failure and Sol implemented content verification. Root exclusively executed the
next tests and corrected two further issues. While Luna reviewed the resulting
source, root tested it and integrated the owner-controller reader. Sol then
implemented the next CPU wrapper harness while root ran controller preflight
and archived these results. No implementation agent executed a test or build.

Research, source implementation and root work overlap. All actual builds,
tests, profiling, model runs and cleanup remain serial; the measurement lock
is shared. The CPU preflight envelope delegates that lock to its pinned child,
whose preflight exits before GPU helpers or model launch.

## Root's actual CPU results

| Revision / check | Actual result | Scope |
| --- | --- | --- |
| Original reader `b1d43473`, v1 | 38/39 pass, rejected, 21.738 s | Stale-manifest comparison accepted changed content |
| Sol hardening `4e5684f2`, v2 | 46/48 pass, rejected, 21.452 s | All five stale-content cases pass; two ledger cases fail |
| Root correction `4a565c03`, v3 | 48/48 pass, 22.990 s | CPU reader/framing/GEN coverage/content pairing only |
| Five-GEN controller `188cbb94…`, CPU preflight | Pass, 2.250 s | Source, receipt, historical fixture and invocation pins |

Every child exited normally, v1/v2 with code 1 and v3/preflight with code 0.
No forced cleanup or surviving owned process occurred. Durations include the
supervisor; test-only durations are retained separately in each original
receipt. Original failed receipts and individual cases are byte-identical in
this archive. They are not rewritten into passes.

The original comparator skipped current payloads when old manifest hashes
matched. The fix streams each whole file, including headers, against its parsed
digest before any equality shortcut. Four deterministic stale payload/header
tests simulate unchanged metadata, in addition to the original stale test.
The original run did not record per-case before/after stat values, so timestamp
granularity is not established as the observed mechanism.

The first full synthetic test shared its descriptor generator with the reader.
The corrected fixture independently transcribes all producer hooks and pins
both `prefill.cpp` and the separate wire header. It contains 690 frames and
133,495,872 bytes; this validates CPU framing and pairing, not real top-k/math.

The v2 ledger failures exposed two test/contract problems. A deeply nested array
is valid JSON; root rejects it by requiring the five-GEN ledger schema. A rapid
same-size write need not appear as a metadata change. The identity-change test
now explicitly changes its private fixture timestamp; it does not promise to
detect every hostile concurrent write. Ledger input is private, orderly closed
and owned by the controller. Duplicate keys and bytes paths also reject.

## Controller and model gates

The reviewed v2 controller requires a caller-supplied exact controller SHA256,
pins the passing CPU reader/source/test receipts, and binds the parser's five
intervals to each actual GEN record's before/after byte counts. It parses only
after normal QUIT and the close-error sentinel check. Parser failures preserve
their partial manifest. Incomplete request histories retain framing-only
findings and cannot become complete owned diagnostics.

The reader's structural result, numeric findings and paired captured-window
comparison remain separate. `full_lifecycle_passed`, `performance_eligible` and
`adopted` stay false. The bounded sequence omits clipped-tail, RESTORE, refusal
and later-control checks. Equal instrumented captures cannot clear the original
uninstrumented C rejection; additional waits may mask it. CPU preflight does
not execute the new capture integration. A future model run still requires
fresh health and fault checks, reviewed pins and root's serial ownership.

## Research reconciliation and next implementation

Round25's QSA candidates are conditional key-tile reuse in block scoring and
event-gated overlap of selection with attention. They require phase attribution
and bitwise score/selected-ID/state checks. StreamIndex's public README describes
synthetic, H200-specific experiments and states that memory traffic remains
unchanged; it does not establish a B570 benefit. vLLM's allocator issue describes
a different GPU/allocator; Strata already uses a fixed score allocation.
[StreamIndex](https://github.com/RightNow-AI/StreamIndex),
[vLLM allocation issue](https://github.com/vllm-project/vllm/issues/56457).

The proposed Q2_0 Gate/Up pair does not presently justify implementation for this
pack. Its known GU types are 18/21/22/23; nine Down type42 layers do not establish
GU type42 exposure. The source's Q2 GU route requires a Q2 GU format. Preserve
the report, but prioritize actual current-format dispatch evidence.

Round27's generic parity-target recommendation is corrected by round28: SYCL
currently has no `iq_avx2_parity` target, resolves the common
`src/kernels/cpu/native_expert.cpp`, and compiles the migrated
`sycl/src/kernels/cpu/iq_avx2.cpp`. There is no migrated native-wrapper file.
The generic target alone misses the actual SYCL IQ entry/compiler mapping.
H has no build receipt yet; do not infer one from T's old receipt. Two research
reports contain a duplicated segment in a written registry digest; the actual
v23 digest is `5dfcd721be2d40077ef584264f05ba370c118f5f94aad44c3117fb7733f19e2b`.

Sol's submitted next source is `25d5003253a97aaf6fa710e1e3715c2ecc3b44d3` in
`diag-native-wrapper-parity-20261010`. Root read its complete harness and CMake
delta. It links the actual SYCL CPU library, calls production GU/Down/Q2 paths
for 48 type/NT cases, and emits full canonical output for matched T/H comparison.
It is unbuilt, untested and unadopted. Round29 independently checked its math
and target/output scopes. The empty-interval check reused a buffer with an
allowed interior range, so it could miss an erroneous write. Root corrected it
to use a fresh buffer whose every word must remain sentinel, commit
`87b50497bc4bac7b40113f181ad51b47e64ba7ae`.

Root applied the identical two-file harness/CMake patch to a new frozen-T tree,
commit `61a70650a004d27ec622046b3d4d6499e7012ca0`, without changing T's wrapper.
The CMake, harness and shared fixture source are byte-identical between these
T/H qualification trees. Both branches are committed and pushed; neither new
target has been built or executed. Root next builds the two targets with
matched actual flags, and runs every environment profile in fresh serial
processes. Current dimensions test dispatch/row arguments, not full model shape.
Do not demand that NT1 equal NT2 when their existing accumulation paths differ.
On Zen3, forcing gather does not make `cpu_gather_fast()` true or qualify IQ3S's
special one-token selection.

Registry v28 is an immutable snapshot made just before that harness returned;
its assignment status says running. Registry v29 records its returned source,
root correction, matched-T preparation and actual cleanup. Share v29 and this
report with the next research round. Past reports remain available
under the shared research directory and Git archives; research is iterative.

## Artifact retention

All compact individual results, original failures, commands, source pins and
research reports are committed. The retirement plan names 164 synthetic CPU
fixture/represented-console files, 401,587,140 logical bytes and 402,623,488
unique-inode allocation bytes. Their source construction and original
size/hashes/case outcomes suffice; no raw consumer remains. Root verified
committed/pushed evidence, closed owners and all identities under the exclusive
lock and actually removed all 164 files in 0.403 s. Original receipts remain
byte-identical with their failed/pass statuses. Model/session tensors and real
small producer inputs are outside this plan. The separate result records actual
deletion; available-space change was not measured.

Archived research is preserved byte-identical. The blanket whitespace check
reported only existing trailing blank lines at EOF in three original reports;
the check with `core.whitespace=-blank-at-eof` passed. Root did not rewrite those
reports or their pinned hashes to remove formatting-only warnings.
