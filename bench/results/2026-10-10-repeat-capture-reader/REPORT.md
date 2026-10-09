# Repeated-prefill capture reader

Status: **SOURCE ONLY / UNTESTED. Reader CPU runtime gate OPEN. GPU admission
remains HELD.** No build, test, syntax/help query, capture-reader invocation,
model, GPU, profiler, service change or cleanup was executed by this agent.
Root owns all serial validation and later capture decisions.

Base: `9c2ebde89e5157c81a9c9ae135719452a0244ca3`, the debug capture producer with
unknown-retirement quarantine. Worktree branch:
`diag/repeat-capture-reader-20261010`. Changes are limited to this directory.
The original gprofng worktree, root controller, direct B files, research registry,
prior receipts and main mirror are unchanged.

The producer contract was read from `sycl/src/prefill/repeat_capture.hpp`, its
actual hooks in `sycl/src/prefill/prefill.cpp`, and QSA definitions in
`include/strata/kernels/qsa.hpp` / `sycl/src/kernels/cuda/qsa.dp.cpp`. Round 22's
controller audit, registry v21 and committed producer report were read. The
existing producer's CPU validation receipt reports 12 ASan/UBSan fake-queue
cases passed; this implementation does not reinterpret that as reader or GPU
validation. The earlier large fake-producer fixture ordered phases differently
from the real hooks and is not an owned-full coverage oracle.

`capture_reader.py` performs no import-time file operations. Its API is:

```python
manifest = read_capture(path, ledger=capture_request_offsets, mode='owned_full',
                        byteorder='little', expected_identity=optional_pinned_identity)
comparison = compare_requests(path, manifest, 'full256k-first',
                              path, manifest, 'full256k-repeat')
```

Use the actual producer architecture's byte order; the default is the reader
host's native order. An external consumer on another architecture must pass the
producer byte order explicitly. `expected_identity` may pin `/proc`-independent
file stat fields such as device/inode from the controller; later comparison
requires the parser's complete identity, including size and change timestamps.

The reader rejects symlinks in every opened path component. The final parent
must be owned and private; the final file must be regular, owned by the effective
user, exactly mode 0600, and have one hard link. The cap is 128 MiB. It uses
64-KiB payload reads and hashes, with at most 690 compact record manifests, and
never puts payload arrays in JSON. Security and malformed framing failures
raise `CaptureError`; its `.manifest` retains completed-record metadata,
findings and the error. File identity must remain unchanged during parsing and
comparison. Root must parse only after orderly producer shutdown and check its
close-error sentinel. A read-only consumer cannot prove historical freshness:
root must retain the producer's O_EXCL creation/provenance and GEN ledger.

The exact framing is 12 native uint64 words plus a zero-padded 32-byte ASCII
phase (128 bytes), then four bytes per element. Validation includes magic,
version, header size, ordinal starting at zero and increasing by one, phase,
type, source layer, signed-field limits, chunk/row extent, multiplication bounds,
shape equality, payload extent, trailing bytes and truncation. Arithmetic is
checked before reading or allocating any length derived from a header.

`owned_full` validates exactly these five contiguous before/after GEN intervals,
starting at zero and ending at file size:

1. `control32k-before` — empty.
2. `resume32k-reference` — empty.
3. `full256k-first` — nonempty.
4. `control32k-between-full-reads` — empty.
5. `full256k-repeat` — nonempty.

Each record must fit completely inside one intended full interval. Request IDs
are assigned from that ledger because the producer embeds no request ID. Each
full requires the exact 345 unique semantic records in actual hook order, for
p0=98304, T=8192, physical rows 98304 through 98335:

- Layers 0–12: R_input, R_post_attention_gdn and R_post_moe, each 32×10240 f32.
- QSA layers 3,7,11: K_quant_input, V_quant_input, indexer_raw, query, q_indexer
  (32 rows, widths 512,512,128,6144,512); then all 32 individual score rows;
  then 32 interleaved steps/selected_ids pairs; then attention_output (32×6144);
  then the post-attention and post-MoE R hooks.
- Score active width is `(position+1)//4+1`; four int32 step values are position,
  position+1, `(position+1)//4`, and 2051. Selected-ID payload width is 2051.

All score rows precede the steps/IDs capture loop; attention_output follows
selection. The actual source places R_input before these QSA hooks and both
post-R phases after them. The source-derived total is 66,747,936 bytes per full
and 133,495,872 bytes for the pair, leaving 721,856 bytes under the cap. These
are source arithmetic, not measurements from this implementation.

`framing` mode accepts known producer phases plus `test` and `float_rows`, and
generic positive dimensions within the chunk. It deliberately does not claim
model coverage or request provenance; use it only for retained small real
producer fixtures and parser tests. Duplicate semantic identity is rejected
within an owned request; repeated identity across the two full requests is
expected. Process-global ordinal remains monotonic across both.

Numerical findings are independent of framing completion. NaN/inf counts,
finite ranges, first raw bad word/physical row/column, step inconsistencies and
selected IDs outside `[0,position]` remain in the manifest. The reader imposes
no selected-ID sort or uniqueness rule. A structurally valid capture may contain
numerical findings and must still be retained for diagnosis. JSON contains raw
bits and string labels for nonfinite values, never nonstandard NaN literals.

The pair comparator matches semantic layer/phase/chunk/row/shape/type identities
and ignores process-global ordinal. It compares payload hashes, then reads
bounded chunks for differing frames and records the first different raw word,
physical row, column, byte offsets, bits, and finite float/int value summary.
Earliest means the first differing captured hook in the left manifest; it is
not proof of the first changed operation outside this fixed window. Equal
instrumented captures never clear the original C full-repeat numerical rejection
`a3413ac4c3f20ae593d8507a7db0ac405cd7fab2de286547460e6ba82b0d9404`.
Synchronous waits/readbacks can mask the original failure. Adoption,
performance eligibility and full-lifecycle qualification remain false.

Root integration must catch `CaptureError`, save its partial manifest, and leave
`diagnostic_complete=False`; successful framing sets only structural completion.
Save numerical findings and pair results separately from that gate. Match the
ledger against the controller's independently recorded five request names and
before/after byte counts. Freeze and assert reviewed controller + reader hashes
at the launch boundary after root's source review and CPU validation. This
source does not alter or pin the existing B controller automatically.

`test_capture_reader.py` is a root-run CPU test source, with a fresh private output
directory and no fixture deletion. It tests malformed magic/version/header/
ordinal, forged lengths and overflow, type/layer/phase/row/shape errors, malformed
padding, record-count bound, truncated header/payload, trailing bytes, duplicate
and missing owned-stream coverage, wrong coverage/order, ledger gaps/order/
extra intervals/boundary splits, symlinks/hardlinks/nonprivate files/parents,
NaN preservation even on a later framing error, invalid steps/IDs, stale pinned
identity, finite/int first differences, chunk-boundary NaN raw bits, and a full
690-record fixture in actual hook order. Repeated zero selected IDs in that
fixture explicitly check that no uniqueness requirement was invented. The full
pair also checks that different global ordinals do not imply payload inequality.

Root execution recipe, under its existing exclusive measurement lock and finite
CPU supervisor, with an unused B output path:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 /absolute/source-directory/test_capture_reader.py \
  --out /absolute/B/repeat-capture-reader-cpu-validation-v1 \
  --producer-source /absolute/producer/sycl/src/prefill/prefill.cpp \
  --real-normal /absolute/B/repeat-capture-cpu-validation-v2/normal.bin \
  --real-partial /absolute/B/repeat-capture-cpu-validation-v2/partial-write.bin
```

Root must substitute the actual retained fixture paths from their receipts.
The optional real fixtures are read-only and tested in framing mode. Keep the
reader and test side by side. The test records individual outcomes, source
hashes, Python/uname/boot identity and fixture hashes. Root's wrapper owns the
lock, deadline, total byte supervision and owned process cleanup; this test is
not a second concurrent runner. Its planned fixture budget is 140 MiB, including
one 133,495,872-byte full pair and small malformed fixtures. No command above
was executed here. Failed cases retain original failure status and evidence.

Retention owner: root reader CPU validation / repeated-prefill capture
investigation. Current consumers: parser admission, framing and first captured
intermediate comparison. Keep large fixtures/captures outside Git. Next review:
root closes validation, commits compact evidence, then retires successful bulky
fixtures under the shared lock with a path/hash/reason manifest. This source
performs no cleanup. A future valid parser does not itself authorize a model
run or remove round 22's separate controller digest/close/lifecycle requirements.


Round-24 hardening follow-up (source only, untested)

This isolated implementation branches from reader commit
`b1d4347305b1f840f15cdf51cfaedde3d3d9a6ee`. The original reader worktree is
unchanged. No build, test, syntax/help/version invocation, GPU/model/profiler or
service work, cleanup, or push was performed by this implementation agent.
Root exclusively owns serial execution and review. Source-writing Python was
used only to edit these three scoped files; it did not import or run them.

The shared descriptor-relative private-file opener now accepts a byte bound.
The CLI ledger uses that opener with 65,536 bytes, reads at most 65,537 bytes
in total, checks descriptor identity again and verifies the read length against
its initial size before UTF-8/JSON decoding. Every opened path component rejects
symlinks; the final parent is owned/private and the final file is owned 0600,
regular and single-link. Empty and root paths raise CaptureError; read_capture
preserves an incomplete manifest. Ledger size and each interval now explicitly
require integer values and `0 <= begin <= end <= size <= MAX_BYTES`; booleans
are rejected.

The full synthetic fixture no longer calls reader.expected_full_records or
uses its phase-width table, magic or header-field ordering. It separately
transcribes the actual producer hooks and checks all 345 descriptors against
the reader. Its producer reference is HEAD
`9c2ebde89e5157c81a9c9ae135719452a0244ca3`, prefill.cpp SHA-256
`acd062d1a4f4ab29230630e066fbc29080f92c270cb1fc3e900be12502a17b56`.
Root must add `--producer-source /absolute/producer/sycl/src/prefill/prefill.cpp`
to the execution recipe above. The test refuses to generate the full fixture
unless that source hash matches. This is a pinned static reference, requiring
fresh independent review when producer geometry changes; it does not execute
the model or prove device behavior. Source arithmetic remains 66,747,936 bytes
per full and 133,495,872 for the pair.

New test source covers empty/root paths, negative/huge/noninteger offsets in
every ledger slot and invalid sizes, exact 64-KiB JSON, oversized/invalid UTF-8/
malformed JSON ledgers, final and parent symlinks, hardlinks, directory/FIFO,
nonprivate files/parents, deterministic growth after initial fstat and a
same-size mutation during ledger reading. Mutation injection is confined to
root's CPU test via a scoped os.read mock; it verifies the 64-KiB-plus-one read
budget and stable-fstat rejection. No runtime result is asserted here.

Reviewed evidence: repeat-capture R15/R19/R20/R22 and the two reader R24 audit
reports, plus registry v23 SHA-256
`5dfcd721be2d40077ef584264f05ba370c118f5f94aad44c3117fb7733f19e2b`.
The original C rejection remains unchanged, math gate false, rejection-cleared
false, adoption false, performance eligibility false and full lifecycle false.
No parser result authorizes model admission. No artifact was deleted.
