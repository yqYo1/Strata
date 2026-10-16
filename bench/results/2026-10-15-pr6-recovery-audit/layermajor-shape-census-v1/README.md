# Layer-major exact shape census proposal v1

Status: SOURCE PROPOSAL ONLY. No product file changed. No compiler, fixture,
model, GPU, profiler, service, process-control or Git command was run. The active
Main-owned timing run was not inspected or touched. All validation is NOT RUN.
Project-only strata-upstream-sync instructions, AGENTS.md, ARTIFACT_RETENTION.md,
and R620 actual-source audit were read. This directory is the entire handoff.

`generate_proposal.py` reads the two current product files, pins their SHA256,
and produces `source.diff` plus proposed-source hashes without copying product
files or applying the patch. `layermajor_shape_census.hpp.in` is the actual
proposed standalone C++ header consumed by `cpu_fixture.cpp`. Regeneration was
performed as source generation, not as a test. Main must compare base pins before
applying; regeneration against changed source creates a new review boundary.

## Scope and data contract

New exact opt-in: `STRATA_PREFILL_LAYER_MAJOR_SHAPE_CENSUS=1`, read by outer
`run_layer_major` before allocation. Absence leaves only null-pointer branches
and a null borrowed-pointer lease; no observer allocation, formatting or sync.
The old `STRATA_PREFILL_ROUTED_FP16_OBSERVE` rejection stays intact. This new
flag has no collection contract for short/chunk-major requests. A collector must
reject absent, partial or unqualified records, rather than treat silence as a
zero histogram. No environment mutation is added.

48 layers retain their own format pair and uint64[8193] count histogram until
outer success. Histogram payload alone is 48*8193*8 = 3,146,112 bytes. Two active
layer uint64[8193] expected-call debts use another 131,088 bytes, plus small
per-layer metadata. A header static_assert bounds the entire owner by 4 MiB.
There is no per-call allocation/formatting, tensor copy, device event, queue
operation or wait. Existing drains are unchanged. Optional allocation and memory
admission failure disables qualification with a small failure record and leaves
inference running; arithmetic, geometry and conservation failures similarly
latch an observer failure without replacing inference errors or return status.
Existing inference failure paths and lease retirement remain authoritative.

Authoritative `m.cnt` increments a layer's expected GU/Down debts by exact ne.
Distinct hooks immediately before each actual `Gemm::f16` invocation consume
these debts and separately increment actual GU/Down call and row counters.
Extra/wrong-ne calls fail immediately; missing calls fail at layer retirement.
After exact equality is proven, sparse histogram entries serialize as
`[ne, routed_experts, actual_verified_gu_calls, actual_verified_down_calls]`.
The latter two equal the second field for ne>0 and are zero for ne=0. They are
not inferred by joining a historical global census to a format table: the format
pair and all counts belong to the same current layer and current outer request.
A debt is aggregate by exact ne, not an expert-identity trace; it does not prove
that two equally sized experts were individually issued exactly once.

Each layer verifies mass == 512*actual_chunks, routed rows == n*10, all observed
chunk sizes including the final tail, exact-ne GU/Down debts == zero, nonzero
mass == actual GU/Down calls, weighted rows == both actual row totals, and all
selected configured slots resident. Logical GEMM dimensions are
GU [ne,1280,2560], Down [ne,2560,640]. Native stored row strides, offsets, total
blob bytes and GU/Down tensor row extents are serialized directly from that
layer's NativeFmt. No oneMKL internal padded tile or hardware cache/DRAM traffic
claim is made. `configured_slots=512/configured_resident=512` means the published
temporary layer cache and residency map, not a hardware cache hit rate.

Request, position and chunk arithmetic stays int64; admits the original full
262144 token physical context and rejects positive position that exceeds it.
Histogram capacity is 8193 and expert-ne is bounded by actual chunk <=8192.
Counter increments and weighted histogram multiplication check overflow.

## Issuer and lifetime proof

Actual SYCL resolver: cmake/sycl_backend.cmake:111 compiles
src/prefill/prefill.cpp, which includes layer_major.inc at :1528. R620's
actual issuer/count locations were inspected in these sources, rather than the
root CUDA generate implementation. Additional read-source pins are in
`evidence-pins.json`.

Current `stream_all` is explicitly `!m.layer_major_active && ...` at
prefill.cpp:1793. `threaded_issue` is `stream_all && issuer_on` at :1918.
Thus no copy-issuer worker starts in layer-major. `StagerDone` is null for LM
(:2729), resident map absence fails rather than starting staging (:2734), and
compute/count calls execute inline in run_impl on the outer caller. A defensive
threaded_issue guard declines the observation if that condition becomes true;
this proposal does not support a future threaded compute issuer. External
concurrent access to one Prefill object remains unsupported; this diagnostic
does not add a concurrency API or cross-thread owner references.

Owner declaration precedes borrowed-pointer RAII, original Restore, and checked
DraftUploadGuard. Reverse unwind is checked retirement -> original Restore ->
borrowed-pointer restore -> owner destruction. No callback or worker captures
the census owner. run_impl joins any issuer via IssuerJoin before return; LM
cannot start it. Per-layer validation follows successful run_impl and its
existing checked drain/cursor completion. Successful emission follows existing
final request completion and `lease.retire()`. An aborted retirement preserves
the existing abort policy; a missing final census is incomplete, never a pass.
An inference error after owner creation emits a small incomplete marker from
its destructor without changing the original error; earlier admission failures
may produce no census, which the collector must reject.

## Output and collector policy

No layer record is emitted early. After complete outer success, a dry formatting
pass calculates the entire serialized size in bounded 512-byte scratch before
writing any layer data. Hard cap: 1 MiB for one census. A dense valid histogram
can exceed that cap: it yields an explicit `output_overflow`, complete=false
record; it is not silently truncated or presented as current mix evidence.
No >1MiB report buffer is allocated. The second pass writes one JSON object
with complete=true only at its end. Writer failure stops serialization; absence
of the final complete token or a partial JSON record is incomplete. No durability
guarantee is claimed for a failed host output sink. A successfully parsed,
retained full object is the evidence consumer, never a terminal tail excerpt.

Main's collector must require a normal owned run receipt, exact input/env/source
pins, exactly one full v1 JSON report per requested outer run, complete=true,
all 48 unique layers, known histogram columns, byte cap <=1MiB, and independent
recalculation of each layer's histogram mass/rows/calls/chunks/format geometry.
Any false marker, missing report, truncation, duplicate layer, changed format
metadata, out-of-bound ne or conservation mismatch rejects qualification.
Current source hooks validate slot identity == expert and fallback ring slot
==-1, so static all-512 residency plus actual-call validation is explicit.
Do not use observer runs as clean timing samples.

## Main-owned validation, all NOT RUN

Suggested host-only commands, to execute only once Main's measurement lock
permits tests (no GPU build needed):

```
c++ -std=c++17 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined -fno-omit-frame-pointer cpu_fixture.cpp -o /tmp/strata-lm-census-host-fixture
/tmp/strata-lm-census-host-fixture
```

Fixture directly includes the proposed production header, derives independent
reference counts from synthetic token-by-token routes, and compares every exact
bin, layer format pair, GU/Down calls and rows. Includes tail chunk, full262144
geometry, overflowing position, row conservation corruption, wrong-ne call,
missing Down, uint64 overflow, native extent underflow, configured residency,
unsupported thread ownership, dense output-cap rejection, partial-request
rejection, and exception-unwind borrowed-pointer restoration. Assertions must
remain enabled. Main should add output-I/O fault injection, actual patch compile,
default-off/source-path review, and isolated current 32K/64K observer requests
before qualifying any current histogram. None of those are claimed complete.

Retention: no measurements, logs or large raw captures were produced. Keep
compact source/diff/pins/fixture/report together; no cleanup or deletion occurred.
Next review: Main source review, then host sanitizer fixture after active timing
closes; only then decide product integration and isolated current collection.
