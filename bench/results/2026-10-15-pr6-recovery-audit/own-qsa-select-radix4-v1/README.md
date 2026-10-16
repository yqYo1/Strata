# Own exact QSA radix4 selector proposal v1

Qualification: **UNRUN** build, GPU correctness, timing, driver/fault, model, RESTORE,
and physical-context execution. This package only prepares source. Main owns all
serialized builds, tests, GPU work, environment setup and integration. No active
source file was edited, no Git mutation was performed, and no runtime was created.

Origin: independent implementation against this fork's actual current selector
bytes and weighted MSD reasoning. No XeStrata code was inspected, copied, adapted
or used as an implementation reference. R738 is algorithm context only; its
scalar-oracle order and distribution-dependent load prediction are corrected
here. The CPU oracle sorts expanded cells by key descending then ID ascending,
takes the requested width, and finally sorts the selected IDs ascending.

`base/` contains the actual byte snapshot used to make `proposal.patch`.
`files/` contains the complete proposed files. `manifest.json` and `SHA256SUMS`
pin those bytes. `prepare.py` creates mirrors/patch/manifest from that snapshot;
once captured, it never refreshes the base from the active tree. The selector's
base SHA256 must be
`45084756ca5e8e36085dae46581049c81366a45cdfde23e8c6f38defbba0c809`.
Main should review the full patch and compare the current touched-file hashes
with `base/` before applying it. A changed active base requires a fresh explicit
review, not replacement from these mirrors. The fixture is a new file. Only
`src/kernels/sycl/qsa_select.cpp`, `cmake/sycl_backend.cmake`, and
`tests/sycl/qsa_select_radix.cpp` are in the proposed patch.

## Behavior and arithmetic

Only exact `STRATA_SYCL_SELECT_RADIX4=1` enables the experimental specialization;
unset, `0`, empty, and every other value retain the original 32-bit threshold
specialization. The value is read once on the first selector call. Set it and
all SYCL/Level Zero/driver/experiment variables in the native child's environment
before exec and keep that environment immutable. The source and fixture contain
no environment writes.

Both specializations retain one 256-work-item work-group per query, the existing
score kernel, geometry/span validation, active bounds, device step guards,
identity/width-zero paths, key mapping, weighted greater/equal budget, and
ascending-ID emission. The existing `qsa_block_topk_ref` forwarding behavior also
remains; use separate processes for a reference arm. There is no new physical
context validation. Since nonnegative `n` is loaded from int32 steps, every
histogram/subset/reduction count is at most `n <= INT_MAX`. The code does not
assume `n <= 262144`. The spare has zero weight at tail zero; tails 1/2/3 have
that exact weight. NaNs map below negative infinity, signed zeros tie, and all
threshold ties choose low IDs.

The candidate scans the same fixed partition eight times with private 16-bin
weighted histograms and 16 group reductions per round. Each scan must read each
key before applying the prefix check: **8*B histogram key visits + 2*B final
visits**, independent of prefix distribution. There is no compaction. This is a
source work count, not measured memory traffic. Register, spill, reduction,
local-memory and performance effects are unknown; no speedup or resource claim
is made. The source does not allocate a new global scratch buffer or change the
score calculation.

## Manual fixture after Main review and integration

Use the existing configured SYCL build directory (`build-sycl` below is an
example, not an assumption about Main's configured path). The target links the
existing `strata_kernels`; it does not compile a second selector copy. It is
`EXCLUDE_FROM_ALL` and has no automatic CTest GPU registration.

```sh
cmake --build build-sycl --target sycl_qsa_select_radix_test -j2
STRATA_SYCL_SELECT_RADIX4=0 build-sycl/sycl_qsa_select_radix_test --dump /PRIVATE/baseline-normal.ids
STRATA_SYCL_SELECT_RADIX4=1 build-sycl/sycl_qsa_select_radix_test --dump /PRIVATE/radix4-normal.ids
cmp /PRIVATE/baseline-normal.ids /PRIVATE/radix4-normal.ids
```

The normal fixture runs 17 selector batches plus host validation, all untimed:
eight score patterns each run a 75-query small batch and a seven-query long
mixed batch, followed by a ten-query mixed-valid/invalid-step batch. Small cases
include `n=0..12` with widths `0,n/2,n`; `n=1016..1027`, corresponding to
255/256/257 score slots including every tail, width 127; and `n=2050..2053`
with widths `min(n,2051),1,3,4,5,2049`. Long mixed batches contain identity
`n=0,5`, actual `n=78453`, and physical `n=262141..262144`, long width 2051.
Each score row has two unused stride slots filled with positive infinity.
The zero-weight spare is also positive infinity. Physical-context rows run no
timing loop.

Patterns cover varied finite sign/exponent/mantissa keys; all equal; mixed
NaNs/infinities/zeros/finite extremes/subnormals; all NaNs of both signs and
varying payloads; signed zeros only; adjacent representable floats; large tie
groups crossing partitions and partial rank budgets; and a deterministic
permutation of distinct/repeated finite values. Every returned ID is compared
to the independent expanded-cell oracle and checked for range, unique ascending
order, row-tail canaries and outer canaries. Score and step storage is checked
unchanged byte-for-byte (scores may contain NaNs). Invalid steps cover negative
n, mismatched nb, nb at active bound, negative width, width over cap and width
over n; their entire output rows must remain sentinel. Host validation includes
nq/blocks/cap/active/shape bounds, null/misaligned pointers, and writable aliases.
Runtime and allocation ownership use checked drains before owners are destroyed,
with fail-stop on a failed drain, following the existing prompt fixture.

## Optional clean component comparison

Run each line in a fresh owned native process, with the same immutable provider
and device environment set by Main. Run multiple independent processes per arm
and alternate arm order; do not change variables from inside a process. A clean
measurement needs its own deadline, fault/teardown receipt and compact sample
record under `docs/ARTIFACT_RETENTION.md`.

```sh
STRATA_SYCL_SELECT_RADIX4=0 build-sycl/sycl_qsa_select_radix_test --bench varied 5 --dump /PRIVATE/varied-arm0-r0.ids
STRATA_SYCL_SELECT_RADIX4=1 build-sycl/sycl_qsa_select_radix_test --bench varied 5 --dump /PRIVATE/varied-arm1-r0.ids
cmp /PRIVATE/varied-arm0-r0.ids /PRIVATE/varied-arm1-r0.ids
STRATA_SYCL_SELECT_RADIX4=1 build-sycl/sycl_qsa_select_radix_test --bench equal 5 --dump /PRIVATE/equal-arm1-r0.ids
STRATA_SYCL_SELECT_RADIX4=0 build-sycl/sycl_qsa_select_radix_test --bench equal 5 --dump /PRIVATE/equal-arm0-r0.ids
cmp /PRIVATE/equal-arm0-r0.ids /PRIVATE/equal-arm1-r0.ids
```

Benchmark CLI accepts only `varied|equal`, 1..20 repeats, and a required full
dump path. It uses exactly nq=256, n=78453, nb=19613, width/cap=2051,
score stride=19616, active=19614, tail weight=1. One invocation completes before
the timed submission loop. `total_submit_drain_ms` covers all repeat submissions
and the final checked runtime drain. Host oracle, transfers and output checks
are outside that interval. Each query receives the full expanded-cell oracle,
including benchmark queries; the complete first output plus all guards is dumped,
and the final repeated output is checked against it. The dump is a native-endian
uint64 element count followed by all int32 output elements, repeated per case.
No event-service, end-to-end prefill or actual-model inference claim follows
from this component interval.

Main must still assess the real 78453-token request, exact selected-row parity,
whole-dependency-path prefill, repeated clean samples, full physical context,
fault/teardown and RESTORE before any adoption. The candidate stays default off.
The source proposal is compact evidence; generated dumps are temporary comparison
fixtures consumed by Main's arm comparator and should be retired after recorded
hashes/comparison results, under the project retention rules.
