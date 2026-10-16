# PR #6 engine restoration for isolated review

This directory restores the engine source closure selected by
`d0c5dda0dc2e8d82458c4b0ff31c4cef44c40b32:sycl/CMakeLists.txt`, including its
historical root fallback files and headers. The implementation is
`f263d87d44aa671d056c33e6c6b6b0635fc25b8c`; engine files are identical at those
revisions. This is an independent validation route. The normal root build and
`feature/sycl` are unchanged.

The first restoration commit preserves every historical project file exactly.
It is an unsafe source reproducibility anchor, not a validated executable.
Later commits hold separately attributable safety fixes. Main must review the
whole patch before building or executing either version. No build or test has
been run by the worker; >600 tok/s has not been remeasured.

The new outer CMake wrapper changes only isolation/dependency provisioning:

- It requires Main's explicit reviewed-build flag.
- It delegates to the unchanged historical `sycl/CMakeLists.txt` and resolver.
- It requires external historical DPCT/ggml-common headers, checking all 49 file
  SHA256s from `historical-closure.json`. No dependency tree is copied here.
- It requires an existing offline llama.cpp checkout at the historical pin
  `3cf03257f219afbe7334045ff7c6a06ac68c627d`. Main must verify its HEAD and diff.
- It disables the historical GPU parity target group because this restores the
  engine target `strata`, not the unrelated legacy fixture suite.

`sycl/tools/migrate.sh` and `fixups.py` document the original generation route;
regeneration is neither required nor appropriate for reproducing these blobs.
`sycl/tools/prefill_profile.py` is the final PR helper. It starts native programs;
do not run it until Main schedules execution. Its default cwd is historical to
this nested snapshot; pass an explicit cwd for a scheduled reproduction.

See `../../bench/results/2026-10-16-pr6-faithful-restore/report.ja.md` and
`manifest.json` for provenance, differences, bugs and Main-owned execution gates.
`historical-closure.json` inventories all copied files, their original Git blob
IDs and SHA256s, resolved translation units and external headers. Runtime sources
under `sycl/src/kernels/cuda/*.dp.cpp` are the historical SYCL migration output;
no CUDA/HIP implementation is built or newly imported.

Safety corrections v2 supersede the earlier incomplete closure count: the full
historical resolver requires 84 TUs and 198 project files (4,007,615 original
bytes). The faithful anchor still contains its original 197 files; R1 restores
the missing original peer stub in a later commit. Original reports and review
statuses remain historical records. See `safety-corrections-v2.md` and its JSON
in that same results directory for exact corrected-source pins and Main's gates.

V2 explicitly uses one selected in-order queue for prefill copies and compute,
checked event retirement and finite fail-stop deadlines. It requires the outer
wrapper: the wrapper generates a private checked DPCT header from the hash-pinned
external dependency and force-includes it for the engine targets, preserving
the vendor notice and leaving the external installation untouched. This changes
synchronization/error handling and uniform queue profiling, so historical
performance claims do not qualify the corrected route. Its status remains
`SOURCE_ONLY_UNSAFE_TO_RUN` until Main reviews and executes the separate gates.

V3 repairs the D1 queue-query boolean, D2 position/pointer admission, D4 embedding
one-attempt load, and D5 normal issuer join deadline at source level. The new
`pr6_validation_control` and `pr6_bounded_join` ordinary C++ fixtures do not load
SYCL and have not been executed by the source worker. Empty MTP/draft-KV prefill
is a no-op only at an admitted nonnegative endpoint; nonempty calls require both
pointers. Span admission uses physical 262144 and actual session/QSA/draft-state
capacities, with draft lookahead checked separately. Full physical256K execution
is still unqualified.

D3 is a hard exclusion, with no environment override. `doorbell_init` refuses
before allocating; decode `session_loop`, token graph capture/replay and every
verifier mode (including eager, host-boundary, no-host/all-resident, split and
commit) refuse before recording or launch. The historical device handshake source
is preserved as held comparison source; host fences are not an atomic-host proof.
Verifier construction keeps null queue pointers, so refused init cannot initialize
or later destroy borrowed DPCT defaults.

The CLI refuses these modes after argument parsing, before device queries or
allocation. Its retained original `--no-capture --no-pool` route is explicitly a
GPU-floor decode diagnostic: it has no CPU expert hook and is not full-model
decode. It uses no doorbell allocation. Ordinary `Prefill::run` and draft-KV math
remain available to Main's separate prefill harness; the engine CLI is not a
full native-pack or historical600 reproduction route. No new scheduler or
handshake is substituted. All native routes, including this fallback, retain
`SOURCE_ONLY_UNSAFE_TO_RUN` pending Main's source/build/provider/numerical gates.
See `safety-corrections-v3.md/json/patch` in the results directory for the exact
base, complete source patch, preservation checks, remaining gates and timing
changes. No prior report or fixed pin is superseded in place.
