# One-pass layerwise periodic checkpoint capture v2

**PROPOSAL ONLY. Product application, compilation, CPU fixture execution, GPU,
RESTORE, numerical, performance and physical256K qualification are all NOT_RUN.**
Only this new v2 directory was written. v1 is preserved. No Git, process/service
control, build, test, benchmark, GPU, server, profiler or cleanup action occurred.
Main's Direct0 and Direct1 same92acf71e-binary full benchmark arms must BOTH close
before Main applies, builds or tests. Main owns execution after FINAL review.
The existing project-only strata-upstream-sync skill/retention instructions
continue to apply; proposal-only scope overrides build/integration/Git steps.

## Main's blocking finding and v2 correction

Main correctly found that v1 broke the required `STRATA_CKPT_REREAD` diagnostic.
The real request path retains nonempty valid `checks`, sets `reread_to=resume`
and `read_from=0`, then replays `[0,reread_to)` without discarding those checks.
v1 entered `begin(checks,begin=0)` for that long run. Its append-only planner
correctly rejected old checkpoint lengths greater than zero, producing ERR
instead of reread. The INT64_MAX threshold prevented periodic events but did
not prevent entering that transaction. v1's Python oracle never called the
actual C++ implementation and could not expose this integration failure.

v2 adds `layerwise_capture_enabled`, called by the ACTUAL generated serve code.
It requires `reread_to<=0` in addition to the existing exact-LM/single-GPU/
positive-cache/positive-interval, batched and genuinely-long conditions. This
bypasses new capture for the ENTIRE reread request, including later outer
segments. It leaves the original outer list, whole Prefill::run, INT64_MAX
threshold/no periodic event, checkpoint_at root/turn behavior and MTP
kv_restore conditions intact. It does not weaken `begin` to accept an old
checkpoint beyond an append-only run's beginning.

## Callback lifetime correction

`ScopedLayerwiseCallback` is an actual candidate-header RAII helper used at the
real serve call site. It moves the previous callback into a guard BEFORE the
borrowed callback is assigned, and unconditionally restores that previous slot
at every scope exit: normal return, bool error, break, early return or exception.
An exception constructing/assigning the borrowed callable also unwinds the guard.
The disabled path never moves/clears/assigns the callback slot. Moves and default
construction are statically required noexcept.

The LayerwiseCheckpoints owner is declared in the enclosing serve scope, while
the guard and borrowed callable are in one outer-segment iteration. Thus the
borrowed callable is removed before its captured owner can be destroyed on
normal and exceptional unwinding. The run's inner checked queue lease retires
before wrapper return/unwind; native capture copies remain inline and blocking
through checked retirement. A failed queue retirement still aborts with owners
live. RAII does not attempt to unwind a deliberately fail-stopped GPU context.

## Actual C++ implementation fixture

`cpu_contract_fixture.cpp` includes the generated
`proposed/include/strata/program/layerwise_checkpoint.hpp` unchanged. It actually
instantiates and calls `LayerwiseCheckpoints::begin/layer/prompt/commit/discard`,
`layerwise_capture_enabled`, `layerwise_next`, and `ScopedLayerwiseCallback`.
It also uses the real product `conv_cache.hpp` and `conversation_memory.hpp`.
No copied planner or replacement implementation is under test.

Only GPU-dependent core types, checkpoint sizing and checkpoint validation are
replaced through two clearly scoped headers under `fixture_stubs/strata/core`.
They supply small host tag buffers with the same fields consumed by the header.
The fake copy recorder synchronously copies real bytes and verifies layer,
absolute prefix, field kind and PLE owner against independent mathematical tags.
The fixture's retention oracle interprets token time/chunk ends and the written
root/LRU policy without calling the candidate planner or production eviction
helper. It compares the actual published token/image prefixes, state bytes and
used stamps, and asserts zero externally visible state/stamp changes before
commit. A manual 256K/cache6 expected-position case checks the oracle as well.

The fixture covers:

- Actual 32K/64K/256K, chunks4096/8192, interval1/16384/16385, cap1/6,
  nonmultiple tails, shifted begins, overdue window threshold, INT64_MAX sentinel,
  old-state retention, pinned root, periodic/root duplicate position and stamps,
  exact image-prefix inclusion and capture count bounded by final NEW survivors.
- Fake GDN rows, QSA tail/dead/blockpos and PLE bytes at layer1. Every new
  surviving checkpoint requires 36+3*12+1 fake copies; evicted positions have no
  copies. These are position/ownership tags, not real tensor arithmetic.
- Stale run, missing/out-of-order/duplicate layer callbacks, missing/duplicate
  MTP completion, early cancellation and every fake copy submission failure
  point; no partial commit. Full-feed cancel/run/expert-restore/refill failures
  use the caller discard path with REAL nonempty old checkpoint payloads.
- INT64_MAX saturation, invalid/overflowing schedule inputs, exhausted stamps,
  invalid geometry/token IDs, unknown/low RAM, and checked byte addition/product
  overflow injected through stub sizes BEFORE allocation.
- Callback restoration after success, early return, break, run exception and an
  exception before callback installation; disabled-slot preservation.
- The concrete reread regression: old checkpoints at2048 and32768,
  reread_to32768>chunk8192, read_from0. Negative control calls the ACTUAL v1-style
  begin and requires its old-chain error; v2's ACTUAL gate bypasses that begin
  for the reread and all later segments, preserving old payloads/stamps and the
  original sentinel/no-periodic/root-turn event schedule.

The retained `state_position_fixture.py` is the v1 independent specification
oracle, byte-identical to v1. It remains supplementary; it is not execution of
the C++ candidate. No fixture has been run by the author. Assertions are required;
the C++ fixture rejects NDEBUG builds.

The stubs do NOT qualify real core checkpoint validation, SessionState geometry,
SYCL queue ordering/retirement, numerical tensors, QSA/MTP KV reconstruction,
actual cancellation, RAM telemetry/allocator behavior, full physical256K or
performance. The caller lifecycle cases test actual discard/no-publication with
fake outcomes; they cannot prove native MTP reattachment or loan refill. Main
must still review and execute the actual integration paths.

## Preserved architecture and memory contract

The rest of the v1 design remains: a distinct internal native per-layer callback
at run_impl's checked m.cs chunk boundary, one GDN slice or one QSA tail/dead/
blockpos per owning layer, PLE history only at layer1, matching token/image
prefixes and run identity, fixed48 completion mask, and final target/MTP callback
completion. Checkpoint restore still derives PLE prev from saved IDs and retains
existing QSA pooled dead-row reconstruction and MTP kv_restore. Positional target
and draft KV arrays and checkpoint serialization format are unchanged.

Preview the exact existing root-pinned/LRU event/stamp/de-dup policy using metadata
before allocating state. Allocate ONLY final new surviving positions, not all16
periodic snapshots at256K/cache6. Keep old checks alive until preallocated,
noexcept-move publication after full run + MTP expert restoration + slot refill.
Cancel/error discards pending states and preserves previous published checks.
Snapshot byte-size success never establishes absolute state position.

One uninterrupted existing outer segment still makes one full expert sweep,
**50,292,326,400 B logical packed expert payload**. No new16K caller segmentation
is introduced. Existing root/turn/reread boundaries remain. Non-LM, multi-GPU,
cache0, interval0 and short calls retain their old running path. No new toggle
and no setenv/unsetenv occur. Environment is fixed in native child before exec.

Memory formula remains: S(P)=GDN+optionalPLE+12*(tail+dead+blockpos)+4P+image keys.
O=resident old payload capacities, N=sum S(P) for final NEW survivors, M=bounded
metadata/owners/publication, B=2*(N+M) conservative new capacity budget. Admit
**O+B+8GiB <= O+available_after_old**; old states are already charged in free RAM
and cannot be reclaimed early. Preview gets its own preflight; actual capacities
are checked; LM later admits residual/upload owners cumulatively against remaining
RAM with8GiB reserve. Existing six and new up to six can coexist until commit;
there is no third state copy. One GDN checkpoint is117669888B=112.21875MiB,
six are706019328B=673.3125MiB before metadata/IDs/PLE/QSA/capacity.

R609 payload accounting is useful. Dividing bytes by observed2.25GB/s bulk or
5.79GB/s expert-copy rates gives scenarios, not hardware lower bounds and not
proof that a throughput target is impossible.

## Main's serial validation plan — all NOT_RUN

Only after BOTH Direct arms close:

1. Compare current input source hashes against sha256-ledger.json; review source
   resolver, source.diff, header and stub boundary. Generator writes this directory
   only and rejects unexpected source input hashes. source.diff is a full candidate
   against the same pinned product source, not a patch to apply after v1.
2. Build/run actual C++ fixture with assertions using an owned output path, then
   build/run with ASan/UBSan. Example from repo root (commands NOT EXECUTED):

   ```sh
   proposal=bench/results/2026-10-15-pr6-recovery-audit/layerwise-checkpoint-capture-v2
   c++ -std=c++20 -UNDEBUG -O1 -g -Wall -Wextra \
     -I "$proposal/fixture_stubs" -I "$proposal/proposed/include" -I include \
     "$proposal/cpu_contract_fixture.cpp" -o "$main_owned_output/cpu-contract"
   "$main_owned_output/cpu-contract"
   c++ -std=c++20 -UNDEBUG -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
     -I "$proposal/fixture_stubs" -I "$proposal/proposed/include" -I include \
     "$proposal/cpu_contract_fixture.cpp" -o "$main_owned_output/cpu-contract-sanitized"
   "$main_owned_output/cpu-contract-sanitized"
   python3 "$proposal/state_position_fixture.py"
   ```

   Main must set main_owned_output to its private existing output directory.
   Stub include order is mandatory and restricted to this fixture executable.
   After product integration, use the integrated candidate header for the fixture;
   never place fixture_stubs on a native product target's include path.
3. Run actual configured SYCL build and relevant existing checkpoint/cache tests.
   Verify RAII on real exception paths, callback owner lifetime, queue retirement,
   copied source offsets and old/new RAM overlap. CPU stubs cannot prove these.
4. First real regression: cache6, actual nonempty retained checkpoint at>chunk,
   fresh separate immutable-process reread diagnostic. Confirm STRATA_CKPT_REREAD
   reads `[0,resume)` in its original whole segment without ERR, no new layerwise
   transaction or periodic publication occurs anywhere in that request, and
   root/turn, MTP skip/restore, RESUME/REUSED and subsequent decode remain correct.
   Compare controlled reread and RESTORE on the same actual IDs/schedule.
5. Small cache0/interval0/short/nonLM controls and long cache6/interval16384 first.
   Then actual matched32K/64K restored-v-reread comparison across every retained
   point: all36 GDN, PLE history/prev, all12 QSA tail/dead/blockpos and pooled dead
   row, target/draft KV, first logits/residuals, fixed output IDs, MTP offered/
   accepted. Include root/turn de-dup, shifted resume, interval16385, windows63/64/65,
   shared-root/branch/miss/LIVE, images/cvec and tokenizer-confirmed thinking turns.
6. Real cancel/error lifecycle around early/PLE/middle/final layers, after MTP,
   wrapper restore/refill/commit, then next request. Preserve published old checks,
   drop incomplete pending/LIVE, and never swallow errors with a simultaneous stop.
   Native queue-retirement failure remains fail-stop, not a graceful test arm.
7. Full PHYSICAL max-context262144 allocation with real host-backed/streamed KV,
   near-end prompt and legal later decode; prove actual n-1 batched boundary,
   page/slot/fallback paths and retained-prefix restore. Logical fixture indices
   and a CLI flag are not physical256K qualification.
8. Only after correctness/fault/cancel gates, clean repeated32K/64K/256K and real
   agent turns, identical settings/IDs/binary within each pair. Record individual
   PP/TTFT, copy bytes/time, MTP restore, refill, decode/restored samples and expert
   sweep count. No speed/numerical adoption follows from this proposal.

No native arm changes environment after runtime/provider initialization. Keep
compact receipts, commands, environment, individual outcomes and decisions per
ARTIFACT_RETENTION.md; keep only named necessary large captures outside Git.

## Pins and retention

sha256-ledger.json records old/new product hashes, generator, actual header,
fixture/stubs, diff, v1 originals and supporting reports. The generator performs
only artifact generation/hashing; it does not compile or execute fixtures.
Consumer/owner: Main review, integration and later serial qualification. Budget
2MiB for this directory; next review at Main source review/integration/rejection.
No raw captures, deletion, cleanup or external backup was created. Deletions=[].
