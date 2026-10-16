# Main's serial validation plan (all NOT_RUN)

## Freeze release and identity

Before even CPU compilation, Main closes and records both existing Direct0/1
full benchmarks of the same 92acf71e binary: original terminal results,
thinking-on three runs, agent turns, long-generation results, faults, and all
server/client/controller owner retirement. Do not alter source/build/binary
while either formal benchmark has a live owner. Do not infer release from a
timeout, missing poll, one finished arm, or this proposal.

Main then holds the existing exclusive measurement lock for every build/test,
including CPU and offline comparator work, and uses its owned-process runner.
Record boot ID, PID/start ticks, argv, exact exec environment, timestamps,
deadline, exit/owner retirement and fresh kernel/driver fault-check boundaries.
Stop advancement after any failed, incomplete or unknown-completion case.
No recovery/reset/service action is authorized by this proposal.

Verify source-pins and artifact-sha256 against bytes. Pin current product
revision/diff, all linked archive/binary hashes, compiler/runtime/driver versions
and hardware. Main may run `git apply --check` then apply source.diff only after
the release and review. Recheck exact new product hashes. No candidate default
change is proposed. Subsequent fixes require a new recorded revision.

## Host and build gates

1. Build the host fixture in a new Git-external directory with `/usr/bin/c++`,
   `-std=c++20 -O2 -fno-fast-math -ffp-contract=off -Wall -Wextra -Werror`,
   `-I<proposal>/proposed/include -I<worktree>/include` and
   `<proposal>/cpu_fixture.cpp`. Run once under a 600 s deadline, 4 MiB stdout+
   stderr budget. Record every format completion, totals, exit and source hash.
2. Separately build/run `-O1 -g -fsanitize=address,undefined
   -fno-omit-frame-pointer` with the same arithmetic flags, all corpus cases,
   immutable `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and
   `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1` passed before exec.
   Deadline 1200 s, logs 8 MiB. Neither run is a CPU performance benchmark.
3. Rebuild the existing native product with its pinned compiler/configuration.
   Confirm actual iq.cpp command still disables contraction and fast math, uses
   size_t SYCL ranges, and source bytes match this proposal. Build the standalone
   gpu_fixture.cpp using the same native SYCL compiler, `-std=c++20 -O3
   -fno-fast-math -ffp-contract=off -fsycl -fsycl-id-queries-range=size_t
   -Wall -Wextra -Werror`, `-I<worktree>/include
   -I<worktree>/include/strata/sycl/compat`. Link against that build's
   `libstrata_kernels.a libstrata_core.a libstrata_sycl_runtime.a`; the current
   sycl_kernels_test entry in build.ninja is the matching link recipe. Preserve
   platform libraries/rpaths/device-link flags from that actual configuration.
   Do not invent an icpx2026.1 build or import another engine. Record the full
   final argv and link/binary hashes, including any necessary link adjustment.
4. Inspect generated kernel symbols/resource reports without device execution
   if available. Kernel private-array elimination is an expectation, not an
   established spill/occupancy result. Product OFF must submit the same old
   kernels/count/order without candidate allocations or diagnostics. Do not add
   tracing to product code just to prove this; use one separate bounded device
   diagnostic if needed.

No commands above have been run. Compiler warning/error or fixture issue is a
new failed result to preserve and correct, not permission to weaken the gates.

## Single-arm GPU captures

Use Main's existing runner to construct a **child env dictionary** from its
reviewed baseline before exec; pass it to the child, never mutate parent or
native process environment. Pin selector, SYCL/UR/ZE controls, threads and
LD_LIBRARY_PATH. The only candidate difference is exact flag value. A native
process handles one arm for its whole lifetime.

The native argv arrays are:

```json
[
  ["<gpu-fixture-binary>", "off", "quick", "<new-private-root>/off-quick"],
  ["<gpu-fixture-binary>", "on",  "quick", "<new-private-root>/on-quick"],
  ["<gpu-fixture-binary>", "off", "full",  "<new-private-root>/off-full"],
  ["<gpu-fixture-binary>", "on",  "full",  "<new-private-root>/on-full"]
]
```

OFF child's STRATA_SYCL_PARALLEL_IQ_DEQUANT is `0` (also screen unset separately);
ON child's value is `1`. Both process environments remain immutable. Every arm
requires normal exit0, complete capture, owner retirement and fresh fault0
before launching the next. Quick arms precede full arms. Initial deadlines:
quick 180 s, full 600 s; each native log 4 MiB, raw capture 128 MiB hard budget.
Whole initial two-quick/two-full job raw budget 512 MiB, text budget 16 MiB.
These finite enlarged capture budgets are justified by complete FP16 matrices
and CPU references; they are not an exception for unbounded tracing. Budget or
deadline exceedance is incomplete/failure, never silently truncated PASS.

After both matching native arms close and Main checks binary/source/input/env
identity, run the offline comparator:

```text
python3 <proposal>/compare_captures.py <off-quick> <on-quick> <NEW-quick-report.json>
python3 <proposal>/compare_captures.py <off-full> <on-full> <NEW-full-report.json>
```

Compare all half words, guard words, full input files, oracle buffers and F32
arrays. Device OFF/ON NaN bits must be identical. Canonical-NaN CPU/device sign
differences may be reported separately by the native fixture and comparator;
the sign exception never applies between device arms. Any finite/Inf/signed
zero/subnormal difference fails. Preserve the first useful divergent full
buffer with its exact consumer and question; do not mask or threshold it.

The quick corpus covers 1,024 packed pattern IDs at every actual output index,
18 cycling scale edges, first/last block, offset2 generic and offset0 GU,
interleaved rows, and all other supported types' two half entry paths. CPU's
scale sweep is broader. 20 negative entry calls must reject before submission.
F32 checks use finite scales, six formats and guard words. Add out-of-order
queue rejection and null-stream completion checks using Main's existing owned
runtime/context fixture if needed; do not switch process environments inside
legacy fixtures. The candidate retains both contracts by source inspection,
but this delivery contains no new runtime qualification of them.

For exact flag parsing, additional **separate** OFF quick processes may screen
unset, empty, `01`, `true`, `1 `, `2` and `-1`. Native fixture accepts any OFF
value except exact `1`; Main captures the exact value externally. Compare their
complete buffers and independently verify old route/count. No new product flag
diagnostic is added. Correct bytes alone do not prove candidate dispatch.

The native fixture emits host-submit time, host wall through queue drain, and
the interval between two fixture marker events. Record these independently.
The marker interval can include host submission gaps; it is **not** exclusive
dequant device duration. Markers are fixture-only, two extra launches/case.
The complete receipt's expected_product_launches_per_valid_case is a source
expectation; it is not observed route evidence. These one-shot times include a
diagnostic environment, are not clean speed samples, and must not decide adoption.

## Performance and actual model gates

After exact buffers pass, take clean independent-process component timing
samples, alternating OFF/ON with at least three samples/arm and more if the
effect overlaps variance. Record all samples by format, role and shape;
host submit, completed batch wall and actual device kernel durations separately.
Use a bounded official device-timing collection separately from clean timing;
do not infer exclusive wall from the profile's sum. Compare repeated-load costs,
launch count/order, dequant+existing library GEMM together, and any spill/cache
evidence. Weight coverage by actual call and routed-row histograms, not just
synthetic bytes. If a format loses, retain the old implementation for it rather
than claiming a blanket win. This v1 is deliberately all six formats opt-in
until measured per-format decisions exist.

Run paired actual fixed-history 34,603-input / 128-output jobs in independent
OFF/ON processes, same built binary/model, source/backend, Direct setting,
MTP4, chunk4096, int8 KV, KV residency, cache/agent settings and request body.
Pin input and request hashes and all 128 IDs/work. Preserve first and selected
boundary logits, routed expert work, staging buffers and GU/Down output
fingerprints. Compare full half staging tensors on all seven observed format
pairs using bounded Main captures if an existing hook is available. Missing
capture support is a named uncompleted gate, not proof from final text alone.
The synthetic fixture covers all six formats but not actual model tensor
identity, task quality or downstream state.

Keep prefill, cold decode and restored decode separate. A prefill gain does not
justify a decode regression. Check numerical distributions at the same fixed
history/positions and exact FP16 operands; do not compare later logits after
divergent histories or treat synthetic random weights as model quality.

Then repeat the pinned official bench/client with thinking enabled, at least
three valid samples per arm, long generation and full agent session. Include
actual >=32K and ~64K fresh-input cases (record actual token counts), all agent
turn read/reuse/TTFT/decode/draft counts and request/output hashes. Start from
Main's completed Direct benchmark configuration; setting screens (including
agent cache) are separate experiments, not attributed to the dequant flag.
Use normal bench.sh --long --think --long-generation --runs 3 as appropriate to
the already-pinned client. Record original individual results and failures,
not just medians. This proposal has no speed/adoption evidence.

Complete physical 256K gates with engine evidence of actual last-cell/chunk
consumption, full input262144 boundary behavior, nearby output-space case and
out-of-range rejection. Capacity configuration or a small array with a large
logical index is not full context validation. Run actual RESTORE from the
current session fixture, compare next-token/distribution/state behavior and
restored decode timing. Exercise streaming cancel, drain, owner lifetime,
subsequent request and detach/reattach/restore recovery under both arms. These
paths are untouched by source but remain regression/qualification obligations.

## Retention and decision

Main owns this proposal and its first serial validation. Keep proposal source,
all exact commands/env/pins, compact individual results, correctness/fault and
owner receipts in Git. Necessary raw arrays live outside Git; the current
consumer is complete OFF/ON equality and the first numerical divergence review.
Review at the first successful comparison, rejected candidate, or next source
revision. Preserve one required unresolved mechanism fixture; retire redundant
successful captures only after verified compact evidence, closed owners and a
path/size/hash/reason deletion manifest under the measurement lock. No deletion
or cleanup is performed by these fixtures or this proposal.

All gates are **NOT_RUN**: source application, CPU normal/sanitized, SYCL build,
quick/full GPU arrays, actual route and counts, faults, component performance,
34K128 work/numerics, clean >=32K/64K bench, task quality, physical256K,
RESTORE/cancel and performance adoption. Default remains OFF.
