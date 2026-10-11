# Preload-disabled native-dequant model diagnostics

2026-10-11, Intel Arc B570 10 GiB, Ryzen 5600X (AVX2), kernel 7.0.0-38-generic.
This boundary preserves finite correctness diagnostics and their rejected attempts.
Native batching remains opt-in, disabled by default and not adopted for speed.

The five-line `STRATA_PRELOAD_KERNELS=0` guard is published in the user's fork at
`0740b43488d812117f021ccbe4bd0e15d223f955`. The guard skips eager kernel preload;
other values preserve the existing path. It changes first-use compilation timing,
not arithmetic. The actual builds and model runs used parent `fa3e20e842fbed6986a69bade8afe3edabedee07`
plus that exact uncommitted guard. Publication checked all 38 pinned source-file
contents against the built version. Historical build provenance is preserved as
historical provenance; the build receipts are not relabeled with the later commit.

Release builds for `free` and `contrib-llvm` each passed 12 CPU tests and skipped
one unsupported AVX512 test. Both host-only ASAN/UBSAN builds passed the same CPU
tests. Device code was not sanitizer-instrumented. The actual lint runner found
the same two existing diagnostics in each mode and no new ones. CUDA, HIP and
contrib-icpx were not qualified here. GPU parity executables were built; those
executables were not run as part of these build checks.

The source-built intel/llvm 7.1.1 toolchain has a 7.1.0pre/Clang 22.1 banner.
The actual model process mapped IGC 2.41.5, FCL 2.41.5 and OpenCLClang 22.1 from
`/usr/local/lib`; hashes, ELF build IDs and selected stable mapping rows are in
each receipt. Installed distro package names alone do not identify the provider
used by these runs. No driver, runtime or service was changed at this boundary.

## Finite model checks

Each arm used an independent owned process, physical context 262144, int8 KV,
32768 resident KV positions, requested prefill 6144, requested cache 128
(effective 144 slots, three per layer), five CPU workers, pcie-frac 0, MTP4 and
spec-min-p 0. The full-stream/routed ring capacities were 16/8. Preload was
disabled, fused dequant and asynchronous issuer were disabled. Model diagnostics
used warning/error logging, validation, Strata progress and census output.
They are explicitly ineligible as performance samples.

| Prompt tokens | Actual batched positions | OFF prefill ms | ON prefill ms | OFF cold decode ms | ON cold decode ms | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 64 | 63 | 4939.9 | 4744.5 | 3259.6 | 3358.7 | Both PASS |
| 2049 | 2048 | 20817.0 | 21065.9 | 2577.7 | 2523.7 | Both PASS |
| 32768 | 32767 | 127433.6 | 130715.0 | 2599.6 | 2258.7 | Both PASS |

All pairs produced exactly the same 16 greedy token IDs and all emitted top-five
logprob entries. The main comparison independently joins every route and H2D
row; it excludes only the intentional legacy dequant-wrapper call counters
from route equality. Each run reached DONE, accepted normal QUIT, exited zero,
left no owned session, and passed independent health probes and fresh kernel
fault checks on the same boot. This is narrow request equivalence, not broad
model-quality, RESTORE or full-256K qualification. Small requests are correctness
checks; they are not the speed comparison requested by the user.

At 32K, OFF and ON each submitted 144639 H2D requests totaling 295956300800
logical bytes. ON admitted 35831 native group calls per axis covering all
137407 logical experts, with zero declines or fallback. This reduces host API
call count by about 73.9%, without reducing those H2D bytes or demonstrating a
speed win. OFF was 257.14 diagnostic prefill tok/s and ON 250.68. Neither a speed
regression nor a decode improvement is established from these single diagnostic
samples. Prefill and decode decisions remain separate.

The last chunk has 2047 positions; it uses routed staging while the first five
6144-position chunks use full nonresident streaming. Its mean group width is
1.044, versus 7.54–9.24 in the streaming chunks. Zero counted held-reuse flushes
in streaming is structural: that counter increments in the routed branch only.
It does not prove absence of stream-credit pressure. Individual window-size
histograms and exclusive device phase times were not collected.

## Rejected diagnostics and controller corrections

The v4 N64 OFF attempt hit its aggregate log budget after READY/GEN, before DONE.
Its model stderr was 64623852 bytes / 369878 lines. The original receipt stays
FAIL; owner cleanup reaped its SIGTERM exit and emptied the session. The final
journal observation also exceeded the shared budget, so the separate successful
fresh health check does not rewrite that failed gate. A bounded excerpt preserves
progress, the final API window and the recorded budget stop. No complete-history
consumer remains for this trace.

The v5 N64 OFF attempt completed 16 tokens and exited normally, but Main's
controller passed additional IGC/compiler mappings into the unchanged strict
core-runtime validator. The provenance gate rejected that mismatch; the original
receipt stays FAIL. Main corrected the caller to filter the original runtime
tuple while retaining the extra mappings separately. CPU fixtures cover this
filter, pending-manifest denials, mapping identity/budget failures and normal
QUIT after READY or DONE without overwriting a primary failure. The corrected
v6 diagnostics are new results, not rewritten failed samples.

The first archive-controller attempt also failed while planning deletion:
synthetic CPU child logs live in two owned child directories outside `commands/`.
The second attempt verifies the existing partial copies and accepts only those
two explicit directories with embedded terminal ownership proof. No deletion or
GPU action occurred in either archive attempt.

The earlier all-ID preload PackedA build error is still unresolved. Successful
selected kernels with preload disabled do not qualify every template or prove
the provider fault fixed. The representative earlier error remains in the
previous committed diagnostic report.

## Next work and retained evidence

The pending quiet32K controller is source-only and untested. It must receive
Main source review and CPU qualification, exact published-source pins and strict
diagnostic-reference checks before launch. Planned timing requires at least
three independent samples, individual prefill and decode times, and a reviewed
execution order. No benchmark is launched by this archive work. Existing event
timers can answer exposed wait/compute questions in separate diagnostic runs;
their non-exclusive intervals are not complete DMA service times.

An unmeasured fixed chunk 6140 is a separate candidate for the exact 32767-position
fixture: five chunks plus 2067 retain six passes and avoid the 2047 routed tail.
Actual admission and correctness must be checked. It is not a general default
or a full-context result. The 4096 proposal would add two passes and is not
accepted without evidence that compensates for their transfer cost.

The user's [official parameter-audit TODO](../2026-10-11-upstream-parameter-audit/TODO.md)
remains open for independent tuning and validation. Historical official runs
used explicit nonstandard settings; that is not evidence of optimal settings.
No current result is described as a win over an independently optimized official
baseline. Official v0.1.42 pristine measurement and safe update evaluation remain
pending after XeStrata qualification.

[Archive manifest](archive-manifest.json) identifies byte-exact originals and
lossless historical source/patch encodings. [Paired comparison](main-paired-comparison.json)
contains all individual results and joined evidence. The incremental
[research registry](report-registry-v131.json) preserves 18 newly reviewed reports,
545 total entries, and the assignment snapshot; inspect live agent state on
resume. [Provenance corrections](main-provenance-corrections.md) preserve original
reports while correcting links and superseded evidence scope.

[Retention manifest](retention-v1.json) plans removal of 402 closed raw command
logs (93466690 logical bytes). Cleanup requires committed, verified compact
evidence under the exclusive measurement lock, fresh absence checks and exact
path/hash checks. Models, source, binaries, active fixtures and source-only
quiet-controller inputs remain retained. Removed logical/file allocation is
reported separately from measured available filesystem space.

The v2 model-width addendum is archived as lossless base64 JSON because its original extra EOF blank line failed staged whitespace checking. Its decoded bytes and original SHA-256 are preserved. The first publication attempt stays FAIL; the correction does not normalize or change the report.

## Completed retirement and next qualification

The [executed manifest](retention-executed-v1.json) records all 402 removals: 93466690 logical bytes and 95279104 allocated bytes. Measured filesystem available space changed by zero bytes between snapshots; the removed file totals do not prove exclusive space reclamation. Original receipts, fixtures, models, binaries and required source remain unchanged.

The [next registry](report-registry-v132.json) contains two additional fully reviewed Luna reports, 547 total entries. Actual agent state was checked: two researchers and a separate Sol implementer are running new scopes. The quiet v1 audit identifies order bias; its ON preflight requires prior OFF, so simply alternating commands cannot reverse pairs. The separate source-only v2 assignment must support balanced four-pair ordering, a strict immutable diagnostic reference for each first arm, a fresh same-index first-arm comparison for the second, and validated completed predecessors. Pending candidates remain unexecuted and performance-ineligible. Main owns all future tests and GPU runs.

The event-timer audit confirms adjacent compute-stream timestamps include idle/dependency intervals; they are not isolated dequant or DMA durations. Successful-path event lifetime was reviewed in source. Exceptional destruction semantics require a separate source/specification investigation and do not establish a bug. Its reference to an active 32K root describes a path it excluded; both actual 32K receipts were already terminal when this follow-up was assigned.
