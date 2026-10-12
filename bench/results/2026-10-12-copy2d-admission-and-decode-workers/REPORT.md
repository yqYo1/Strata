<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Copy2D admission and fixed-task decode worker comparison

The fresh remote check still resolves official main/v0.1.43 to 7695859 and
XeStrata main/xe0.1.42 to da3d426. Both revisions were already measured on 32K.
The combined source at 9b03c18 includes the required upstream server features.
There is no additional update to import. The pristine official measurements
remain diagnostics, not a ranking against fully optimized official settings.

Main reviewed the proposed expert Copy2D cohorts and added a shared production
admission scan for owned buffers, 16-row/tail/credit limits and source gaps.
The completed runtime investigation then exposed a decisive constraint:
the inspected UR Level Zero v2 context reports native 2D USM copies unsupported.
SYCL's later memory-manager branch emits one linear copy per row and an event
join. An extension call therefore cannot establish one native transfer command.
See the [memory manager source](https://github.com/intel/llvm/blob/504366f4b8/sycl/source/detail/memory_manager.cpp)
and [context capability](https://github.com/intel/llvm/blob/504366f4b8/unified-runtime/source/adapters/level_zero/v2/context.cpp).

The proposal was neither built nor executed on a GPU. It is retained as an
unqualified patch, including the separate Sol allocation-owner fixture.
All active engine/build files in this report branch were restored to their
byte-identical combined base. No capability bypass or hidden fallback was run.
The unresolved fallback event-pool concern in R291 is source analysis, not an
observed device fault. R292's unchanged-layout 1D option applies only to one
maximum-width layer: its command-count reduction ceiling is 1.95%, before
actual source gaps; it is not a speed measurement.

The next measurement uses the existing qualified binary, identical 32K input
and 16 output tokens. It compares workers 5/3/4/4/3/5 in fresh processes with
18 row tasks fixed, preserving every other setting. It checks effective INFO,
actual task/participant logging, CPU topology/affinity, exact token/LP5 output,
normal QUIT/reap and fresh kernel faults. Main's 36 CPU INFO checks and two
prelaunch rejection checks passed. No model ran for those host checks.

This is screening with two samples per worker count. Prefill and decode stay
separate, and no recipe promotion follows from short/noisy decode results.
All six measurements passed; individual samples are in worker-screening.json.
No worker setting was promoted. Full physical
262144-token input, RESTORE and the phase-lifetime proposal remain unqualified.

## Completed worker screening

| Sample | Workers | Prefill ms | Decode ms |
| ---: | ---: | ---: | ---: |
| 1 | 5 | 93276.5 | 2265.4 |
| 2 | 3 | 90732.6 | 2068.3 |
| 3 | 4 | 86378.0 | 1744.3 |
| 4 | 4 | 93587.7 | 2485.5 |
| 5 | 3 | 94182.3 | 2303.6 |
| 6 | 5 | 90918.1 | 2102.0 |

Both 4-worker samples differ substantially (1744.3 versus 2485.5 ms decode).
Means cannot remove that uncertainty; keep the selected recipe unchanged and
qualify a longer GEN128 comparison next. All reported token/LP5 pairs matched
the original reference; every sample closed normally without a fresh fault.
Closed duplicate logs/input copies totaling 1,049,760 logical bytes were retired;
original structured receipts remain byte-identical and all evidence is archived.

The existing paired IQ3_S GU candidate passed the CPU fixture in both free and
contrib-llvm modes, with its switch 0 and 1 in separate processes. The fixture
checks four production-quantized activation patterns, both scalar/gather variants,
640 rows, partition boundaries and rejection canaries. ELF dependencies exclude
SYCL/Level Zero. This is synthetic-kernel bit parity, not model accuracy, speed,
or a sanitizer result. The candidate stays OFF pending the remaining gates.

R295 corrects previous GDN accounting: the current source totals 144128 bytes
per token. Main checked the eight take expressions independently. Its 10240
chunk recommendation remains a source projection. For a 32K prompt, 10240 still
requires four chunks, including a small tail; it does not guarantee fewer expert
loads. Compare longer inputs and the tail route before selecting that setting.

## Resume boundary

The GEN128 controller proposal and its CPU fixtures are archived unchanged after
Sol FINAL, but have not been imported, reviewed or tested. Main must qualify the
new parser/admission and normal QUIT/STOP mocks before any long model run.

Main independently checked R297: the current no-MTP recipe rejects GEN0, while
positive-output admission reserves context for an output. The present protocol
cannot prove all 262144 input cells were processed with no output. Implement
and test a separate bounded drain before declaring full-capacity qualification.
R296 keeps serialized MTP and suffix lookup as distinct future candidates, with
all old failure conditions retained. The selected normal recipe is unchanged.
