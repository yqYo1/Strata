# QSA address reuse and enqueue grain: source-only proposal

Status: **unapplied, uncompiled, untested**. Base commit:
`0081480b1236b740aeba4fe8a066cfca310aba31`. This directory is the entire
worker deliverable; no live source, CMake, qualified test, branch, index or
runtime was changed. Main exclusively reviews, integrates and runs all checks.
No performance or inference adoption is proposed by this receipt.

`proposal.patch` adds `tests/sycl/qsa_resource_control.cpp` and a manual
`sycl_qsa_resource_control` target in `cmake/sycl_backend.cmake`. It is generated
from the named commit, with the new source available separately under `files/`.
`cmake.patch` is the same CMake hunk for review convenience; apply the combined
patch only once. No production kernel or existing fixture bytes are changed.
The target is `EXCLUDE_FROM_ALL`, requires `STRATA_SYCL_TESTS=ON`, and registers
no automatic CTest/GPU run. It requests C++20 and strict host floating point,
following the existing fixture's compilation controls.

The configured source resolver is root `CMakeLists.txt:193`, which includes
`cmake/sycl_backend.cmake`; that file's `strata_kernels` source list names
`src/kernels/sycl/prompt_attention.cpp`. The fixture links that target and calls
the public FP32 `qsa_prompt_attn_batch` directly, passing the owning runtime's
default in-order compute queue. A false capability result is an error, with
no alternate arithmetic path. The API returns a bool rather than its kernel
event; this proposal deliberately records no GPU event interval.

## Source origin and review

Implementation derives only from this fork's own MIT
`tests/sycl/prompt_k_cellmajor.cpp`, own public prompt attention and runtime
interfaces, and newly written host control/proof/timing code. `LICENSE` is MIT
with the original project attribution, retained in the new files. No external
implementation, XeStrata source/helper/test/archive, archived proposal, or LLVM
implementation was consulted or copied. No nested worker was created.

`provenance.json` pins every consulted base source and the separately consulted
instructions/research metadata. Consulted engine/API/build files match the
pinned base; the live `docs/RESOURCE_PROFILING.md` differs and is independently
hashed as policy context, never used as a patch preimage. R916's JSON and short
Japanese digest were read; only R914 lines 29–32 were retrieved. The index was
inspected as metadata, without loading unrelated report bodies. R914-C1 is
applied: the old benchmark already has resident pages; only widths and address
reuse need control. KMapping2 and HRows8 remain rejected and are not retried.
No upstream network check or synchronization is represented by this work.

## Controls and exact host-model argument

Controls are parsed once from argv before runtime creation. The fixture requires
these four exact values already present in the child's pre-exec environment:

```text
STRATA_SYCL_PROMPT_DIRECT=1
STRATA_SYCL_PROMPT_K_CELL_MAJOR=0
STRATA_SYCL_PROMPT_NATIVE_KV_MIN=0
STRATA_SYCL_PROMPT_SOFTMAX_SUBGROUP=0
```

It makes no process environment writes. Mode1 is selected by INT8 K/V pointers
and FP16 scale pointers, with all other pool format pointers null. The selected
source dispatch is `launch<1,true,false,0,false>`: Mode1, Direct1, Nativefalse,
KMapping0, Softmaxfalse. The native option does not affect Mode1, and is still
required to be zero to keep the receipt explicit.

Bench uses nq128 or nq256, page size4, 2 KV heads, 24 query heads, dimension256,
2113 useful base cells and cap2051. **Every query width is 2051**, and every
selected page is resident. All queries use the same ordered logical cohort
`(17*j) % 2113`, while queries themselves retain distinct deterministic FP32
values from the own fixture. There are no duplicate or missing benchmark IDs.

Let P=529, C=4P=2116, R=2C=4232, query b, base cell c, KV head h.
The allocated table contains nq replicas, each with P logical/physical pages:

```text
table[b*P + p] = b*P + (P-1-p)
shared_id(b,j) = c
replicated_id(b,j) = b*C + c
row_shared(c,h) = (table[c/4]*2 + h)*4 + c%4
row_replicated(b,c,h) = b*R + row_shared(c,h)
```

The complete K/V code and scale blocks in replica b are byte copies of replica0.
The proof actually checks ID translation, table and pool bounds, missing-row
equivalence, that row equality, and `memcmp` equality of both 256-byte code rows
and both four-element FP16 scale rows for **every active selected slot and KV
head**. It also checks finite host queries/scales. This is an executed check in
the proposed program, not a worker-run proof/test result.

The pinned FP32 entrypoint consumes only `kStepWidth` from the step vector and
uses IDs solely for the page/row lookup. It does not perform RoPE, causal
masking, positional bias, sorting, deduplication or index selection. It therefore
observes the same ordered numerical operands in both address arms. The other
step fields and query bytes are unchanged. Missing IDs stay -1; missing pages
remain missing per replica in smoke; duplicates keep their original sequence
slots and multiplicity. Invalid smoke widths normalize to zero exactly as in
the kernel. No positive out-of-bounds ID is submitted.

For enqueue grain, query order, tensor bytes, IDs, rows, output layout and queue
are identical. Calls advance q/out by `b*24*256`, IDs by `b*cap`, steps by
`b*kStepCount`, and pass `min(grain,nq-b)`. Grains are1,8,32,whole in bench.
All calls enqueue before one final checked drain; there is no inter-call wait.
The pinned `finish(non-null-stream,event)` does not wait. More grains also mean
more public-entrypoint capability/span checks; their host cost belongs to this
control's measured enqueue interval. There is no claim that only driver issue
cost changes: grid scheduling and available concurrency can change too.

Both address arms allocate, upload, and finally read back **all replicas and
both ID arrays**. The footprint and initialization are identical for a fixed
nq, including unused backing in the shared arm. Logical selected payload bytes
count each unique selected physical row once, with K+V codes and scales
(`2*256 + 2*4*2 = 528 bytes/row`). They exclude Q, IDs, table and output traffic.

| nq | Shared selected payload | Replicated selected payload | Total device allocations, guards included | One dump |
| --- | ---: | ---: | ---: | ---: |
| 128 | 2,165,856 B | 277,229,568 B | 294,681,024 B | 3,145,752 B |
| 256 | 2,165,856 B | 554,459,136 B | 589,361,088 B | 6,291,480 B |

These are source-derived extents, not measured traffic or capacity results.
Physical rows here mean indexed rows in the kernel's physical-page pool layout;
they do not describe hardware DRAM page placement or transaction addresses.
The source enforces total device data below1 GiB; the bounded argv geometry's
maximum is about562.06 MiB. Logical reuse counts selected row occurrences
across queries/KV heads divided by unique physical selected rows: nq for shared,
1 for replicas. This does not measure hardware loads, cache hit rate, occupancy,
SLM pressure, barrier stalls, physical bandwidth or arithmetic utilization.

## Numerical, timing and lifetime behavior

The compact `--smoke` uses nq3/cap67/cells79 and three cases: fully resident
uniform widths; missing pages/negative IDs/one repeated slot with widths13,31,67;
and widths0,-1,68. Each case checks both address arms at grains1,2,whole against
shared/whole, for18 full-output comparisons total, plus one full small FP64
oracle per case. These are proposed checks, all pending Main execution.
Existing broader qualified fixture coverage is preserved without modification.

Bench first obtains shared/whole reference output, checks every output finite,
and runs the inherited FP64 oracle at queries0,1,2,last, all24 heads and all256
dimensions. The tolerance stays `1e-5*(1+abs(reference))`. It then checks the argv
arm's complete output bit-for-bit against that reference, including prefix/tail
guards, and writes **one initial frame**. Two to five separate warmups follow,
each with a checked drain. Warmups do not count as measured repetitions.

Each of1–20 measured repetitions enqueues the whole nq workload and drains it.
The reported `steady_clock` enqueue, wait and whole-wall intervals are host
intervals; Linux `CLOCK_THREAD_CPUTIME_ID` gives corresponding issuing-thread
CPU intervals. The wait includes the checked runtime wrapper and idle transfer
queue drain, as well as any driver completion polling. Adjacent clock calls
contribute small instrumentation overhead and the boundary clocks are not
simultaneous. Wall subintervals partition the whole host interval; CPU time and
wall time must not be added. No timestamp is labeled GPU-only, and other
runtime/driver threads are outside this issuing-thread CPU measure.

No allocation, copy/readback, oracle, dump or logging occurs inside a timed
window. One compact sample line is emitted after each drain, preserving earlier
completed samples if a later repeat fails. Logging between repeats can affect
pacing; keep logging/supervision identical across arms. The final repeated
output is compared with the initial frame; there is **no per-repeat numerical
readback**, and that limitation is printed. Full input readback then verifies
all input bytes and head/tail guards, including unused backing and both ID
arrays, followed by a final output comparison. A sample remains unqualified
until final fixture, process, runtime and fresh fault checks pass.

All buffers use the own registered Device Allocation owner and live on the
same runtime/context. A scope drain is declared after every allocation and
before submitting inference. Copy methods additionally hold a drain guard
after their temporary host storage, so a partially accepted copy followed by
an exception cannot unwind its host source/destination early. Any checked drain
failure invokes the existing public `cleanup_fail_stop` (`_Exit`) before owner
destruction. There is no injected fault/hang and no unqualified recovery/free
after unknown completion. Kernel throws and false-return errors remain failures.

## Main-owned review and execution gates

1. Review the complete patch against the exact base and preserve any later live
   changes. Verify provenance and source hashes. Main may apply it to its owned
   isolated build tree only after the current serialized PR6 work closes.
2. Compile the dedicated target with the qualified SYCL recipe. This worker
   ran no compiler/preprocessor/CMake/build/test/native/provider/device/profiler
   command. Linux thread-clock availability and all C++/SYCL integration remain
   compile/runtime holds.
3. Supervise a90-second smoke process with the qualified trace/parameter
   validation environment fixed before native exec. Retain a bounded trace
   excerpt only for a distinct unresolved failure. Stop on an unsupported
   capability, numerical failure, checked error or fresh driver fault.
4. Run a120-second first bench per arm, or record a reviewed finite replacement
   budget **before** launch. Start with nq128. Trace/instrumented first checks
   qualify correctness/provider behavior; speed decisions use clean processes.
5. For address reuse, fix grain=whole and use shared versus replicated at each
   nq. Obtain at least3 independent processes per arm, with process pair order
   AB/BA/AB. For grain, fix reuse=shared (or a separately documented fixed
   replica arm), and compare whole against1/8/32 using at least3 independent
   processes per compared arm. Repetitions within a process are not independent
   process samples. No full factorial is required.
6. Use `compare_frames.py` on every successful same-nq cohort's full initial
   frame files. It verifies size/header, all finite FP32 elements and exact
   complete bytes, emits SHA256s, and returns failure for any difference.
   Also compare printed input/shape/count/queue/environment metadata. Different
   nq cohorts have different frames and must be compared separately.
7. Main records effective environment, exact source/diff/binary and runtime/
   driver pins, hardware, command/order, boot ID, PID/start/deadline/exit/cleanup,
   fresh fault-check boundaries and every individual sample including failures.
   Require final `fixture_status=PASS` plus supervised process/fault receipts;
   a dump or comparator pass alone does not qualify the run.

The illustrative argv forms, to be used through Main's owned supervisor, are:

```text
sycl_qsa_resource_control --smoke
sycl_qsa_resource_control --bench --nq 128 --reuse shared --grain whole --repeats 20 --warmups 2 --dump <private-initial-frame>
sycl_qsa_resource_control --bench --nq 128 --reuse replicated --grain whole --repeats 20 --warmups 2 --dump <different-private-initial-frame>
sycl_qsa_resource_control --bench --nq 128 --reuse shared --grain 8 --repeats 20 --warmups 2 --dump <different-private-initial-frame>
python3 compare_frames.py <successful-same-nq-frame-1> <successful-same-nq-frame-2>
```

Main supplies real private paths, qualified provider/environment and binary
location. These are not executed commands or a new supervisor implementation.
Keep at most one <=7 MiB raw frame per process, with named cross-arm comparator
consumer and review point at cohort closure. Total raw text budget is64 MiB/job.
After verified comparisons, record path/size/SHA256/reason in the deletion
manifest and retire successful raw frames/full traces under the artifact policy;
never alter original failure statuses, touch active runs or remove required
fixtures. This worker created no runtime captures and performed no cleanup.

A repeatable address-arm change would support cache/internal-memory/address
working-set sensitivity, including possible transaction/TLB effects. A grain
change would support sensitivity to public submission and/or grid scheduling.
Neither identifies a particular hardware bottleneck. Device-event timing,
physical traffic, cache hit rate, SLM/register residency, occupancy, matrix
utilization and any model tok/s conclusion remain unmeasured. Official long-input
benchmarks, task/quality checks, physical262144 and RESTORE remain Main-owned
requirements before any production inference adoption; this diagnostic changes
no production path and supplies none of those gates.

`manifest.json` records file extent and source-derived bounds. `SHA256SUMS`
pins all deliverables except itself; its own SHA256 is the completion snapshot
identifier sent to the coordinator. No patch application, compilation, numerical
pass, provider qualification or performance result is claimed by these hashes.
