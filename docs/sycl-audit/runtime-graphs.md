# Runtime graph capture and replay

`strata::sycl_upstream::Runtime` now records and replays native SYCL command
graphs. Recording does not execute the commands. This extends the isolated
[stream/event domain](runtime-streams.md); it does not yet connect the original
engine's CUDA calls to that domain.

## Source and specification requirements

The pinned `src/core/graph.cpp` begins thread-local capture, rejects a zero-node
capture, instantiates it, and records a reusable completion event after every
launch. `src/core/session.cpp`, `src/core/verify.cpp` and `src/core/mtp.cpp` also
capture work on temporary streams. These call sites require recorded commands
to survive the recording stream and to read changed inputs on later replays.
The runtime inventory finds 21 `cudaStreamBeginCapture` candidates; this is a
source scan, including harnesses and inactive code, not 21 validated engine paths.

The implementation follows CUDA's [capture restrictions and graph execution
ordering](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html):
end capture on its origin stream and host thread, rejoin captured event branches,
and order launches of one executable behind its prior launches even across
streams. Completion events are recorded outside capture, following the original
`CapturedGraph::launch` pattern.

The comment in upstream `CapturedGraph::reset` says destroying an executable
blocks. CUDA's guide explicitly shows destruction after an asynchronous launch
without synchronizing that launch. The new runtime follows the documented
asynchronous behavior: releasing the public graph handle does not wait for
pending replay. This discrepancy is recorded here; the original source remains
unchanged, and no CUDA hardware comparison has been run.

SYCL recording, finalization and replay use the [experimental command graph
extension](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc).
The implementation was compiled against oneAPI 2026.1.1. Support on other
versions or devices has not been established.

## Implementation and lifetime

Each capture owns a modifiable native graph. Participating queues retain their
normal execution tails separately from captured tails. Waiting on an event
recorded in an active capture enrolls the waiting stream and records the native
dependency. Before finalization, the runtime checks that the origin's final node
is the only leaf in the native DAG. This checks that all branches rejoin without
an ancestor search for every node. Node counts describe SYCL commands, including
dependency markers; they are not claimed to equal CUDA node counts.

Invalid captures detach all participating queues and cannot produce an executable.
Synchronizing an active capture, merging two independent captures, or submitting
legacy work while an ordinary blocking stream is capturing invalidates the
associated captures. Device synchronization invalidates all active captures.
The origin thread can explicitly abort recording. A failed submission callback
invalidates recording without emitting an executing recovery command.

Replay submits `handler::ext_oneapi_graph` with the previous replay's event as a
dependency. It also preserves the runtime's ordinary/default-stream ordering.
There is no host replay loop or manual subdivision of the graph. The runtime
retains each launched executable before its first submission, so allocation
failure cannot leave an enqueued executable without an owner. Later submissions
reuse that retention without querying completion. Released, completed graphs are
reclaimed at stream creation, capture creation or device synchronization, or at
runtime teardown after waiting for pending work. A retained public graph can
outlive its runtime; runtime teardown first waits for that runtime's work.
Recorded pointers remain caller-owned and must stay valid until their uses finish.

A captured event describes an internal graph dependency, not an external
completion event. Querying or synchronizing it is rejected. Re-recording the same
event outside capture restores normal completion behavior. Waiting inside capture
on a previously executed external event, nested executable launches, cross-context
launches and merging independent captures are explicitly rejected. CUDA capture
modes other than the implemented origin-thread contract, graph updates, external
event nodes, debug export and full CUDA error behavior remain unimplemented.

## B570 evidence

On Intel Arc B570, Linux, Level Zero, oneAPI 2026.1.1, 2026-10-04, both JIT and
`bmg_g21` AOT pass the following six graph scenarios, with 32 total replays:

- Capture two dependent kernels on an ordinary temporary stream, destroy that
  stream, then replay 12 times on two nonblocking streams with changed inputs.
  Completion events are recorded again after each replay.
- Fork through a captured event to a second stream and rejoin the origin. Eight
  changed-input replays produce the expected values. A captured event cannot be
  queried externally until it is recorded again outside capture.
- Record pinned H2D copy, a kernel, D2H copy and host function. Five replays each
  observe the new input and invoke the callback once; capture itself invokes none.
- Reject legacy, empty, duplicate, wrong-origin, wrong-thread, unjoined and
  invalidated captures; reject a throwing body and a merge of independent
  captures. Device synchronization invalidates both independently active captures.
  Discarded writes never execute and subsequent ordinary submission succeeds.
- Hold a preceding host task and submit the same graph on two streams. Both
  submissions and public graph release return before the gate opens. The second
  stream remains pending, including after stream management, and all three
  increments (one warmup plus two pending replays) complete in order.
- Capture the ported Q8 MMQ chain: iota, activation quantization, gate/up product
  and Stream-K fixup, SwiGLU, hidden quantization, down product and fixup. This
  records eight native nodes and retains caller-owned scratch. Four distinct
  inputs produce 6,144 outputs exactly equal to direct execution of the same
  chain, with all outputs finite. Dimensions are K=512, hidden=512 and three
  input rows, one expert. This is graph-versus-direct evidence; the separate
  [product](mmq-product.md) and [stage](mmq-stages.md) tests provide CPU references.

The seven pre-existing runtime ordering scenarios also pass in both builds,
including the 40-batch prefill transfer ring. The unchanged MMQ component suites
were not rerun for this runtime-only extension. [Results](runtime-graphs-results.json)
include source/binary hashes, commands and logs. Test elapsed times are correctness
harness times, not inference throughput. The 256 original source/build files in
the shared manifest remain byte-identical.

The initial test invocation without the oneAPI environment found no GPU. After
loading `/opt/intel/oneapi/setvars.sh` and selecting `level_zero:gpu`, both builds
passed. This was an environment-selection failure before any scenario executed.

[Mapped-host CPU/GPU handoff](runtime-handoff.md) now has separate B570 evidence.
Whole-model graph capture, the CUDA frontend binding, allocation APIs, asynchronous
device-error recovery and complete engine integration remain open. These component
results do not establish full upstream parity or a new prefill/decode speed.
