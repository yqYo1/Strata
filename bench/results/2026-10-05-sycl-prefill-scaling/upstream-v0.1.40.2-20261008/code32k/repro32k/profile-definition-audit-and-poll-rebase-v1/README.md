# First repaired full image and profiling preparation

The indexer-spare candidate passes its first262140-input/4-output read on
the ArcB57010GiB / Ryzen5600X /128GiB host. Head, used prefill state, output
IDs/logprobs and MTP counts match the earlier full reference. It reaches
physical cell262143. SAVE contains262143 consumed visible-prefix IDs and
all13 main/MTP KV layers through262144 cells, with no tensor exclusions.
All12 saved main indexer spare rows now equal the live idx_dead key. The
following fresh32768-token input matches the accepted numerical control.
See [the completed-portion snapshot](full-first-v5-snapshot.json). Its parent
was running the second full read at capture. It subsequently stopped in
the host watchdog; see [the terminal evidence and separate32K definition check](../full256k-watchdog-abort-and-uniform32k-v1/README.md).
Disk roundtrip, restored clipped tail, refusals and later32K were not reached.

Actual compiler preprocessing shows that source-local
DPCT_PROFILING_ENABLED changes two definitions in dpct/device.hpp. The
translated verifier adds a profiling queue property and records an event
with a barrier; hand-written conversation state omits the property and
records a single task. These are definitions of the same class-inline
factory and external-inline function across translation units. Matching
tokens are required by the [C++ one-definition rule](https://eel.is/c++draft/basic.def.odr).
This is a definition inconsistency; it does not establish that it caused
an earlier hang or that these are the project's only C++ correctness issues.

The private header candidate keeps the translated profiling branch
uniformly. Preprocessed factory/barrier tokens now match across both actual
commands and match the old translated definitions. The explicit copy-queue
factory retains its supplied profiling property. Original and candidate
headers, exact preprocessed excerpts and diff are retained. A single-header
include prefix initially failed to override dpct.hpp's relative include;
the corrected full shadow tree and compiler -H/dependency output prove
which header was used. Earlier parser/selection negatives are preserved.

The original strata link has116 compiler commands and67 users of this
device header. Seven users lack the source-local profiling macro. They are
recompiled with unchanged source, flags and numerical kernels, using the
uniform shadow header. Only those seven objects in three copied archives
are replaced; every other member/link input remains unchanged. Compilation
and link pass. The executable was GPU-untested at this preparation snapshot
and is not adopted. Its separate first32K numerical control subsequently
passes in the follow-up linked above. Its four fresh32K reads, actual disk
continuation and normal owned exit also pass there; the physical256K gate
remains pending. Its base is the indexer candidate; the polling policy is unchanged.

A separate executable rebases the CPU-tested polling object onto the
indexer candidate. Only its prefill archive changes; its header, object
and sanitizer/deadline/cancellation evidence remain unchanged. It is also
GPU-untested. Future polling comparisons must use the proved compatible
baseline; no synthetic query count is reported as an inference improvement.

Installed VTune2026.4.0/build632893 help accepts xpu-offload with CPU sampling,
hardware metrics, bandwidth, power and stacks disabled and programming API
tracing enabled. Only version/help are executed. No collector or GPU
support claim is made. [Intel's release notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/vtune-profiler/current.html)
describe improved reused LevelZero command-list/SYCL graph tracing, excluding
native graphs; the hardware list does not explicitly validate this B570.
Actual timestamp/graph coverage and a full32K numerical comparison are
required before using a trace to identify bottlenecks.

All speed comparisons require at least32768 input tokens and separate
initial requests from repeated full reads. Diagnostic/capture durations
do not count as throughput. Large payloads, preprocessed files, help output
and binaries remain private with hashes. No reset, rebind, reboot, service,
package or global setting is changed. Production, accepted binaries and
main remain unchanged; PP1000/TG70 remains unachieved.

The [subsequent GPU timestamp rejection and polling gate](../device-profile-rejection-and-poll32k-v1/README.md) retain the actual earlier VTune startup failure on this Ryzen processor and a failed logged 32768-token unitrace GPU timestamp run. The application watchdog stopped at layer 26 of chunk 16384; no new kernel fault was recorded and post-exit GPU/runtime health passed. Its empty/incomplete trace and durations are excluded from performance conclusions. The DD5-based polling candidate subsequently passed four fresh 32768-token reads, all head/used-state/output/logprob/MTP comparisons, actual disk continuation and normal exit. It is still private: quiet matched inputs of at least 32768 tokens and the complete physical 256K gate remain required. First and repeated full reads must be reported separately; the underlying pending-counter cause is unresolved.

The [subsequent quiet polling comparison](../poll-matched-quiet32k-v1/README.md) completed sixteen fresh 32768-token reads in control/candidate/candidate/control order on the same B570. First process reads are separate from later full reads. Subsequent prefill was 427.510 tok/s for DD5 and 427.225 tok/s for polling backoff (-0.067%); decode was 16.486 and 16.373 tok/s (-0.686%). All output IDs, logprobs and visible MTP counts matched, with RESUME 0 and REUSED 0, normal QUIT and no new kernel fault. There is no useful prefill gain, and this small sample does not establish a decode benefit or regression. Polling backoff is not adopted. At that quiet 32K boundary, the complete physical 256K gate and pending native counter diagnosis were unresolved; diagnostic/profiled durations are excluded.

The [later complete DD5 physical 256K sequence](../uniform-full256k-and-qsa-source-v1/README.md) passed two fresh full reads through cell 262143, all 13 saved KV layers with 262144 cells, every saved state/KV tensor byte on repetition and actual disk restoration, a clipped two-token tail, both capacity refusals and a later fresh 32768-token input. Head/used-state/output/logprob/MTP gates and normal owned exit passed without a new GPU fault or reset. Diagnostic durations are excluded; this does not resolve the earlier pending-event cause. A separate twelve-head QSA reduction has CPU arithmetic/routing checks and source-only preparation, with no engine compile, GPU numerical result, speed gain or inherited capacity proof at this archive boundary. Every future performance comparison uses at least 32768 input tokens and separates first/later full reads.
