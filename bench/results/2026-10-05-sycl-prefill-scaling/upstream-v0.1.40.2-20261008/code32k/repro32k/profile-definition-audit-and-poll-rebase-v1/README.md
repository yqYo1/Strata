# First repaired full image and profiling preparation

The indexer-spare candidate passes its first262140-input/4-output read on
the ArcB57010GiB / Ryzen5600X /128GiB host. Head, used prefill state, output
IDs/logprobs and MTP counts match the earlier full reference. It reaches
physical cell262143. SAVE contains262143 consumed visible-prefix IDs and
all13 main/MTP KV layers through262144 cells, with no tensor exclusions.
All12 saved main indexer spare rows now equal the live idx_dead key. The
following fresh32768-token input matches the accepted numerical control.
See [the completed-portion snapshot](full-first-v5-snapshot.json). Its parent
is still running the second full read; disk roundtrip, restored clipped
tail, refusals, later32K and owned exit/fault checks remain pending.

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
and link pass. The executable is GPU-untested and is not adopted. Its base
is the separately running indexer candidate; the polling policy is unchanged.

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
