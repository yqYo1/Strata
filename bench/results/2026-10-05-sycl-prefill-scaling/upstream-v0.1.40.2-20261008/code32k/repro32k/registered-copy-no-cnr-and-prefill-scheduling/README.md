# Registered nonprofiling copy: no CNR and32K prefill scheduling comparison

Hardware: Arc B57010GiB, Ryzen5 5600X,128GiB RAM, kernel7.0.0-38,
NEO26.31.39395.14 and oneAPI2026.1.1, on the same boot as the earlier controls.
The [registered-copy candidate](../event-ack-registered-copy-cb-cnr/README.md)
is unchanged at SHA256 f02213fc440a1f11b64857fceaa028ab43eeee7fdfb40e5cec33abca411a9705.
Removing only MKL_CBWR=AUTO permits three fresh32K executions to match all66
prefill state parts, all248,320first-head float bytes, all64IDs and every
protocol logprob with the production counter-conversion-off control and each
other. This demonstrates that CNR is unnecessary for these tested controls.
EnableImplicitConvertionToCounterBasedEvents=0 remains set in every run here.

A scheduling-only candidate at SHA256 e82fc5de5480b72255b601759497d5810e86bf691350417ed0dc11ed6c429323 removes
exactly14 use_root_sync properties in the prefill kernel translation unit.
Every kernel body, arithmetic operation, ND-range, subgroup attribute, queue,
barrier, explicit release drain and graph operation remains unchanged.
The [build receipt](build/record.json) verifies all other archive members and
link inputs remain identical and the matched registered-copy build is unchanged.
All compilation uses the same matched header definition in every translation unit.
The first logged32K check and two fresh unlogged state/head repeats also match
all66state parts, full head and complete64-token output with the registered-copy
no-CNR control. All six state/head jobs finish normally without new xe faults.

The subsequent [clean ABBA sequence](sequences/clean-abba.json) uses registered /
no-root / no-root / registered, each as a fresh process. Every request has32,768
input tokens and64greedy output tokens, normal MTP4, context33024,8192-token
chunks, int8 KV, cache128, five CPU workers, pcie0, FIRST0 and RING8.
Prompt/conversation caching is off; no short-input timing is used. Timing comes
from engine DONE intervals, excludes model startup and includes cold request
JIT/graph preparation. No API tracing, validation, state/head dumps, transfer
profiling, profiler or additional waits are enabled. An uninterrupted owned
GDB/PTY observer is common to both arms. Each clean run requires all64IDs and
every protocol logprob equal the logged control, normal exit, complete owned
cleanup and no new xe fault; all four pass.

| Private configuration | Clean repetitions | Prompt tok/s | Decode tok/s |
| --- | ---: | ---: | ---: |
| Registered copy, original prefill launch properties | 2 | 406.78 | 16.96 |
| Registered copy, without14 prefill root-sync properties | 2 | 408.52 | 16.06 |

These are means of two runs per arm. Relative no-root change is
+0.43% for prompt and
-5.34% for decode; two runs do not
establish statistical significance. The no-root decode runs are17.11 and15.01
tok/s, versus17.03 and16.90 in the registered control. The second no-root run
has slower decode despite identical output and a changed prefill-only launch
property; these data do not justify a decode regression or gain attribution.
The prompt difference is below the registered control's within-arm spread,
so this experiment does not establish a useful prompt speed improvement.
Only the scheduling property differs in
this comparison. The earlier unmodified-upstream393.65/17.14 rates are a
separate completed sequence with different resident cache packing and timing;
they are not contemporaneous paired controls for these two private candidates.

The [root-group extension specification](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_root_group.asciidoc)
requires use_root_sync for cross-work-group root-group functions and limits
the permitted work-group count. No root-group calls or cross-work-group atomic
coordination occur in this prefill translation unit. Its independent groups
use local barriers and partition recurrence state by head/column. The installed
2026.1 kernel_launch_helper.hpp also marshals the property. This supports removal
as a scheduling experiment; it does not prove the property caused an earlier
stall or that an installed backend takes a particular cooperative launch path.
Core persistent/global coordination kernel properties are retained.

Both candidates remain private and unadopted. These completed32K jobs do not
prove general prevention of GPU/native-event waits, default counter-event
conversion, full262,144-cell serving/repeats/restore, or the PP1000/TG70 goal.
Diagnostic and state/head job timings are excluded from the speed table.

The first scheduling-controller preparer fails before writing a controller or
launching a GPU process: its string replacement also changed a CNR control
prefix. A separate v2 preparer targets exact mode and configuration names and
retains the earlier failure source. Executed controllers are never overwritten.
Full multi-GiB API logs, state/head captures and the private binary remain outside
Git, with hashes and byte counts in private-artifacts.json. Public records retain
the exact environment/argv, fixture, protocol, project progress, journal probes,
build provenance and process ownership. No reset, rebind, reboot, service,
package or system setting change is performed.
