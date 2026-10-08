# Copy-engine semaphore and serve shutdown, 2026-10-06

The first two retained B570 faults share GPU address `0x0000d556aa2e0000`
and BCS0. They occur at October 5 16:24:54.289299 and 16:25:52.945485 JST.
The normal and long AOT MTP reports were written 61.604 and 52.622 ms later,
respectively. Both controllers save these reports after `QUIT` and process
exit; both engines returned zero with the same frozen binary. See
[shutdown-associations.json](shutdown-associations.json) and
[the actual kernel records](first-two-faults.kernel.json). File timestamps
do not identify the faulting ASIDs or give exact process-exit timestamps.

Three bounded 64 KiB application-buffer probes then inspected actual allocation
logs on this B570, stock NEO 26.31.39395.14, Level Zero V2, persistent cache off.
Each performs three H2D/kernel/D2H rounds and checks all 16,384 integer words.
All return normally with no new kernel fault/reset messages. Internal runtime
allocations add about 200 MiB beyond the small application buffer.

| Process environment | Allocation observation |
| --- | --- |
| Copy offload disabled | One `SEMAPHORE_BUFFER` at `0xffffd556aa590000`; no BCS semaphore at the recorded fault address. |
| Default copy offload | A second `SEMAPHORE_BUFFER`, 65,536 bytes, starts at `0xffffd556aa2e0000`. Its lower 48 bits exactly match the original BCS fault address. |
| `NEOReadDebugKeys=1 EnableDirectSubmission=0`, default copy offload | No `SEMAPHORE_BUFFER` or `RING_BUFFER` allocations. The same VA falls inside a `COMMAND_BUFFER` in this different allocation sequence; matching a VA across configurations alone does not identify an object. |

The earlier failing processes have no allocation logs. This is an exact match
in a current healthy controlled process, not retrospective PID/ASID attribution.
The environment change was confined to the probe; no running service, host
configuration, GPU binding or firmware changed. Disabling direct submission
removes the observed internal wait mechanism; these small passes alone do not
establish a lower failure rate or a complete Strata fix.

NEO's Linux direct-submission destructor calls `stopRingBuffer(true)`, waits
for its completion fence, then releases its resources. The stock and earlier
user-local versions have identical destructor source. See the pinned
[26.31 source](https://github.com/intel/compute-runtime/blob/26.31.39395.14/shared/source/direct_submission/linux/drm_direct_submission.inl#L90)
and [26.35 source](https://github.com/intel/compute-runtime/blob/26.35.39758.10/shared/source/direct_submission/linux/drm_direct_submission.inl#L90).
Strata's old successful serve shutdown used `_Exit(0)` after a queue wait whose
exceptions were swallowed. `_Exit` skips runtime destructors; completion of
application queue work does not invoke the internal ring-stop destructor.
Together with the two shutdown-time associations, this makes that Strata exit
path a strong trigger candidate. It does not establish how the xe recovery
path later became permanently wedged.

The candidate Linux serve exit now stops and joins its stdin reader and
watchdog, checks all queue completion, releases its raw main resources and
returns through ordinary C++ destruction. Failed queue completion still exits
with failure without unsafe reclamation. The Windows branch is unchanged;
this port's supported build path is Linux. The real-pipe CPU test preserves
STOP/CRLF/EOF handling, unblocks an open-pipe shutdown and preserves a complete
262,144-token protocol line. This input test is not full-context GPU validation.

Intel's [related report](https://github.com/intel/compute-runtime/issues/948#issuecomment-5536265115)
also pairs a similar internal semaphore VA with a kernel fault, while marking
its residency diagnosis unconfirmed. Our inference is based on the local
observations above. [evidence-manifest.json](evidence-manifest.json) pins their
logs and both source versions. A real short MTP check initially crashed during
ordinary destruction: an unused Strata verifier had two aliases of a borrowed
default queue, destroyed it through one alias and waited through the other.
Correcting owned queue initialization makes the short model/checkpoint/shutdown
check pass with direct submission disabled. See [the captured failure and fix](../queue-ownership/README.md).
The final guarded engine also passes the short MTP/checkpoint/shutdown check
with default direct submission and no new xe faults. Its actual allocation log
again identifies a semaphore at the historical fault VA. This strengthens the
object association without recovering the past ASID's allocation history.
Full-capacity arithmetic/state equality and repeated fault-rate checks remain
separate gates.
