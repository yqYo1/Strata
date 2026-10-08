# Residency and host-upload lifetime audit, 2026-10-07

The starting failure is Strata's repeat-request SIGSEGV in
`NEO::GraphicsAllocation::prepareHostPtrForResidency`, followed 4.288931 s
later by a BCS page fault. The
[actual crash](../draft-lease-runtime-crash/README.md) has matching installed
library symbols, but no retained core identifying the invalid allocation.
This audit traces a project path capable of invalidating that kind of reference.
It does not identify the actual crashing object or establish GPU prevention.

## Recorded references survive a virtual-address change in backing

The NEO source below is pinned to the installed package's upstream tag
`26.31.39395.14`, commit `050536ff3b6f830199ec99cff8ffcac3d804c40a`.
Full source digests and the extracted functions are retained in the
[CPU mechanism record](neo-cpu/record.json).

1. `KernelImp::setArgBufferWithAlloc` stores a raw `GraphicsAllocation*` in
   `privateState.argumentsResidencyContainer[argIndex]`.
2. Recording `CommandListCoreFamily::appendLaunchKernel` copies argument
   residency into the command container. `CommandContainer::addToResidencyContainer`
   stores the raw pointer; it does not look up the address on later replay.
3. `Context::destroyPhysicalMem` frees the physical node's graphics allocation.
4. `CommandQueue::makeResidentForResidencyContainer` dereferences each stored
   allocation and calls `prepareHostPtrForResidency`. This is the function
   identified by the actual crash's matching symbols.

The tagged kernel setter can refresh its allocation lookup when the allocation
ID changes, but a previously recorded list already has its own residency vector.
Refreshing only the kernel does not refresh that vector. Source references:
[kernel arguments](https://github.com/intel/compute-runtime/blob/050536ff3b6f830199ec99cff8ffcac3d804c40a/level_zero/core/source/kernel/kernel_imp.cpp),
[recording](https://github.com/intel/compute-runtime/blob/050536ff3b6f830199ec99cff8ffcac3d804c40a/level_zero/core/source/cmdlist/cmdlist_hw_xehp_and_later.inl),
[physical destruction](https://github.com/intel/compute-runtime/blob/050536ff3b6f830199ec99cff8ffcac3d804c40a/level_zero/core/source/context/context.cpp),
[queue residency](https://github.com/intel/compute-runtime/blob/050536ff3b6f830199ec99cff8ffcac3d804c40a/level_zero/core/source/cmdqueue/cmdqueue.cpp).

The CPU test uses these real functions and recording fragments. Hardware,
mapping, argument setup and the graphics-allocation layout are explicit
stand-ins. It deliberately destroys one allocation object and creates another
with the same simulated virtual address. ASan quarantines the freed object;
this is a controlled mechanism, not native NEO execution or an actual GPU fault.

| Controlled lifetime | Result |
| --- | --- |
| Keep old list, destroy and replace physical object | ASan heap-use-after-free |
| Update kernel argument, keep old list | ASan heap-use-after-free |
| Retire list before destruction, record again after replacement | Three cycles pass |
| Keep physical object alive across replay | Three cycles pass |

The graph-retirement candidate already drains consumers and discards all six
MTP graph arrays before physical destruction. Verifier cache leases retire and
rebuild their graphs too. These changes are supported by the source mechanism
above. An allocator reusing an object's address could conceal this defect in a
smaller probe; that is an explanation to test, not a measured explanation for
the earlier nine successful real storage-probe rounds.

## API contract audit

The current official
[SYCL virtual-memory specification](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_virtual_mem.asciidoc)
treats accessible mapped pointers as device-USM pointers. It requires the
original mapped range for unmap, one context, and unmapping before physical
object destruction. Strata unmaps individual segments, with rounded reservation
and physical sizes, rather than a subrange or one unmap spanning separate maps.
The [Level Zero physical-memory API](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api/apis/physical_mem.html#zephysicalmemdestroy)
requires device references to have completed before physical destruction.

Those conditions do not establish that an old executable graph can survive
replacement of its backing allocations. The audited specification text does
not explicitly settle that graph/VMM combination. The implementation must
retire its graph references before destruction rather than relying only on
same-address payload restoration. This is an implementation lifetime defect
candidate, not an asserted specification prohibition on every VMM remap.

Current upstream UR V2 source also records kernel launches into native command
lists. It is a comparison only: that branch was fetched during this audit and
has not been identified as the exact installed July compiler revision. Its
URLs, dates and hashes are in [the fetch receipt](upstream-fetch.json).

## Definite project host-source lifetime defect

In the real `MtpDrafter::prefill` function, the batched device-input path queues
an upload from local `std::vector<int32_t> rec`, then creates and submits graphs.
If graph capture returns false, `rec` is destroyed before the final queue wait.
A later queue drain can still read its freed storage. This is separate from the
unidentified allocation in the actual repeat crash: that run did not log a
capture failure before its CPU fault.

The [host-upload test](mtp-upload-cpu/record.json) extracts the complete actual
prefill function from control commit `990a631` and the candidate. Its CPU queue
defers copies and injects failures; it runs no GPU kernels or model arithmetic.
The control capture-failure case reports ASan heap-use-after-free in the deferred
copy. A second control returns success with a queued asynchronous error because
`wait()` does not consume it. Both controls are expected failures of old behavior.

The candidate waits for the initial upload while the source remains alive and
uses `wait_and_throw()` for the completion boundary. This adds one upload wait
per chunk; the device groups remain batched. Seven candidate cases pass:
batched success, capture failure, second submission failure, upload rejection,
final asynchronous error, and success/error with per-group synchronization.
No GPU timing change or prevention result is claimed.

The interactive `VRAM` command is another physical-resize path. It lacks graph
retirement and recovery of the residency table after partial failure.
`--vram-elastic` is therefore rejected by the SYCL executable before device
selection until this path is implemented and validated. An actual executable
test with valid nonexistent selector `level_zero:999` returns 2 for both ordinary
and 262,144-context elastic requests; help returns 0. See
[preflight](preflight/record.json). The context argument in that test is only
configuration validation, not consumption of any context cells.

## Validation before the external reboot

The current candidate builds successfully with the existing oneAPI/MKL
toolchain: [build](build-record.json). Its frozen SHA-256 is
`79a4b363d33f66f41d910be6274e609b4eb73f62afb0dc49bca72bf2a538d223`.
At that point it had not been submitted to the GPU. A failed first link omitted MKL's library
search path; the corrected build environment and both logs are retained.
The earlier graph-retirement candidate is a different binary.

Before a model comparison, the installed small health executable was run once,
with persistent cache off, stock NEO, UR V2, copy offload disabled and direct
submission disabled. It selects the B570 and exits 1 after **12.8724 s** with
`UR_RESULT_ERROR_OUT_OF_RESOURCES`. It does not hit the controller's 15 s limit
and leaves no health process alive. See [receipt](gpu-health/record.json),
[stdout](gpu-health/health.stdout), [stderr](gpu-health/health.stderr) and
[kernel messages](gpu-health/kernel-filtered.json).

At 01:45:30.274983 JST, GuC ID 0, sequence 44,744 is reported as not started.
At 01:45:35.394984 its scheduling-disable request has no response. The captured
window records 37 reset starts and 37 reset completions, 36 messages for jobs
in no process, followed by a timeout naming the health process's own queue.
Queue flags `0x73` decode as KERNEL, PERMANENT, HIGH_PRIORITY, LOW_LATENCY and
MIGRATE in the
[kernel definitions](https://github.com/torvalds/linux/blob/v7.0/drivers/gpu/drm/xe/xe_exec_queue_types.h).
This identifies an internal migration queue; it does not identify the new
dump's engine instance or the first stalled instruction.

The boot remains `ef28c8b7-46f0-4806-9326-36e3158bceb7`, the same as the
earlier Strata crash. The health failure is post-fault evidence and cannot
exonerate Strata or establish an independent original trigger. Its new dump is
root-readable only; normal reading and noninteractive sudo could not preserve
it: [access result](gpu-health/dump-read.json). No reset, rebind, service shutdown
or runtime update was attempted. The model comparison did not start, and further
GPU submissions stopped after this health failure.

Failed diagnostic attempts are retained in [harness controls](harness-controls.json):
the first ASan control exceeded its 5 s limit while aborting; disabling core
abort made its expected report deterministic. An initially malformed selector
was rejected by runtime parsing before the corrected preflight. These are
harness failures, not passing checks or evidence of another GPU failure.

The user has authorized returning to tuning once there is enough prevention
information. Further cause investigation is deferred; the invalid object remains
unidentified. Before GPU performance comparisons, remaining gates are candidate
model validation on a usable GPU, cancellation/checkpoint/partial-restoration
paths, and consuming all 262,144 cells in both CLI and normal-MTP serve. CPU
work can continue while GPU validation is unavailable. Kernel reset completion
messages alone do not make those checks safe or prove recovery.

## Subsequent GPU validation

After an external reboot, this same frozen executable runs the actual model:
[post-reboot comparisons](../post-reboot-retirement/README.md). Four processes
complete 16 requests and ordinary exits, without new xe faults, with direct
submission disabled. Both retained controls pass full-head equality. Releasing
draft weights and recreating graphs still changes repeated target-head results,
including with prompt/conversation reuse disabled. The release optimization
is not accepted; the earlier health failure describes the previous boot,
not the current operational GPU state. Full capacity remains unvalidated.
