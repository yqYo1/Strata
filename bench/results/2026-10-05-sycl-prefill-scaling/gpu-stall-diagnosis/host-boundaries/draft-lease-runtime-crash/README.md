# CPU runtime crash before a BCS fault, 2026-10-07

The actual Strata lease run crashes after completing its first normal-MTP
request. A CPU SIGSEGV is recorded before the GPU fault. The successful
default-mode checks and standalone mapping probe do not validate this path.
This is a project-triggered failure requiring investigation, not evidence
that the GPU happened to fail independently of Strata.

The frozen engine SHA-256 is
`9a2da2252d50b0a594f891490816fbf8f2f6bb385a972c0bbf599fdc07cbd74f`.
It uses the IQ3_S model, normal MTP, context 128, prefill 32, default Level Zero
V2, default copy offload and default direct submission. Persistent SYCL cache
is off. `STRATA_PREFILL_LAYER_MAJOR=2`, `STRATA_PREFILL_RELEASE_DRAFT=1`,
`STRATA_PREFILL_DRAFT_VERIFY=1` and `--short-read 0` force the release path.
The [original receipt](record.json), [engine stderr](normal-mtp.log),
[inner controller](controller-source.py) and [outer controller](outer-controller-source.py)
retain the actual inputs. The receipt's controller digest refers to the outer
controller; the inner checker has its own file digest in the manifest.

The first request emits `[760, 10849, 88851, 2272]`, matching the default
normal-MTP check. It completes two release/restore pairs, reports DONE and
parks a 42-token conversation. The repeat restores a checkpoint, reads six
prompt tokens and completes a third release/restore pair. Each release frees
939,524,096 physical bytes (896 MiB); each restore copies 931,016,700 payload
bytes. All payload comparisons pass before unmapping and after remapping.
The three measured RAM-to-VRAM copies take 176.085, 175.987 and 176.243 ms.
The repeat never emits DONE. Copy success does not prove that retained
command lists still hold valid memory-management objects.

The [kernel record](kernel-cpu-and-gpu.jsonl) establishes this order in JST:

| Time | Recorded event |
| --- | --- |
| 00:10:39.203062 | Strata PID 1183420 faults at address 0 inside `libze_intel_gpu` |
| 00:10:43.491993 | ASID 49 faults on BCS at `0x0000d556aa2e0000` |
| 00:10:43.492555 | Fault response returns `-ENOENT` |
| 00:10:43.492767 | BCS engine reset, GuC ID 6 |

The observed delay from the CPU fault to the BCS fault is **4.288931 s**.
The bounded outer controller did not hit its deadline. The kernel and Apport
records establish SIGSEGV; the old inner checker did not save `waitpid`'s
return status or every stdout allocation log, so those cannot be recovered
from its final exception. It did not retain even its completed first request
as JSON before the later failure. This diagnostic gap is fixed in the
[new checker](../../../../2026-10-05-sycl-upstream-arc/serve_check.py).

## Exact binary symbolization

The installed package is `libze-intel-gpu1 26.31.39395.14-1~24.04~ppa1`.
Its [debug package](debug-symbol-package.json) was downloaded and extracted
in private diagnostic storage; nothing was installed. Both ELF files have
build ID `a150ba51180689c9ab166533b5d6503441b9e1c1`:
[installed notes](installed-library-notes.stdout),
[debug notes](debug-symbol-notes.stdout).

The fault IP minus the module's mapping base is `0x7ca4d9`. Exact matching
symbols resolve it to
`NEO::GraphicsAllocation::prepareHostPtrForResidency(NEO::CommandStreamReceiver*) [clone .part.0]`:
[symbolization](symbolization.stdout), [instructions](faulting-instructions.stdout).
The tagged source reads the allocation's per-context task information in
this function before making it resident. The observed faulting instruction
is the read of that task information. The source and package identity are
recorded in [diagnosis.json](diagnosis.json).

There is no core file: the limit was zero and Apport ignored the executable
because it is not owned by an installed package. The relevant
[Apport messages](apport-filtered.log) are saved. An exact call stack, register
values and the identity of the invalid allocation are unavailable. The
disassembly and symbol do not distinguish an old physical allocation from
an old host-copy allocation or other corrupted residency metadata.

## Causal interpretation and prevention candidate

The BCS address agrees with the lower 48 bits of the direct-submission
semaphore address recorded in healthy controls. That supports investigating
fatal process teardown while the copy submission ring is still live. The
failed process's allocation log was lost, so its exact allocation type is
not confirmed. A client BCS reset alone also does not establish the earlier
machine-wide migration-queue stall or the GPU's current usability.

The project kept all MTP executable graphs across physical unmap/destroy and
remap. Backend command lists can retain `GraphicsAllocation` references as
well as virtual addresses. Stale residency is a leading hypothesis for this
repeat-specific failure, not an established identity for the crashing object.
The prevention candidate drains consumers and destroys the graphs before
unmapping, then captures fresh graphs after restoration. The target model's
expert cache now follows the same retirement/rebuild order even for unchanged
virtual addresses. The [candidate checks](../graph-retirement-candidate/README.md)
cover ordering and error handling on CPU and a successful build. It has
**not** been executed on the GPU; no recovery or prevention success is claimed.

The installed NEO tag lacks upstream commit
`64ad4a0f37213f6c01e4ba88dee995645aa4d364`, which releases bindless surface
state when virtual memory is unmapped, and
`3efe60fca42a8b7da186d6b21d0677e02d1e9c51`, which fixes memory-free callback
handling. Neither commit is an ancestor of the previously used 26.35 tag;
that tag also retains the old implementations. Their source
changes are relevant audit material, not proof that changing the driver fixes
this crash or explains the original stall. No runtime update was applied.

Additional GPU tests stopped when this fault appeared. Remaining work is to
identify the invalid object/caller, validate the candidate without intentional
fault reproduction, and only then resume cancellation, checkpoint, partial
release and exact 262,144-context model tests.
