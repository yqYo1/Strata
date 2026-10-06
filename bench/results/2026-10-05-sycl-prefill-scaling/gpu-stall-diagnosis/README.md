# Arc B570 failure investigation, 2026-10-06

The observations distinguish a reproducible CPU-side SYCL cache crash from
the later xe/GuC failure. The first Strata stall's exact userspace location was
not captured before termination. Neither a small GPU test nor a successful
build proves the in-place residual change or full 256K execution correct.

Hardware: Arc B570 10 GiB, Ryzen 5 5600X, 128 GiB RAM. Kernel:
`7.0.0-38-generic`, Ubuntu package `7.0.0-38.38~24.04.4`. SYCL runtime:
oneAPI 2026.1.1. Stock compute runtime: `26.31.39395.14`; user-local runtime:
`26.35.39758.10`. All timestamps below are JST.

## The reproducible llama-bench crash

After the 12:15 reboot, a 64 KiB H2D/kernel/D2H test passed with stock NEO and
with explicitly forced Level Zero V2 plus user-local NEO. Every one of 16,384
integer words was checked. The old D-state process was absent and the boot ID
changed to `f0f32621-752d-415c-9056-9f7459e9ac01`.

The current installed llama-bench is build `8345f3339`, SHA-256
`f967bdb8c4d75b5a5a4213a5dfa02d01f83c0e51bd74412a9fc89fc7bf1adde2`.
Its file changed at 01:58, so earlier runs of the October 1 binary are a
different artifact. The following comparison records the current binary.

Using Qwen3 0.6B Code Expert Q4_K_M, all GPU layers, PP512/TG128, batch and
ubatch 512, three repetitions and six CPU threads:

| Persistent SYCL cache | Observation |
| --- | --- |
| `SYCL_CACHE_PERSISTENT=1` | SIGSEGV in `strcmp`, called by `getSortedImages` while constructing the persistent device-code cache key. The second string pointer is `0x8000000000000000`. This occurs during Q4_K dequantization's first JIT lookup in prompt warmup. |
| `SYCL_CACHE_PERSISTENT=0` | Normal exit 0; PP 13,084.21 token/s and TG 312.09 token/s. No new xe kernel events. This is a GPU health check on a 0.6B model, not the Strata speed target. |

The cache-on confirmation at 12:23:43 records the executable hash before and
after the run, matching the cache-off normal run. The cache-on GDB session
ended at 12:23:45; the kernel then recorded a BCS page fault at
`0x0000d556aa2e0000`, `-ENOENT`, and an engine reset. A subsequent 64 KiB
integer test passed at 12:24:32. The earlier 12:21:11 GDB reproduction had the
same CPU stack but no xe events. Thus a cache crash need not wedge the GPU;
the 12:23 event also does not prove why an outstanding copy faulted during
process teardown.

A second BCS fault occurred at 12:41:12, when the first 256-token serve
boundary check was shutting down after ordinary EOS. That check used
`SYCL_CACHE_PERSISTENT=0`; its controller correctly rejected the one-token
answer as insufficient to fill the context. The fault again names
`0x0000d556aa2e0000`, now with ASID 64, followed by `-ENOENT`, memory CAT
error 18 and an engine reset. See `serve-eos-shutdown.xe-kernel.txt` and
[the failed boundary report](../residual-inplace/post-reboot-proof/serve-256-early-eos.json).
This temporal association does not map the ASID to the process or prove a
userspace lifetime bug, but shows that the persistent-cache crash alone
cannot account for all observed GPU faults. The subsequent complete serve
boundary check and legacy short-first comparison both passed. Shutdown and
cross-queue resource lifetimes remain under investigation; existing MTP,
prefill and verifier destructors already include queue waits.
The serve exit path itself calls `queues_wait_and_throw()` inside a catch-all
and then `std::_Exit(0)`, so those C++ destructors do not run there. A wait
exception would be hidden by this path. Exit status zero alone therefore
does not establish that its last global queue wait succeeded. No such
exception was captured in the failed EOS run; this is a diagnostic gap to
test, not an observed exception or a proven cause.

Intel's [issue 21972](https://github.com/intel/llvm/issues/21972) reports the
same cache-sort crash family. Its
[fix, PR 23200](https://github.com/intel/llvm/pull/23200), merged September 18,
handles device images with empty entry tables. The installed runtime's July
build still calls `strcmp` on the unchecked entry. Disabling persistent
caching avoids this path without changing model weights or inference flags.
The device-code disk cache is separate from model weights and KV caches.

The initial cache-off GDB run completed inference normally, but GDB itself
returned 1 because a post-exit `info registers` command had no live inferior.
It is not an inference failure. The following plain run records exit 0 and
parseable benchmark JSON. `first-fault.gdb` now emits a trace only while an
inferior remains alive, avoiding that reporting error.

## Project-trigger audit after the 19:05 human reboot

The investigation prioritizes Strata's trigger and prevention paths. Other
programs failing after a xe fault do not exclude Strata as the first trigger.
No full-model or performance test has run since that priority was established.

The retained first page fault is October 5 at 16:24:54.289299, ASID 81, BCS0,
address `0x0000d556aa2e0000`; see [the kernel observations](earliest-retained-faults.json).
The AOT normal-MTP report's preserved filesystem timestamp is 16:24:54.350903,
61.604 ms later. Its controller writes this report only after sending `QUIT`
and waiting for the engine to exit. All four requests completed and the engine
returned zero. The engine hash and historical `generate.cpp` hash match the
saved AOT build record. See [the association record](earliest-fault-normal-mtp-association.json).
This prioritizes serve shutdown for investigation, but the filesystem timestamp
is not an exact process-exit timestamp or a mapping from ASID 81 to its PID.

An ordinary cache-close path lacked the queue drain required before SYCL USM
reclamation. The actual method with deferred CPU read/write consumers reproduces
two AddressSanitizer use-after-free errors. The fix passes five CPU cases and
rebuilds the engine; see [the cache-close proof](expert-cache-close/README.md).
It is a proven lifetime gap under pending consumers, not yet the demonstrated
cause of the real GPU faults. Serve uses `_Exit`, so this destructor fix by
itself does not cover its shutdown path.

Small GPU checks passed after the human reboot once the recovery probe's missing
UMF library path was fixed. Those checks do not validate full Strata execution
or prove recovery without reboot. Orca's headless-service termination is being
investigated independently. Human recovery no longer stops the GUI; the
[incident evidence](recovery-incident-20261006/) records that correction.

## What the preserved xe dump proves

The user saved the privileged dump and process stack before reboot. The raw
GPU ring/GuC payload remains outside Git; its digest is in
`evidence-manifest.json`. `xe-devcoredump-fields.txt` omits encoded payloads.

At 01:44:42.853499, the dump records a scheduling-disable request with no GuC
response, GuC ID 0, queue flags `0x73`, a reserved BCS8 engine, and five
outstanding G2H messages. The flag definitions in
[xe_exec_queue_types.h](https://github.com/torvalds/linux/blob/v7.0/drivers/gpu/drm/xe/xe_exec_queue_types.h)
decode this as KERNEL, PERMANENT, HIGH_PRIORITY, LOW_LATENCY and MIGRATE.
This is an internal kernel memory-migration queue, rather than an application
decode kernel. Start/current sequence numbers are both 376. Zeroed captured
engine registers do not locate the first stalled instruction.

At 01:51:07 the kernel exhausted recovery attempts, declared the device
wedged, and blocked new IOCTLs/executions. PID 2195423 had already aborted
during a tiny allocation attempt. Its saved stack runs through
`do_exit -> __fput -> drm_release_noglobal -> xe_file_close -> xe_vm_close ->
dma_resv_wait_timeout -> dma_fence_default_wait`. It was blocked while closing
the GPU VM and waiting for a fence, not running inference. Missing entries in
`xpu-smi ps` are insufficient evidence that no process owns VRAM: the installed
[xpu-smi 2.0.1 process filter](https://github.com/intel/xpumanager/blob/v2.0.1/hal/core/process.cpp)
skips processes with no supported active engine flags.

Later allocation errors and NEO `checkResetStatus` aborts were observed after
GPU failure. They do not establish ordinary VRAM exhaustion as the initial
cause. The legacy-adapter experiment also began after earlier failures and
cannot compare adapter reliability from a healthy state.

## A missing xe recovery-order fix

The installed `xe.ko` was revalidated after reboot. Its uncompressed SHA-256
is `f0405334d27deb38f6da791d2ff61a0057e07c9679f5cf557f80dfcf4ebf94ae`.
The disassembly in `installed-xe-timeout-asm.txt` shows the KERNEL flag branch
calling `xe_device_declare_wedged` at address `0x3bf39`, then immediately
jumping from `0x3bf3e` back to `0x3bca1`, which sets job errors and walks the
pending scheduler jobs. This is the ordering removed by the upstream
[d42df9d fix](https://github.com/torvalds/linux/commit/d42df9dce7b3), originally
`a889e9b06bfd`: declare the device wedged only after finishing queue cleanup.

The module therefore still contains this known recovery-path defect. Our
logs do not show the freed-memory general-protection fault from the upstream
report, so this does not prove that exact race occurred here or identify the
first page-fault trigger. It is evidence for updating the recovery path, not
a claim that this one patch fixes every initial hang.

The first fault in the earlier boot was at 16:24:54 on October 5, at the same
BCS fault address. GuC message timeouts occurred by 20:00:36, before the
residual-in-place candidate was first built at 20:38. The candidate cannot
explain those earlier events. This timing does not exonerate every earlier
Strata/runtime code path.

Both the old dump and the new boot load GuC 70.44.1 while the kernel recommends
70.54.0. That mismatch is a candidate factor; a recommendation warning alone
does not prove firmware incompatibility. No kernel, firmware, host security
setting, GPU binding, or service state was changed by this investigation.
The embedding service remains stopped at the user's request.

The local APT indexes offer the currently installed kernel, firmware and
oneAPI compiler versions as candidates. This is an observation of cached APT
metadata, not a fresh package-index update or proof that no newer upstream
release exists. A xe kernel fix and the separate SYCL runtime fix are different
updates.

## Strata verification after reboot

The immutable cache-policy control, SHA-256 `a009a403...`, completed native
weight upload, 257-token layer-major prefill and one generated token at
12:29:34, with the user-local runtime and persistent caching disabled. Every
saved head byte, all 257 FP32 residual rows and the complete dumped persistent
state match the original reference exactly: 993,280, 10,528,776 and
124,852,808 bytes respectively. No xe events occurred during this run.

A preceding test returned a configuration error because `STRATA_DBG_NAN`
disables compact layout and layer-major prefill requires that layout. This
test flag was present in the pending residual checker. The checker now requires
the actual compact-hc layout and scans every saved residual FP32 value plus
every head value directly, without disabling the tested path. The guard
accepts the frozen 257-row reference and rejects NaN, infinity, a wrong row
position and trailing bytes.

All nine short residual runs subsequently completed: eight candidate equality
comparisons and one fresh short-first reference run. All compared head,
residual and persistent-state bytes match, and no stage records a new xe
event. See [the residual proof](../residual-inplace/post-reboot-proof/summary.json).
The independently run frozen legacy executable also matches the fresh
short-first control byte for byte; see
[the legacy cross-check](../residual-inplace/post-reboot-proof/legacy-short-first-equality.json).
The corrected real 256-token normal-MTP serve check fills the final KV cell,
shortens its last verify window to two tokens, refuses both overflowing
requests and handles a valid request afterward. It is a small boundary
integration check, not the full 256K result.
The subsequent full CLI run processes 262,141 prefill tokens and generates
two tokens after 262,142 inputs at exactly 262,144 capacity. It exits zero,
has a complete finite head and records no new xe events. See
[the full CLI report](../residual-inplace/post-reboot-proof/cli-262144.json).
This does not prove the separate normal-MTP final-KV-cell check.

Full normal-MTP serve fails before prefill because its 1.27 GiB layer cache
does not fit the 0.43 GiB available after decode's expert-cache release. At
13:44:31, after that failed request exits, BCS again faults at
`0x0000d556aa2e0000`, ASID 72, followed by `-ENOENT` and a reset. At 13:45:24,
GuC action 0509 times out and scheduling-policy enable returns `-ETIME`.
The tiny integer check with forced V2 and
`UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1` still passes all 16,384 words at
13:46:42, but its interval includes repeated kernel migration-queue
timeouts/resets: GuC ID 0, flags `0x73`, no userspace process. Thus its exit
zero does not establish that the driver's recovery is finished. A new
privileged dump was present and preservation was requested; it later expired
without a saved copy, as recorded below. See
`full-serve-failure-and-health.xe-kernel.txt` and
`health-after-full-serve-v2-no-copy.json`. No rebind, firmware or kernel
change was performed.

## Later GPU controls and dump preservation

The installed module's first-failure guard and delayed dump release are
checked separately in [the preservation audit](devcoredump-preservation-audit.json).
The captured flag prevents another snapshot until release; the actual
module passes 3,600,000 jiffies and the installed kernel uses HZ=1,000.
This agrees with the upstream
[xe dump source](https://raw.githubusercontent.com/torvalds/linux/v7.0/drivers/gpu/drm/xe/xe_devcoredump.c).
Thus waiting for privileged preservation did not require a blanket hold on
GPU checks. The dump's inode and directory remain unchanged across the next
probe. A retained dump describes its captured failure, not necessarily the
current recovery state.

At 14:40, the real ExpertCache/captured-graph probe, SHA-256
`b4cfb86d1718b4b29951e9678e6fcb3ae3831cc5018a1f8e5fe300e07454befd`,
aborts with `UR_RESULT_ERROR_DEVICE_LOST` before completing any result.
The kernel records 188 xe messages. Its first timeout is the internal
migration queue, GuC ID 0, sequence 46,312, not started; the later timeout
names the probe's own queue. The prior tiny health run reached migration
sequence 46,311, so this is consistent with an unresolved recovery path.
The probe's `unique_ptr` cleanup calls a throwing queue wait: its abort stack
therefore hides the original operation's exception. The result cannot
classify 8 MiB VMM support or validate the new MTP weight lease. See
[probe metadata](cache-lease-8m-v2-no-copy.json),
[stack](cache-lease-8m-v2-no-copy.stack.txt) and bounded kernel text.

At 14:46, the existing llama-bench executable with the same recorded hash
runs Qwen3 0.6B, PP32/TG8, all GPU layers, one repetition and six CPU threads.
It uses user-local NEO 26.35, forced V2, disabled copy offload and disabled
persistent caching, matching the probe's runtime policy. It also aborts with
`DEVICE_LOST`, at `ggml_backend_sycl_buffer_clear` during KV-cache
initialization's `memset(...).wait()`. It completes no inference or benchmark
result. The interval contains 7,679 xe messages. See
[control metadata](llama-after-vmm-loss-v2-no-copy.json),
[stack](llama-after-vmm-loss-v2-no-copy.stack.txt) and stderr.
Current execution therefore also fails outside the new Strata lease code;
this does not identify the first fault's trigger or exclude a userspace bug
that initiated the earlier driver failure. It does not compare runtime
reliability from a healthy device state.

The old debugger script returns zero after an inferior stops on SIGABRT.
These records explicitly mark inference/probe completion false; debugger
exit zero is not used as proof of target success. The separate
`first-fault-strict.gdb` returns failure when an inferior remains stopped or
its exit status is unavailable, and propagates a normal nonzero exit.
Its [CPU checks](gdb-strict-cpu-check.json) cover successful exit, nonzero
exit and SIGABRT; they do not run the GPU. Future GPU proofs still require
complete result records and the kernel-event audit.

The kernel reports dump deletion at 14:48:58.506821. The requested
`xe-devcoredump-1346.txt` is absent. The old pre-reboot dump remains preserved,
but the newer retained dump cannot be analyzed now. Both new inferior
processes are absent, and no accessible process owned by the current user
holds an Arc B570 DRM file descriptor at the subsequent observation.
This is a process-ownership check, not a GPU recovery claim; inaccessible
processes are outside its scope. See
[process/dump status](post-probe-process-and-dump-status.json).

Checkpoint/cancellation integration, paired warm performance measurements and
full normal-MTP serve remain separate requirements, as do full checks of the
combined GCC build. Raw per-stage metadata and controllers
are preserved in
`~/.local/state/strata-sycl/gpu-stall-diagnosis/post-reboot-20261006`.
The recorded runner source expects that persistent state directory, including
the preserved compiler environment and health binary; it is an investigation
artifact, not an installed general-purpose benchmark tool.
