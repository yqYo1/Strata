# Optional FP16 cache model checks on 2026-10-07

The private cache is rebuilt against the updated upstream executable on Arc
B570 10 GiB / Ryzen 5600X / 128 GiB RAM, kernel 7.0.0-38-generic, NEO
26.31.39395.14 and oneAPI 2026.1.1. Production source and binary are unchanged.
All first model checks retain flushed Level Zero/UR logs, parameter validation,
original per-phase waits/progress, normal MTP and owned GDB/PTY cleanup.

With cache disabled, four context-128 requests match every ID, logprob and
all 248,320 first-head floats. Six release/restore pairs complete; the process
exits normally without a new xe fault. Active cache128 allocates 1,258,291,200
bytes in the context-4096/int8-KV/2048-input test, but the first request stops
at layer24, token offset256, after two dequant launches enter completion wait.
The 60-second serve watchdog aborts it. A xe CCS engine reset appears near
cleanup, after the watchdog; this does not show that a reset began the stall.

A combined candidate retaining the same cache and changing only the two IQ
dequant launches to regular launches stops at layer14/offset256. Disabling
the cache in that exact executable also stops, at layer3/offset1280. Finally,
the unchanged ceff updated-upstream control stops at layer17/offset256 during
the same fully logged recheck. These last three intervals contain no new xe
fault. No failed run completes an output request or restores its released
decode cache. Both owners are gone after every cleanup. Four subsequent
logged H2D/kernel/D2H probes pass three rounds of 16,384 exact words each,
without invoking a reset, rebind or reboot. Small probes do not establish
healthy long model execution. See [assessment](assessment.json) and the
individual terminal records and bounded stderr tails.

The saved main-thread frame chain from the regular-launch cache128 run
contains a return address in NEO CommandStreamReceiver::baseWaitFunction,
resolved with the exact matching build-ID debug ELF. The VDSO instruction
and registers alone do not identify GPU completion tag values or why the
wait never completed. Optional cache capacity is not a sufficient explanation:
the zero-cache candidate and unchanged control also fail. No general hang
prevention or cache parity/speed result follows from these failures.

The [actual IQ-object query](cooperative-limit-query/query-record.json)
submits no GPU kernels. Both original dequant kernels report 1,152 maximum
cooperative work-groups for local size32; the original stalled GU/down
launches use 12,800/6,400 groups with UR_KERNEL_LAUNCH_FLAG_COOPERATIVE.
The installed query maps to zeKernelSuggestMaxCooperativeGroupCount. Its
API differs from the current online root-group-specific query; the first
failed compile and corrected build are preserved. The
[Level Zero programming guide](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html#cooperative-kernels)
requires cooperative group counts to respect the queried maximum. These
dequant bodies use no root-group synchronization, so the setting needs
correction, although its removal alone did not prevent the observed waits.
The [source audit](root-sync-property-source-audit.json) finds 317 property
occurrences in 47 sycl/src files and no textual root-group helper call.
This is not a complete call-graph or resource audit, and does not justify
blanket removal from persistent or atomic synchronization kernels.

The cache owner, bounds, producer/consumer ordering and original arithmetic
remain reviewable in [the header](sources-used/expanded_fp16_layer_cache.hpp)
and [the prefill diff](sources-used/prefill-cache.patch). Clean 0/128/256
timings are not run: the prepared controller's acceptance prerequisite failed.
Both candidates remain private and unadopted. Repeated restoration and all
262144-cell CLI/serve, clipped-tail, refusal and later-valid gates remain open.

Earlier upstream/dequant receipts incorrectly describe this 2048-input test
as two chunks. The configured maximum is1024, but the initial chunk is256:
actual prefill sizes are256,1024,767 (2047 tokens), with the final input in
decode. The [correction](prefill-2048-chunk-label-correction.json) preserves
all144 successful prior trace entries: 48 layers times three chunks, offsets
0/256/1280. Timings and test conditions are unchanged; prior frozen manifests
are retained. Binaries, whole heads and large logs remain private with hashes;
the manifest covers every archived file except itself.
