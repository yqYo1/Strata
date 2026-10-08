# Full-capacity watchdog abort and separate32K definition check

On the ArcB57010GiB / Ryzen5600X /128GiB host, r5's second fresh full
read stops in the host watchdog. Its report names layer33 of the8192-token
chunk starting at196608 and60s without progress. GDB thread82 calls abort
from main's watchdog lambda. The main thread waits in Stager::wait for a
CPU host buffer; three staging workers query DMA events that remain
SUBMITTED/NOT_READY. Their latest2,176,000-byte copies and the last queue
appends return SUCCESS. This does not prove GPU completion or identify which
counter value or dependency stopped advancing.

[The terminal receipt](terminal-r5/record.json),
[all stopped stacks](terminal-r5/debugger/failure.mi.txt), selected readable
stacks and bounded raw log excerpts are preserved. Excerpts give exact byte
offsets and hashes within the54,921,377,495-byte private stderr file; no
full-file hash is claimed here. No new kernel fault is recorded. Owned
processes are absent after cleanup, and subsequent read-only device/runtime
health passes. No reset or recovery command is run. The pending-event cause
remains unresolved, without attribution to either the indexer repair or
the separate DPCT definition inconsistency.

The first full read passes head/used-state/output parity, reaches physical
cell262143 and saves all13 KV images through262144 cells without exclusions.
All12 main spare rows equal idx_dead. The complete sequence fails:
repeated full, actual disk restoration, restored clipped tail, capacity
refusals and later32K are not reached. Earlier partial snapshots retain
their original capture scope.

The separate uniform-header executable begins four fresh32768-token
reads alternating two fixtures, plus actual32K SAVE/RESTORE continuation.
Its first control's complete head, used state,64 IDs/logprobs and visible
MTP counts match the accepted reference. Its SAVE and live continuation
pass. [The first-control snapshot](uniform32k-first-completed-snapshot.json)
covers only this completed portion; the complete sequence remains pending
at capture. This candidate has no polling change or demonstrated stall fix.

The raw r6 binary_sha256 metadata was assigned before executable selection
and still names its7f054 base. It is preserved as recorded. Launch argv,
preflight hash assertion and independently checked live /proc/PID/exe prove
the executable is dd5efb9. The [identity correction](uniform32k-actual-executable-identity-v1.json)
retains PID, start ticks and boot identity; the derived snapshot uses that
actual hash. The unexecuted v1 controller's missing receipt-key guard is
also retained; v2 validates the real fields and every copied archive member.

The separate sequence subsequently completes: all four fresh32768-token
reads match their head/used-state/output/logprob/MTP references, and the
actual disk-restored continuation matches the live continuation. The
engine exits0 normally with no forced cleanup, survivors or new kernel
fault. [The unmodified terminal receipt](terminal-uniform32k-r6/record.json)
and [derived proof with the actual executable identity](uniform32k-completed-derived-proof.json)
retain this result. The initial snapshot and summary remain historical
captures. No physical256K or stall-fix claim follows from this32K pass.

All speed comparisons require at least32768 input tokens, separating the
first request from repeated full reads. Logging/captures and SAVE/RESTORE
durations are excluded from speed. No candidate is adopted before the
complete physical256K gate. Production, accepted binaries and main remain
unchanged; PP1000/TG70 remains unachieved.

The [subsequent GPU timestamp rejection and polling gate](../device-profile-rejection-and-poll32k-v1/README.md) retain the actual earlier VTune startup failure on this Ryzen processor and a failed logged 32768-token unitrace GPU timestamp run. The application watchdog stopped at layer 26 of chunk 16384; no new kernel fault was recorded and post-exit GPU/runtime health passed. Its empty/incomplete trace and durations are excluded from performance conclusions. The DD5-based polling candidate subsequently passed four fresh 32768-token reads, all head/used-state/output/logprob/MTP comparisons, actual disk continuation and normal exit. It is still private: quiet matched inputs of at least 32768 tokens and the complete physical 256K gate remain required. First and repeated full reads must be reported separately; the underlying pending-counter cause is unresolved.
