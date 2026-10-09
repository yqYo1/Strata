# Native gprofng FIFO protocol smoke v2

Status: **SOURCE ONLY / UNTESTED. Runtime gate OPEN.** No build, syntax check,
help/version query, target, profiler, model or GPU execution occurred for this
implementation. Root owns the serial validation and all cleanup. This report
makes no measured performance or profiler-usability claim.

Base revision: `391502facfcf72d3371fc487fdb47e8840bd5b56`
(`docs/sycl-storage-retention-2026-10-09`). The independent implementation branch
is `diag/gprofng-native-smoke-20261010`. The source scope is this directory only;
no prior receipt or research registry was modified.

`native_target.c` runs an unsigned-integer busy function for approximately four
wall-clock seconds. It reads RUN/QUIT from a private request FIFO and writes
READY/DONE/BYE to the reply FIFO. Protocol messages are separate from ordinary
stdout/stderr. It installs no signal handler and creates no timer. The collector
owns SIGPROF and SIGUSR2. `clock_gettime` checks elapsed wall time only.

`check_native_protocol_v2.py` supervises compilation with `-g -O2`, collection
starting paused with `-y USR2`, and text rendering in separate serial stages.
It holds B's `owned-v0141-measurement.lock` throughout those stages and cleanup.
Root passes B with `--base`; source lookup is relative to the controller, so a
copy must keep the C source next to it. A new run name is mandatory after an
existing run: the controller never overwrites results.

Root execution recipe, after independently reviewing this source:

```sh
python3 /absolute/source-directory/check_native_protocol_v2.py \
  --base /home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007 \
  --run-name gprofng-native-protocol-smoke-v2-r1
```

Do not use Python `-O`. The controller explicitly rejects disabled assertions.
No recipe in this report was executed by the implementation agent.

The work deadline is 57 seconds from supervisor initialization, reserving three
seconds for exact-owned termination and reaping within the nominal 60-second
job. It polls combined text, binary and experiment size against 64 MiB. A
budget-triggered stop is rejected; polling can observe a burst after crossing
the limit, so the limit is a supervised stop threshold, not a filesystem quota.
The controller never discards required history to keep running.

READY must identify a discovered owned child with matching `/proc` start ticks
and the exact compiled executable. Before each SIGUSR2 toggle the controller
checks the collector-installed caught-signal bit. Signals use pidfds, which
avoid a PID reuse race between checking identity and sending a signal. Normal
completion must have no forced cleanup and no surviving live owned processes.
Visible competing Strata/build/debug/profiler processes cause rejection without
signaling them. The shared lock remains the coordination authority; it cannot
prevent unrelated processes that ignore that lock from starting.

The v1 race is corrected by draining replies before inspecting process exit,
then draining once more after observing exit. BYE already queued when the
collector exits is accepted. An absent or incomplete final reply still fails.

Profile acceptance requires all of these independently:

- Normal compile, collector and report exits, plus READY/DONE/BYE.
- No `cerror` in the retained collector XML.
- A `profile` schema with a `ptimer` attribute value, nonempty `data.frameinfo`
  and nonempty `data.profile`.
- A parsed positive exclusive user CPU-seconds value for `strata_native_busy`
  in `gprofng display text -metrics e.user -functions` output.
- No forced cleanup, controller termination signal, or surviving owned process.

Total, output length and experiment stub files are never sufficient evidence.
An unsupported schema/metric/output format fails closed. The ptimer stream and
text parser assumptions require root's runtime verification; the implementation
agent did not query installed help to infer alternate commands.

The receipt preserves exact commands, source/controller/compiler/collector and
binary hashes, effective environment, boot ID, uname/CPU description, owned
PID/start ticks, individual stage exits, protocol, collector errors, schema and
payload sizes, attribution rows, and cleanup. The private output directory
retains compiler/collector/report stdout and stderr plus the experiment for
root's review. Available errno/debug output is retained as emitted; no collector
debug switch or privileged probe is invented, and no errno is inferred from the
generic timer error. Root should supplement package/driver versions and any
required kernel-journal boundaries in the compact terminal report. This target
performs no GPU work.

Prior evidence: v1 failed receipt SHA-256
`9acfc1557407e7085c65b6d271e7095a0b9e82d43cf84c4daa54e9087599621e`.
Its collector exit 0 and READY/DONE were independent of both the BYE-drain bug
and cerror 9 (`itimer could not be set`), with empty frame data. The ordinary
libc timer matrix passed all five controls; this does not prove the collector's
internal resolved timer functions work. Research references are rounds 16, 18
and 19 under root's `research-20261009`, with registry v21 unchanged. Round 19
cites GNU binutils 2.42 `gprofng/libcollector/profile.c` and `dispatcher.c` for
the ptimer initialization and generic timer-error boundary. The already-used
collect/display command family is retained with an explicit `e.user` metric.

Retention owner: root's native smoke v2. Raw output consumer: ptimer usability
and unresolved collector initialization/errno diagnosis. Budget: combined
64 MiB. Next review: root closes and reviews the terminal receipt, extracts
compact evidence, and retires surplus captures under the shared lock with a
path/hash/reason manifest. This implementation performs no artifact deletion,
compression, service action, global preload change, sysctl change, ptrace or perf
configuration change. Rejection remains rejection during later retention work.
