# Owned stall stack probe v1 — source proposal, NOT_RUN

All C builds, Python syntax checks, host tests, GDB attaches and model/GPU tests
are **NOT_RUN** in this proposal. Main owns review, builds, execution and closure.
Only files inside this new proposal directory were written. No product patch,
native binary replacement, service action, process signal, cleanup or Git action
is part of this proposal.

The authoritative historical event supplied by Main supersedes research-index's
older live-arm text. Formal Direct0's third nominal32K request had 32089 input
tokens and stalled with watchdog480s at `reading prompt layer27 chunk0` after the
first two PP samples197.7/198.0. Original native3276405 exited SIGABRT(-6) at
2026-10-15T02:18:38.205255Z. Its pool reported18/18done, five workers parked and
no active verify wait. The server then automatically restarted the same92acf71e
binary as3478501 and completed33K prefill plus940decode. Main stopped the invalid
suite through its verified owned-job pidfd, closed ownership, and found no fresh
kernel fault or dump with the boot unchanged. These observations establish no
GPU-hang cause. Direct0 remains invalid; Direct1 is NOT_RUN; the pair cannot
support performance adoption. A later clean native diagnostic is a new run and
cannot repair the historical result. The diagnostic must have no HTTP server,
wrapper, automatic restart or automatic retry.

Historical references are `../prompt-direct-bench-stall-stop-v1.json`,
`../prompt-direct-bench-full-0-v1.json`, and external root
`/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-primary-prompt-direct-bench-full-0-root-v1`.
Their receipt hashes are in `source-manifest.json`; its event summary records
Main's supplied account separately from this agent's source-only work.

The permission design uses a diagnostic-only `LD_PRELOAD` library. Its constructor
calls `prctl(PR_SET_PTRACER, getppid(), 0, 0, 0)` and checks that the parent did not
change. On error it emits one bounded stderr line and exits125 before native
`main`; on success it emits one bounded stderr receipt and leaves stdout alone.
It never broadens permission to ANY or changes dumpability, sysctl, privileges
or the process environment. This grants a relationship only for this live native
process. The existing native executable is read and executed without modifying
it. Dependent library constructors may run before the preload constructor;
this is not a claim to run before every dynamic-library constructor.

With Yama scope1, a nominated debugger and its descendants may attach, subject
to the other ptrace restrictions. Naming Main permits a GDB child of Main to
attach its native sibling. This is directly documented by the
[Linux Yama documentation](https://www.kernel.org/doc/html/latest/admin-guide/LSM/Yama.html).
The positive host gate proves the actual installed kernel/GDB behavior rather
than assuming the prctl success alone proves attachability.

Main should proceed in this order:

1. Keep the shared `owned-v0141-measurement.lock` exclusively held with
   `LOCK_EX|LOCK_NB`. Verify no other owned test/build/GPU job is active. The lock
   path defaults to the same post-reboot root used by existing Main controllers.
   Compile only the two new C files into a new private external host-artifact
   root, through the pinned existing Owner. Do not rebuild, relink or overwrite
   the native92acf71e executable. Suggested argument vectors, expanded by Main
   to absolute paths (not shell interpolation), are:

   ```text
   /usr/bin/cc -std=c11 -O2 -Wall -Wextra -Werror -fPIC -shared ptracer_parent.c -o libowned_ptracer_parent.so
   /usr/bin/cc -std=c11 -O0 -g -Wall -Wextra -Werror -pthread host_target.c -o host_target
   ```

   Record the compiler identity, source hashes, explicit compile environment,
   exact argv and individual Owner receipts. Main must verify compiler
   availability. Build status here remains NOT_RUN.

2. Close/reap those host build Owners before invoking the preflight, because the
   preflight itself acquires the same nonblocking measurement lock. It must be
   the direct controller of both target and GDB. Run it once in a fresh external
   output directory with explicit `--target`, `--shim`, `--output` and the shared
   `--lock`; use `python -B` to avoid source-tree bytecode. Main may supervise the
   controller with its existing Owner, but must not launch the target separately
   or supply an existing target PID. No GPU environment is passed to the host
   target, and no `LD_PRELOAD` is passed to GDB. This gate requires:
   constructor stderr receipt, exact direct parent READY handshake, successful
   bounded GDB attach/all-thread backtrace/detach and normal GDB exit, zero
   `TracerPid` in every remaining target task, a post-detach PONG with an advanced
   worker tick, unchanged boot, and original native fixture exit7 with clean
   Owner session closure. `resumed_proven` is set only after PONG, not inferred
   merely from `TracerPid=0`. Any failure blocks model execution; do not retry
   automatically or relax scope1. Keep the expected exit7 as7, never rewrite it
   to0. It is the fixture's designed original exit, not a model exit.

3. Main reviews this source and implements its own exact token-file sequence
   and native protocol using `main_interaction_adapter.with_probe`. This helper
   accepts Main's readiness parser, request loop and post-stall shutdown parser;
   it supplies no model commands. Main's readiness parser must require the exact
   constructor grant line with native PID and controller PID before permitting
   the first model request. Pin the preflight receipt, compiled shim hash,
   existing Owner SHA256 and original binary SHA256
   `92acf71e6b4fd914f68c8cefcae7e03f170adf62b5e3b94cf31af747344183f7`.
   Preserve the actual original `--serve` native protocol flag, model paths and
   engine args from the historical receipt. `--serve` here is direct stdin/stdout
   protocol; it does not authorize starting the Python HTTP server.

4. Hold the same lock for the full model job. Set the existing reviewed safe
   environment, parameter validation, warning/error logging and `STRATA_TRACE=1`
   in the new native child's explicit exec environment, plus this one shim's
   absolute `LD_PRELOAD` path. Do not mutate `os.environ` or a runtime-initialized
   process. Keep all ordinary mode values, especially Direct0, fixed throughout
   this reproduction; Direct1 is not started by this diagnostic. Do not add a
   full UR trace collector. Main supplies the token files, hashes, lengths and
   request order from the failed case. Reproduce necessary earlier requests in
   the same native process if state is part of the question. This is diagnostic
   work and has no usable clean speed samples.

5. Instantiate two instances of the exact pinned Owner module: native output
   `job/native-owned` and GDB output `job/gdb-owned`. Only the native Owner
   starts the existing executable; only the debugger Owner starts `/usr/bin/gdb`.
   Both instances live in the same single-threaded Main controller. The GDB
   launch is synchronous within `Interactive.poll`, with a separate SID. The
   native Owner's original poll continues enforcing native budgets during
   capture. No additional test/build/service/GPU workload runs concurrently.

6. Suggested whole model job caps are wall2400s, RSS88GiB, text64MiB and job256MiB.
   Reserve2GiB RSS and256KiB text for GDB: use native RSS86GiB and native
   `text_cap=(64<<20)-(256<<10)` so both owned sessions stay within those totals;
   use `cpu=2400*len(os.sched_getaffinity(0))` for the native and `file_cap=64<<20`.
   The debugger has wall30s, CPU30s, RSS2GiB, combined output256KiB and per-file
  256KiB. Existing Owner cleanup can take up to two3s signal observation windows
   plus a0.1s reap attempt; tracer verification adds at most3s. Therefore reserve
   about40s inside the whole native deadline. Owner's256live/200000cumulative
   identity and256fd budgets remain unchanged. No limit is removed.

7. Arm the adapter after READY, before the first selected request. Its silence
   threshold is configurable; the proposal defaults to240s. A150s threshold is
   possible, but32089/197.7 alone is about162s, so150s without stdout progress
   could snapshot an ordinary prefill before it becomes abnormal. Neither a
   stdout-silence threshold nor an attached stack proves a stall's cause.
   Main must pin its selected threshold and whether stderr trace progress is
   separately observed. The timer tracks actual consumed stdout bytes, including
   partial lines, and still catches a blocked `Interactive.send`. Stderr progress
   does not reset this particular timer. A shutdown or long normal response may
   also be silent; Main should place the probe around the intended request phase.

8. On the first silence event the adapter closes its request gate, verifies the
   original boot/PID/start ticks/parent/SID/PGID and every task's `TracerPid`, then
   attaches exactly once. The native leader remains unreaped and pidfd pinned
   throughout numeric GDB attach; no recycled/adopted PID is accepted. GDB reads
   `info threads` and `thread apply all bt 12`, with frame arguments suppressed,
   pagination/confirmation/auto-load/debuginfod disabled and inferior function
   calls disabled. It never runs the target, inserts a breakpoint, calls an
   inferior function or requests memory/register writes. It detaches and exits.
   GDB documents its [bounded backtraces](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Backtrace.html),
   [attach/detach](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Attach.html)
   and [auto-loading controls](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Auto_002dloading.html).

9. After GDB Owner closure, verify every surviving native task has `TracerPid=0`
   and the pinned native is not in stopped stateT/t. If GDB timed out/exceeded a
   cap, its own Owner performs bounded pidfd cleanup and the capture is
   incomplete even if the tracer clears. If the tracer cannot be cleared or
   native identity cannot be proved, record unresolved closure and propagate to
   the native Owner; do not report a safe detach or resume. Do not kill a numeric
   PID/group or adopt the tracer. The original native Owner retains authority.

10. After a successful stall snapshot, `StallCaptured` transfers control to
    Main's bounded post-stall handler. Choose one exact native QUIT/STOP payload
    before launch, or no sends while observing its original watchdog exit.
    The adapter rejects every other send, so no further request is submitted.
    `finish_stalled` must consume the actual protocol and finish native exit;
    a handler failure returns to existing native Owner cleanup. Do not swallow
    its exceptions and continue the suite. Main records the new native exit
    verbatim, including watchdog SIGABRT if it occurs. Any diagnostic-requested
    termination is labeled as such and cannot replace the historical -6 result.
    Do not use a stack to claim CPU service time, GPU idleness or GPU hang.

11. Close the native Owner through its final anchored scan, reap and pidfd
    closure. Then use Main's existing fresh-journal/dump/boot checks; do not run
    another GPU probe merely to infer closure. Record kernel cursor boundaries,
    exact faults/filter/counts and all ownership outcomes. Keep one representative
    stack for the unresolved layer27/chunk0 mechanism, source/command/env/input
    pins and compact receipts. No raw tensor/session capture is required. A256KiB
    debugger cap is a hard completeness condition, not permission to truncate
    silently. Combined native/debugger text stays within64MiB and the entire job
    within256MiB.
    Main reviews retention at this boundary, preserves needed progress and error
    evidence with hashes and writes any later deletion manifest under the shared
    lock. This proposal deletes nothing.

Open limitations: neither the preload nor installed GDB behavior has been built
or checked here. The installed binary may be stripped, optimized frames can be
missing, and auto-load-off may limit thread library assistance. Host PONG must
prove the real target resumes; source review does not. The adapter does not
prove model protocol correctness or validate token files, and deliberately leaves
that work to Main. A stack is a host observation at one interrupted instant.
The next review point is Main's host-only receipt; the model gate stays NOT_RUN
until that passes and Main has reviewed the exact new controller.
