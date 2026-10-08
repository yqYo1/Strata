# Owned debugger main-thread state

The failed batch-32 capacity run stopped on the watchdog's `SIGABRT` in
thread 34. The preceding helper saved all backtraces but registers only for
that selected thread. Its main-thread backtrace did not unwind sufficiently
to identify the call. The old register dump therefore cannot establish a
main-thread syscall result.

The helper now preserves the originally selected thread's registers, locates
the main LWP by the owned inferior PID in GDB's thread list, and reads its
registers, 24 stack words and 16 instructions at the current PC. It restores
the original thread selection before returning. These commands do not call
inferior functions, write memory or resume a real crash. Unavailable memory
is recorded as a partial-capture error; the remaining snapshot is retained.
The receipt names each register context and reports which reads succeeded.

Four actual CPU/GDB cases passed on 2026-10-07:

| Case | Observed result |
| --- | --- |
| Previous helper, worker calls `abort()` | Only the selected worker's register dump; missing main-register control reproduced |
| Current helper, worker calls `abort()` | Worker and main register dumps, stack words and instructions; selection restored to worker; first `SIGABRT` remains stopped |
| Current helper, requested interrupt | Main state captured; only requested `SIGINT` resumed; owned child remained live until cleanup |
| Current helper, intentional CPU call to address 1 | First `SIGSEGV` preserved; registers and stack captured; unreadable instruction memory reported without losing the snapshot |

Each case cleaned up both owned child and debugger, without a surviving
process. The fixture uses two CPU threads and no GPU loader, context or
submission. Its intentional invalid call is a crash fixture, not engine
code. These checks establish diagnostic attribution and lifecycle behavior,
not GPU correctness or a model stall's cause.

`owned-main-thread-cpu/` contains the build, four raw GDB logs, snapshots,
terminal receipt and fixture source. `sources.json` maps the old and new
helpers and actual test harness. `manifest.json` records the archived files.
