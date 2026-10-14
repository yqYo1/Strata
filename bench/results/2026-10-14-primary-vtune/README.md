# Primary profiler capability checks

2026-10-14, VTune 2026.4 build632893, Ryzen 5 5600X and Arc B570.
These are bounded collector checks, not model speed samples.

XPU smoke v1 stopped before collection: generic CLI help omits the hidden
`target-gpu` option. The installed definition was verified for v2 and the
observed GPU BDF was supplied explicitly. V2 was rejected before the runtime
child launched: VTune could not recognize this CPU/current microarchitecture.
Both failed outcomes are preserved byte-identically, with normal owner
closure and no fresh GPU faults. There is no XPU API/task timeline and no
actual model profiling result from these attempts.

The separate hotspots software-sampling check in `check_software_smoke_v3.py`
was also rejected before the runtime child launched: VTune requires a ptrace
scope change from the current value1. Its original failed receipt is retained.
No sysctl, driver, service or GPU reset setting was changed. There are no CPU
samples from these checks, and none can substitute for a GPU task/API timeline.

Intel's [PTI/unitrace documentation](https://github.com/intel/pti-gpu/blob/master/tools/unitrace/README.md)
describes Level Zero host timing, device timing and correlated Chrome API/kernel
traces. This is the next alternative for GPU attribution if the XPU collector
cannot run on this host. It requires its own small correctness/capability check
before a bounded model profile. Do not infer support from installed flags or
a created output directory.

The separate unitrace2.4.0 build pins PTI source
`6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033`, compute-runtime headers
`050536ff3b6f830199ec99cff8ffcac3d804c40a`, and Level Zero headers v1.32.0.
The out-of-source version helper printed an empty commit hash; use the explicit
source pins and binary hashes in the canonical build receipt instead. Main's
already-qualified runtime child exited0 under collection with no new GPU fault.

[Actual extracted evidence](unitrace-runtime-evidence.json) verifies the native
PID, B570 device lane,195 device operations, all64 ordered H2D/kernel/D2H
handoffs and195 correlated host-to-device flows. There are128 copies,66 kernels
and1 barrier: the zero-byte Runtime.copy becomes a device task and the bounded
wait is another kernel. The collector's initial expected65-kernel message was
incorrect. Flow_H2D/D2H categories describe causal flow, not payload direction.

**External wall-clock correlation remains unqualified.** Trace UTC-labelled
timestamps fall outside the real child UTC window. Main's earlier validator
rejected that assertion; the original successful collection status is retained
without rewriting it into full capability. `close_unitrace_evidence.py` keeps
internal-order checks separate from the failed external clock gate. There is
no model profile and no attribution of the current model bottleneck from this
trace. One500709-byte raw timeline remains for the recorded clock-source
question, with a1MiB budget and review before any model-marker attribution.

The three temporary profiler/header worktrees and local branches were removed
after their clean source pins and generated headers were recorded. No remote
profiler branch was created. The default mirrors are clean. The built provider
has a4GiB budget and this named clock-analysis consumer; it is reviewed when
the clock question closes. The [retirement record](unitrace-reference-worktree-retirement.json)
preserves paths, revisions and removed header symlink targets. Native engine
source and binaries were not changed by collection or retirement.

The [pre-Xe regression audit](../2026-10-14-performance-regression-audit/README.ja.md)
takes priority over another model collection. Restore/re-evaluate the prior
qualified own-fork source/configuration; do not optimize the slower stock
configuration as if it preserved the old best implementation.
