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
