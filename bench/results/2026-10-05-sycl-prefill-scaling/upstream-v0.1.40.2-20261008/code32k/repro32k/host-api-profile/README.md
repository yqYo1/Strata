# Host API profile on32K input

The unchanged private e82fc5 scheduling executable runs on Arc B57010GiB,
Ryzen5 5600X,128GiB RAM, kernel7.0.0-38, NEO26.31.39395.14 and
oneAPI2026.1.1. Each fresh process reads32,768 identical code-review tokens,
context33024,8192-token chunks, int8 KV, normal MTP4 and64 greedy outputs.
Default implicit counter conversion and no MKL CNR match the
[three completed controls](../registered-copy-default-counter-conversion/README.md).

The locally built [Intel PTI unitrace source](https://github.com/intel/pti-gpu/tree/6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033)
is clean at the pinned commit. A CPU fixture compiles the actual collector
option derivation and checks host timing plus Chrome call logging: API tracing
is enabled; kernel tracing, device timing and metrics remain disabled.
Option-free unitrace would also enable device/kernel tracing and is not used.
Source hashes and generated callback guards are retained in options-audit.
The chosen command uses `-h --chrome-call-logging --output-dir-path ... -o ...`;
the wrapper execs the engine under the owned debugger without preloading GDB
or its helper. The actual target executable hash/PID identity and environment
are captured. This host mode does not install timestamp event-pool or kernel
append callbacks. No GPU event or hardware-counter trace is collected.

First logged/validated32K use, a quiet host profile, and a fresh unprofiled
control all match all66state parts,248,320first-head float bytes,64IDs and all
logprobs. All exit normally, complete owned cleanup and record no new xe fault.
Both quiet profile and control retain state/head capture; they are not clean
performance measurements. Their single-pair prompt durations differ by
about0.18%, decode by about2.93%. This does not establish negligible overhead
for other workloads.

The quiet trace has4,824,682 native host calls and zero GPU events. In the
approximately84.59-second externally observed request envelope, it contains
2,231,722 zeEventQueryStatus calls across three worker threads and503,650
zeCommandListAppendLaunchKernelWithArguments calls. The interval union of
native synchronization calls is about27.53s; status queries about38.78s;
append calls about10.86s. These overlap one another and GPU work. Their
inclusive concurrent sums must not be added or treated as CPU utilization,
GPU busy time or PCIe transfer duration. AppendMemoryCopy host duration
does not measure DMA completion or bytes transferred.

The pinned timer derives its epoch offset from BOOTTIME but stamps API calls
with MONOTONIC_RAW. Analysis adds the measured boot/raw difference (about1.49s)
before aligning to observed protocol times. Calibration was taken after the
job, so phase boundaries are approximate; +/-10ms sensitivity is recorded.
Native durations are unchanged. No raw trace is rewritten. See the complete
[analysis](analysis/host-profile32k-v01402-analysis.json) for counts, per-thread
intervals and limits. All status polling and almost all native kernel appends
occur before the observed last prefill progress message.

These measurements motivate separate launch batching and polling experiments;
they do not identify a dominant transfer/CPU/GPU bottleneck, prove general
hang prevention, meet PP1000/TG70, or validate full262,144-cell serving.
The executable remains private and unadopted. Public files retain exact argv,
environment, controllers, protocol, journal, state/head hashes and profiler
build receipts. Large API/state/head/Chrome payloads stay private with byte
counts and SHA256 hashes. No reset, rebind, reboot, package, service or global
setting is changed.
