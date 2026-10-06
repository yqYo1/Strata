# Arc B570 recovery and failure investigation

On the investigation host, run this one command:

```sh
~/.local/bin/strata-gpu-recover
```

No arguments or separate sudo command are needed. The script asks for the sudo
password when necessary, checks the GPU's users, tries function-level reset
(FLR), and checks execution. If FLR fails without an unresolved safety condition,
it tries a bus reset only when the GPU is alone on that secondary bus. It stops
after the first verified recovery. The installed entry point is built from
[sycl/tools/recover-gpu.sh](../sycl/tools/recover-gpu.sh).

If clients remain, the GPU drives a display, another recovery is still blocked
in the kernel, or client inspection is incomplete, it stops rather than trying
another reset. It also stops when interrupted. It never kills an unrelated
process or restarts a service. The embedding service stays stopped as requested.
Client detection uses descriptors, mappings and blocked xe close stacks because
`xpu-smi ps` does not show every owner.

The result is printed on the terminal and a summary is saved at
`~/.local/state/strata-sycl/gpu-recovery/latest.json`. Detailed logs stay under
`/var/log/strata-gpu-recovery`. Exit 0 means the small GPU probe passed without
new xe fault/reset messages. Exit 4 means recovery was refused because a blocking
condition remains. Other failures report the step and its log. A successful
probe does not establish that a full Strata workload is correct.

The helper does not save devcoredumps, flash firmware, change xe's timeout/reset
policy, restart a service, or reboot the host. Each sysfs writer and GPU probe
runs in a separate process with a deadline. A writer still in uninterruptible
sleep after KILL is recorded by PID and start time; subsequent runs refuse to
compete with it. KILL cannot make a kernel D-state wait immediately disappear.

The internal helper is [sycl/tools/recover-xe.sh](../sycl/tools/recover-xe.sh),
installed alongside the entry point as `strata-xe-recover-core`. Its inspection
and developer controls are not needed for normal recovery. The probe is
[sycl/tools/xe-health.cpp](../sycl/tools/xe-health.cpp), installed as
`strata-xe-health`. Rebuilding it with oneAPI only compiles it:

```sh
icpx -fsycl -fp-model=precise sycl/tools/xe-health.cpp -o ~/.local/bin/strata-xe-health
install -m 755 sycl/tools/recover-xe.sh ~/.local/bin/strata-xe-recover-core
install -m 755 sycl/tools/recover-gpu.sh ~/.local/bin/strata-gpu-recover
```

The host's three installed files are already prepared. The probe selects PCI
`0000:05:00.0`, uses an in-order Level Zero queue and checks all 16,384 integer
words after H2D, a kernel and D2H in three rounds. It uses no Strata model,
graphs, virtual memory or mapped polling words. The helper runs it as the
ordinary sudo caller using the installed Level Zero V2 adapter, persistent
caching off and copy offload off. New kernel-journal faults/resets invalidate a
data match.

No software reset is guaranteed to recover every firmware/driver wedge. There
is an [upstream B570 report](https://github.com/intel/compute-runtime/issues/962)
where both rebind and PCI reset failed; that report is not proof of this host's
cause. The recovery flow follows the kernel's
[DRM recovery prerequisites](https://docs.kernel.org/gpu/drm-uapi.html#device-wedging)
and [xe recovery guidance](https://docs.kernel.org/gpu/xe/xe_device.html).

## Firmware audit on 2026-10-06

The actual PCI IDs are `8086:e20c`, subsystem `172f:0102` (SPARKLE). All seven
display connectors were disconnected; `boot_vga=0`. The advertised reset
methods were `flr bus`. Firmware was queried without flashing or changing
system metadata, using fwupd's igsc plugin:

| Part | Installed version | Latest matching stable LVFS release |
| --- | --- | --- |
| FWCODE | 21.1182 | 21.1182 |
| OptionROM Code (VBIOS code) | 23.1066.0.0 | 23.1066.0.0 |
| FW Data | 203.1 | No matching public release found |
| OptionROM Data | 23.1051.0.0 | No matching public release found |

The [recorded firmware audit](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/firmware-audit-20261006.json) contains the device GUID matches.
The live stable LVFS metadata was fetched at 2026-10-06 06:34:42 UTC, last modified
October 5. Its SHA-256 is
`3e84ee0f9a44cd8bf6867d07ff4f9544536ed209a9d43bae3dce1686bcaf4c82`.
The FWCODE and OptionROM releases match the device GUIDs reported for this card.
No newer applicable public stable firmware/VBIOS was found. A generic B570
product name does not justify using another manufacturer's OptionROM Data.
[SPARKLE's download center](https://www.sparkle.com.tw/en/Download) does not list
a separate B570 VBIOS update. The physical board SKU is not established from the
subsystem ID alone. OEM-only updates are not ruled out.

Intel's older [Linux firmware article](https://www.intel.com/content/www/us/en/support/articles/000096950/graphics.html)
says Linux driver packages do not update card firmware; that does not mean the
current fwupd/igsc firmware path is unavailable. This host actually exposes all
four parts to fwupd. The GuC firmware loaded by xe (`70.44.1`, recommended
`70.54.0`) is a separate host firmware file, not the card's VBIOS.

For future read-only checks:

```sh
fwupdmgr get-devices
fwupdmgr get-releases 310f45f1f223064b5c16bf6dff31146755a64480
fwupdmgr get-releases 66ce30cd03cc094ee6b0a2cc78519376f1de4707
```

Do not infer a new firmware release from its CAB filename: the current file is
named `intel-arc-bmg-21.1180.cab` but its release metadata says FWCODE 21.1182.

## Scope of validation

The [validation record](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/recovery-validation-20261006.json) records 14 passing CPU test methods and the installed file digests.
The shell syntax, entry sequencing, recovery guards, reset ordering, method restoration, child
timeouts, and rejection of misleading health results have CPU-only tests:
`python3 sycl/tools/test_recover_xe.py`. The probe builds with oneAPI 2026.1.1.
Read-only inspection ran on this host. No root recovery action has yet been
executed by this investigation, so a successful no-reboot recovery is pending.
The full Strata arithmetic/context/cancellation/graph checks remain separate
from this 64 KiB GPU health check.
