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

If the remaining users are exclusively Xorg processes belonging to this host's
`sddm.service`, the script offers to end the GUI session. Save open GUI work
first: entering the literal `yes` ends those applications. Enter, any other
answer, or no interactive terminal cancels without stopping the GUI. This
path also requires a separate connected boot/display GPU, a disconnected B570,
inspectable owners and no stuck recovery writer. It rechecks the approved
process identities before stopping SDDM.

After that confirmation, a transient system service handles the reset outside
the GUI session and attempts to start the login screen afterward, including
an `ExecStopPost` backup if the worker exits early. This uses the service-manager
parent described in [systemd-run 255](https://raw.githubusercontent.com/systemd/systemd/v255/man/systemd-run.xml).
Starting SDDM is a bounded request, not proof that the login screen appeared.
Applications are not restored. A still-stuck writer or another ongoing recovery
prevents competing display startup/reset operations.

Other remaining clients, an attached B570 display, incomplete inspection and
interruption stop the sequence. The embedding service remains inactive and
disabled as requested; the script does not change it.
Client detection uses descriptors, mappings and blocked xe close stacks because
`xpu-smi ps` does not show every owner.

The result is printed on the terminal and a summary is saved at
`~/.local/state/strata-sycl/gpu-recovery/latest.json`. Detailed logs stay under
`/var/log/strata-gpu-recovery`. Exit 0 means the small GPU probe passed without
new xe fault/reset messages. Exit 3 means the detached GUI recovery was handed
off; it does not mean the GPU recovered. Its final receipt must have
`healthy: true`, after another GPU probe covering the display-start request and
fresh kernel messages. A queued or interrupted receipt has `healthy: false`.
Exit 4 means recovery was declined or refused because a blocking condition
remains. Other failures report the step and its log. A successful
probe does not establish that a full Strata workload is correct.

The helper does not save devcoredumps, flash firmware, change xe's timeout/reset
policy, or reboot the host. The only service it may stop/start is SDDM after the
interactive GUI-exit confirmation. Each sysfs writer and GPU probe
runs in a separate process with a deadline. A writer still in uninterruptible
sleep after KILL is recorded by PID and start time; subsequent runs refuse to
compete with it. KILL cannot make a kernel D-state wait immediately disappear.

The internal helper is [sycl/tools/recover-xe.sh](../sycl/tools/recover-xe.sh),
installed alongside the entry point as `strata-xe-recover-core`. Its inspection
and developer controls are not needed for normal recovery. The GUI handoff is
[sycl/tools/recover-xe-display.sh](../sycl/tools/recover-xe-display.sh), installed
as `strata-xe-display-recover`; its worker controls are internal. The probe is
[sycl/tools/xe-health.cpp](../sycl/tools/xe-health.cpp), installed as
`strata-xe-health`. Rebuilding it with oneAPI only compiles it:

```sh
icpx -fsycl -fp-model=precise sycl/tools/xe-health.cpp -o ~/.local/bin/strata-xe-health
install -m 755 sycl/tools/recover-xe.sh ~/.local/bin/strata-xe-recover-core
install -m 755 sycl/tools/recover-xe-display.sh ~/.local/bin/strata-xe-display-recover
install -m 755 sycl/tools/recover-gpu.sh ~/.local/bin/strata-gpu-recover
```

The host's four installed files are already prepared. The probe selects PCI
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

The [original validation record](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/recovery-validation-20261006.json)
records the earlier 14-test version. The
[GUI handoff validation](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/display-recovery-20261006/record.json)
records 39 passing CPU test methods, shell syntax and the current installed
digests. These cover reset ordering, method restoration, child timeouts,
misleading health results, no-argument shell sequencing, explicit consent,
changed/hidden owners, display ownership, borrowed locks, stranded writers,
service-stop failures and interrupted-worker receipts:

```sh
python3 sycl/tools/test_recover_xe.py
python3 sycl/tools/test_recover_xe_display.py
```

The service/reset backends are harmless fakes. Separately, a real temporary user
service proved that its launcher exits before the manager-owned worker finishes
and its `ExecStopPost` runs. The confirmation itself is also tested with a real
private controlling PTY: its prompt appears before input, only `yes` proceeds,
and no/Enter or a detached process without a terminal cancels. This caught and
fixed an earlier `r+` open that requires seeking and fails on real terminals.
Those tests used no root service, GPU or GUI operation;
it does not validate the root SDDM recovery transaction. The probe builds with
oneAPI 2026.1.1.
Read-only inspection ran on this host. A root recovery attempt was refused
before reset because Xorg held the B570 open in an active local X11 session.
The new entry obtains GUI-exit consent on the terminal for that case. No display
shutdown or device reset was performed during these tests, so successful
no-reboot hardware recovery remains pending.
The full Strata arithmetic/context/cancellation/graph checks remain separate
from this 64 KiB GPU health check.

The verifier watchdog registry also had a CPU lifetime race: a callback could
load an atomic raw pointer, then use the verifier after another thread removed
and freed it. Registration now occurs after initialization, and a mutex covers
each diagnostic/release callback and removal before mapped flags are freed.
This follows the C++ rules for
[object destruction](https://eel.is/c++draft/class.cdtor) and
[thread synchronization](https://eel.is/c++draft/intro.races); an atomic pointer
does not provide ownership of the pointed-to object.
The [CPU regression record](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/verifier-registry/record.json)
extracts the actual registration/callback/removal code, substituting CPU payloads
for the verifier's mapped flags and queues. The pinned old control reproduces
ASan heap-use-after-free for both diagnostic and release callbacks. Five
candidate ASan/UBSan cases pass those races, failed initialization/retry,
16-stage capacity/gap reuse and 2,048 ownership cycles across four owner threads
with concurrent diagnostics/releases. Run it without a GPU:

```sh
python3 sycl/tools/test_verifier_registry_host.py --output /tmp/strata-verifier-registry-check
```

Both reference and CPU task-factor-9 engines compile/link with the registry
change. Their measured production CPU archives remain byte-identical to the
earlier exact-output/timing evidence. The
[link environment audit](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/verifier-registry/link-environment-audit.json)
preserves an initial failed identity check: configuring pinned ggml without
`ONEAPI_ROOT` adds `-lm` and changes the executable. Explicit oneAPI/MKL paths
make both builds use the prior recipe; restoring factor 0 reproduces the initial
reference hash. Neither new engine ran on the GPU. This regression proves a
possible CPU use-after-free, not the trigger of the original GPU fault or a
speed improvement. Other source-review and full-context gates remain pending.
