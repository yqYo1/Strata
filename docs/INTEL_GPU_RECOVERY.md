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

The script does not stop SDDM, Xorg, Orca or Codex. Xorg can hold B570 descriptors
and mappings even when every B570 connector is disconnected and another GPU
drives the monitor. Such ownership stops recovery before any reset. The earlier
GUI shutdown coincided with termination of this host's headless Orca service;
it cannot be treated as isolated from the running work environment.
The earlier automatic GUI-shutdown path is disabled, including its worker CLI.

Before a new reset, the entry checks for an interrupted, previously approved
display restoration from the same boot and caller. It can request SDDM startup
and check GPU health, but performs no reset or GUI shutdown in that path. It
does not reuse old-boot approvals. A running recovery or a stuck writer prevents
competing restoration. This compatibility path repairs only the earlier
root-owned 0644 plan inside its private 0700 directory; writable or linked plans
are refused. New plan replacements explicitly use 0600 regardless of umask.
Starting SDDM is a bounded request, not proof that the login screen appeared.

Other remaining clients, an attached B570 display, incomplete inspection and
interruption stop the sequence. The embedding service remains inactive and
disabled as requested; the script does not change it.
Client detection uses descriptors, mappings and blocked xe close stacks because
`xpu-smi ps` does not show every owner.

The result is printed on the terminal and a summary is saved at
`~/.local/state/strata-sycl/gpu-recovery/latest.json`. Detailed logs stay under
`/var/log/strata-gpu-recovery`. Exit 0 means the small GPU probe passed without
new xe fault/reset messages. Exit 3 meant the old detached GUI recovery was handed
off; it did not mean the GPU recovered. A health receipt must have
`healthy: true`, after another GPU probe covering the display-start request and
fresh kernel messages. A queued or interrupted receipt has `healthy: false`.
Exit 4 means recovery was declined or refused because a blocking condition
remains. Other failures report the step and its log. A successful
probe does not establish that a full Strata workload is correct.

The helper does not save devcoredumps, flash firmware, change xe's timeout/reset
policy, or reboot the host. It never stops services. It may start SDDM only to
finish the same caller's previously approved interrupted restoration. Each sysfs writer and GPU probe
runs in a separate process with a deadline. A writer still in uninterruptible
sleep after KILL is recorded by PID and start time; subsequent runs refuse to
compete with it. KILL cannot make a kernel D-state wait immediately disappear.

The internal helper is [sycl/tools/recover-xe.sh](../sycl/tools/recover-xe.sh),
installed alongside the entry point as `strata-xe-recover-core`. Its inspection
and developer controls are not needed for normal recovery. The legacy restoration helper is
[sycl/tools/recover-xe-display.sh](../sycl/tools/recover-xe-display.sh), installed
as `strata-xe-display-recover`; it cannot start a new GUI-shutdown worker. The probe is
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
caching off and copy offload off. Its library search path includes the installed
UMF 1.1 library required by the V2 adapter. An isolated dependency-load check
runs before any reset; a missing runtime refuses recovery. New kernel-journal
faults/resets invalidate a data match.

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
records the earlier 39-test implementation. The
[incident and revised validation](../bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/recovery-incident-20261006/record.json)
records 48 passing CPU test methods, shell syntax and the revised installed
digests. These cover reset ordering, method restoration, child timeouts,
misleading health results, no-argument shell sequencing, explicit consent,
changed/hidden owners, display ownership, borrowed locks, stranded writers,
service-stop failures, interrupted-worker receipts, manager umask changes,
same-boot restoration, runtime dependencies and disabled GUI-shutdown entry points:

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
The revised entry refuses that case without stopping the GUI. No display
shutdown or device reset was performed by the agent during these tests.
Successful no-reboot hardware recovery remains pending.
The full Strata arithmetic/context/cancellation/graph checks remain separate
from this 64 KiB GPU health check.

## GUI recovery incident on 2026-10-06

The human ran the earlier entry from the GUI at 18:57 JST and reported Orca and
Codex hanging, then rebooted the PC. The previous-boot service journal records
SDDM stopping at 18:57:59 and the recovery worker and its backup both exiting at
18:58:03 with `Unsafe recovery plan ownership/permissions`. The plan-save code
depended on the launcher's 0077 umask. A manager worker using 0022 instead writes
a 0644 replacement, which its own reader refuses; the backup cannot restore
SDDM either. Actual old save/read definitions reproduce the same exception with
CPU temporary files. The changed definitions preserve 0600 across replacements.
The privileged original plan has not been read, so its exact mode is inferred
from this reproduction and the journal, not directly observed.

The journal also records xe reinitialization during both attempts. The saved
kernel interval contains no kernel-panic/lockup signatures; that does not disprove
the reported whole-PC hang. Ending the GUI session and failing to restore it are
established defects. This is why normal recovery no longer stops the GUI.

Orca is actually a user `orca-headless.service` running with Xvfb; it is not
running on SDDM's display. Current unit dependencies have no `PartOf`/`BindsTo`
on the graphical session, and user lingering is enabled. The previous journal
records GNOME restarting the user D-Bus at 18:57:59, followed in the same second
by the Orca application scope exiting and its headless service killing remaining
Xvfb/crashpad processes. This contradicts an explanation based solely on Orca
using the display. The exact headless-service shutdown mechanism is being
investigated separately, without stopping that service or D-Bus.

After the human reboot, the first small probe exited with no SYCL GPU available.
Read-only Level Zero enumeration still found one device. The UR loader identified
the missing dependency `libumf.so.1`: the script's sanitized library search path
omitted `/opt/intel/oneapi/umf/1.1/lib`. Including that path let the unchanged
probe pass all three integer rounds at 20:25:31 JST, with no new xe fault/reset
messages. That check performed no reset or service operation. This establishes
small-probe health after reboot, not recovery without reboot or full-model health.

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
