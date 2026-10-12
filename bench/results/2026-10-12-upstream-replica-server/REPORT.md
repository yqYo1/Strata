<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Remaining v0.1.43 server functionality on Linux SYCL

The current remote versions were already evaluated without engine source patches:
official v0.1.43 at 7695859 and XeStrata xe0.1.42 at da3d426. Their existing 32K
diagnostics remain unchanged. There is no new inference revision to import.
See the [shared server qualification](https://github.com/yqYo1/Strata/blob/840db05/bench/results/2026-10-12-update-stream-recovery/REPORT.md)
for the four previously missing reasoning, stream and activity-monitor ports.

The next functional audit found opt-in replica serving and Linux physical CPU
grouping missing from the curated server. Both are now adapted from the official
source. Replica groups use canonical, disjoint PCI identities. Child processes
keep separate logs, learned-profile output and conversation persistence.
The existing single-engine default and its layer split remain available.

The replica front routes full requests with load and bounded prefix hints;
native prefix checks still decide actual reuse. Normal DONE obtained after an
EOS consumer close can establish the input hint. Cancelled or incomplete work
cannot publish one. Per-request statistics are copied before releasing the turn.
Slots and health/metrics expose their replica owner and per-child information.
Ambiguous single-file SAVE/RESTORE and dynamic VRAM actions remain unavailable
through the replica front, preserving their single-engine interfaces.

CPU masks use the parent's actual allowed physical cores and keep permitted
SMT siblings together. A separate Python child applies its own mask before
exec, retaining its PID. No Python preexec_fn runs in the threaded server.
The parent and other processes retain their masks.

Explicit per-replica RAM/VRAM reservations are checked before startup, reload
and recovery. RAM covers a conservative model-file floor including every native
GGUF shard, and intersects host availability with visible cgroup ancestors.
Unknown layouts or GPU availability refuse admission. Reservations are reviewed
capacity declarations, not hard RSS limits or proof of measured peak demand.
Initial eager and lazy load roll back every child after a partial failure;
unload/close visit other owned children even if one operation fails.

Main reviewed the SourceOnly proposal, then corrected shard counting, lazy-load
rollback, request-local statistics, cancelled route hints and EOS completion.
The new focused CPU/fake-HTTP suite passed 59 tests. The first focused run's
conversation-affinity failure remains FAILED in its original receipt; the fix
uses actual normal DONE evidence and does not weaken that assertion.
Mandatory source Python/docs/spelling/secret/license checks have no new findings;
the earlier failed lint receipts remain failures. The final focused suite,
including existing restart-waiter regressions, passed all 70 cases. Complete
server regression ran 622 tests: 611 passed and 11 existing optional-dependency
skips. All 47 selected setup regressions passed. The first complete run's restart
failure remains FAILED: Main's type-checking change bypassed the instance-level
restart hook. The corrected call preserves that hook and the original arguments
when no CPU mask is selected; no old test or assertion was weakened.

No model, GPU, reset, unbind or other service operation ran for this port.
Current device validation targets one Intel Arc B570. Multiple real SYCL device
groups, concurrent native inference, throughput and whole-model quality remain
unverified for this replica port.
Engine C++/CMake and the selected inference binary/settings are unchanged.
The existing pristine diagnostic samples cannot rank fully optimized settings;
the official-parameter TODO, physical 262144 input and RESTORE remain open.

See the source review, individual test outcomes, exact command/environment,
process closure and original/derivative hashes in the receipts. Successful full
mock API traces are retired after compact evidence is verified.
