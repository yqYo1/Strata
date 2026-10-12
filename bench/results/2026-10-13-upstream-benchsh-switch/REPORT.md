<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Switch model speed comparisons to pristine upstream bench.sh

User instruction replaces custom raw-GEN performance controllers with unchanged
upstream bench.sh. Official v0.1.43 at7695859 provides the common HTTP client for
upstream, XeStrata and local configurations. Main prepared a dedicated loopback
server config and verified the actual bench.sh and server CLI entrypoints without
launching a model. A real ignored .venv directory reuses the installed Python and
libraries. The first symlink-only setup failed pristine-status checking and is
preserved. Upstream tracked source stayed unchanged.

The normal command is `bench.sh --url http://127.0.0.1:18095 --long --runs 3
--out PRIVATE_RESULTS`. Preserve upstream prompts, requests, timings and individual
results. Use actual long-row input/read/reused counts, with short/agent results
auxiliary. Standard long output cap is24. Assess prefill/decode separately and
repeat unresolved variation; generated text equality is not a quality gate.
Upstream can return exit0 with failed requests, so inspect results.failures too.
The new bilingual workflow and AGENTS link make this durable.

Main's full compiled-source closure for the native-copy candidate passed: actual
source inventory, current Ninja no-work check, relevant TU compilation lines from
both real builds, header dependencies, compiler macros and link recipe. Binary
b54579 remains unchanged at source410904c9. No model qualification transfers from
this host check. The earlier raw-GEN source has seven passing CPU fixtures but
production admission mismatches; its revised Main packet has a failed namespace
fixture. Both outcomes are preserved, and all such performance controllers are
now superseded, with no model run or speed claim.

Two CPU fixtures for the new supervisor's upstream result failure/geometry
classification passed. The supervisor and shutdown observer remain proposed:
process/lifecycle safety is not qualified by those two numerical-free tests.
Before execution, Main must bind the actual engine log inside the64MiB budget,
normalize quiet STRATA_TRACE to absent, validate complete actual engine argv/env,
and review shutdown-only observation, known-child races and no-restart behavior.
The proposed observer currently wraps close globally; gate observation at the
verified shutdown boundary before use. Check available memory/cgroup/device-idle,
fresh integer health and fault cursors, finite outer owner and normal QUIT/reap.
Then run separate first diagnostics and quiet standard bench.sh measurements.
No new HTTP/model benchmark, performance adoption or full262K result exists here.

R338 explains approximate32K calibration and why long decode must use request
rows rather than the short-chat summary. R339's phase-mode incompatibilities
concern the separate untested phase-cache proposal: leaving that opt-in OFF is
intentional for the ordinary baseline, not a baseline configuration defect.
Phase VMM/context granularity and checkpoint/restart compatibility remain future
work. Quality capture adapter/held-out tasks also remain open. Goal remains active.

Source, exact config, commands, outcomes, failures and pending supervision source
are retained. Successful closed command-log duplicates are retired only after
verified extraction; pinned receipt bytes and current source/recipe consumers
remain. Original tiny API receipts stay under their existing finite exception
until new benchmark admission retires the remaining historical pin consumers.
