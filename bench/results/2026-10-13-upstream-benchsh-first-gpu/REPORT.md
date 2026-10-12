<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Upstream bench.sh: first supervised GPU run

The unchanged official v0.1.43 benchmark client completed its full suite against
the current local free engine on Arc B570 and Ryzen 5 5600X. This verifies the
new benchmark route, actual HTTP timing fields and normal server shutdown.
It is a diagnostic run with validation and tracing enabled; it is not ranked.

Command: `bench.sh --url http://127.0.0.1:18095 --long --runs 3 --out DIR`.
The client revision is 7695859; the inference source is 410904c (C++ a825e23),
with binary b54579b. Full identities, commands and environment are retained.
Settings: native IQ3_S, context 262144, int8 KV resident 32768, no growth,
prefill 8192, expert groups 16, no borrow/MTP, pool workers 5/tasks 18,
cache 128 (effective 144 slots/281 MiB), no prompt/conversation reuse or swaps,
issuer 0/ring 16, native dequant batch 1 and native COPY option 0.
The compiler is source-built DPC++ 7.1.1 with libsycl 9 and the Level Zero
v2 adapter; DirectSubmission is zero and segmented verification is enabled.

The original upstream results.json retains every individual request and all
suite sections. Its nominal 32K row had these actual engine measurements:

| Run | Input/read tokens | Reused | Output | Prefill tok/s | Decode tok/s |
| --- | --- | --- | --- | --- | --- |
| 1 | 32049 | 0 | 24 | 403.9 | 9.1 |
| 2 | 32049 | 0 | 24 | 400.5 | 8.7 |
| 3 | 32049 | 0 | 24 | 399.3 | 8.8 |

Both rates came from the engine, without client rate fallback. The upstream
prefill median is 400.5 tok/s (399.3-403.9). Do not compare these diagnostic
rates with the old raw-GEN measurements or use them to adopt an optimization.
All actual inputs are below the strict 32768 threshold. The pristine CLI
estimates input length and supplies no correction or 64K/output-cap option.
The long row measures at most 24 decode tokens after prefill, not sustained
decode. Short chat and agent sections remain auxiliary results.

All requests succeeded. The process remained the same persistent engine;
earlier suite stages warmed allocations, graphs and runtime before the long
row. Prompt-cache zero prevented conversation reuse despite default ckpt=1.
Five CPU fixtures passed, including real original-close QUIT success, bad
child exit and fallback-signal detection. At idle/in-flight zero, the owned
server received pidfd TERM and its original close wrote/flushed QUIT. Engine
and server exited zero without fallback signals. The outer owner observed an
anchored empty session and reaped it. Exact integer GPU health passed before
and after; fresh kernel windows had no faults/resets and no devcoredump appeared.
No services, device bindings or resets changed. Identity hashes matched after
closure. Full physical 262144-input behavior and model quality remain open.

The next model comparisons use the same pristine client with separate quiet
server processes and actual lengths. Prefill and decode decisions stay separate.
Native COPY model performance, phase-cache changes and upstream parameter
optimization are not adopted by this run. Read-only research repeated across
distinct scopes while Main supervised the actual measurement; reports are
archived here and linked through registry v182 (872 completed reports).

Useful progress, health/provider, lifecycle and complete fault-relevant evidence
were extracted before retiring duplicate normal logs and cumulative journal
windows. Original pinned receipts remain byte-identical; all statuses, including
the seed-v1 missing user-bus environment failure, are preserved. Model/session
fixtures and protected native-copy qualification receipts were not removed.
No quality, precision benefit, speed gain or objective completion is claimed.
