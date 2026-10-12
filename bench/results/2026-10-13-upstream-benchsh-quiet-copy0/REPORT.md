<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Quiet stock bench.sh baseline and phase CPU checks

The unchanged official v0.1.43 benchmark completed with COPY0, validation and
tracing disabled, on Arc B570 / Ryzen 5 5600X. Same source 410904c and free binary
b54579b as the diagnostic arms; source-built DPC++ 7.1.1, UR Level Zero v2,
DirectSubmission 0, native IQ3_S, 8192 prefill/group16, workers5/tasks18,
cache128/effective144, no borrow/MTP/prompt reuse, int8 KV resident32768/context262144.
The client is 7695859, with `bench.sh --url http://127.0.0.1:18095 --long --runs 3`.

| Run | Actual input/read | Reused | Output | Prefill tok/s | Decode tok/s |
| --- | --- | --- | --- | --- | --- |
| 1 | 32049 | 0 | 24 | 407.5 | 9.7 |
| 2 | 32049 | 0 | 24 | 408.0 | 8.6 |
| 3 | 32049 | 0 | 24 | 403.0 | 7.9 |

Prefill median is 407.5 (403.0-408.0); decode is 8.6 (7.9-9.7). All metrics are
engine-sourced through the original upstream client. Retain all samples: decode
variation is substantial, and the 24-token scope does not prove sustained decode.
Every request succeeded, actual quiet health matched seven providers, COPY0 was
verified before bench, and original QUIT wrote/flushed with engine/server exit0,
no fallback signals and an anchored empty reaped session. Fresh kernel windows
were fault-free; pre/post GPU health and all final identity hashes passed.
The complete upstream results are byte-identical. No reset or service changed.

These actual roughly-32K stock rows are eligible for the declared stock comparison;
the historical strict32768 fields remain false and immutable. Generated-text
identity is not a universal quality gate. No COPY1 quiet result or optimization
adoption is claimed yet; earlier debug rates and raw GEN are separate workloads.
Full physical262144 and model quality remain unqualified.

At the closed GPU boundary, Main compiled the separate phase owner CPU target
with source Clang22, Release/NDEBUG and WERROR, then ran its actual lifetime,
readback, mapping, publication and failure fixtures. Release and host ASan/UBSan
both passed. The first full server suite failed three cases because system Python
lacked regex; the original failed receipt is retained. With the existing qualified
venv, the phase lifetime and full server suite passed without fixture changes or
package installation. These checks do not qualify the SYCL VMM/backend, captured
GPU graphs or model inference; full free/contrib-llvm builds/lints and GPU checks
remain required before using that candidate. Its source worktree is uncommitted.

The phase timing source now charges previous GEN retirement/reinit and current
retirement/map/fill/readback/publication to prompt_ms. Main corrected documentation
that described PrefillStats as compute-only: its ms_total includes run transfer
and synchronization wall time. The current measured binary remains unchanged.
Seven source reports are archived in registry v184 (888 completed reports).
R356/R357 give corrected current allocation winners and clipped alias savings;
16K fit is not established. Main records source/registry/byte-arithmetic corrections
separately while preserving the original reports and failed statuses.

Next: quiet COPY1 with this same pristine suite/binary; judge phases separately,
and repeat decode campaigns if necessary. Main owns all further phase validation.
Successful raw progress/polling/journal duplicates are retired after verified
extraction; original pinned receipts and individual upstream results stay.
