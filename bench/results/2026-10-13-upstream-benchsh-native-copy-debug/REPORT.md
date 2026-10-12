<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Native COPY: pristine upstream bench.sh diagnostic

Both COPY0 and COPY1 completed upstream's unchanged v0.1.43 suite on Arc B570 /
Ryzen 5 5600X, with the same free binary b54579b and source 410904c. Command:
`bench.sh --url http://127.0.0.1:18095 --long --runs 3 --out DIR`.
Native IQ3_S, prefill 8192/group 16, cache 128 (144 effective slots), workers 5 /
tasks 18, no borrow/MTP/prompt reuse, int8 KV resident 32768/context 262144,
segmented verification, issuer 0, source-built DPC++ 7.1.1/UR Level Zero v2 and
DirectSubmission 0 were fixed. Only COPY selection changed between arms.
Both processes used validation and tracing: these are diagnostic timings.

| Run | Input/read | Reused | Output | COPY0 prefill | COPY1 prefill | COPY0 decode | COPY1 decode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32049 | 0 | 24 | 403.9 | 475.1 | 9.1 | 8.2 |
| 2 | 32049 | 0 | 24 | 400.5 | 477.3 | 8.7 | 9.1 |
| 3 | 32049 | 0 | 24 | 399.3 | 480.0 | 8.8 | 9.1 |

Rates are tok/s, all supplied by the engine through the upstream HTTP client.
Prefill medians are 400.5 and 477.3, a diagnostic increase of 19.18%; ranges are
399.3-403.9 and 475.1-480.0. Decode is assessed separately: the observed COPY1
range is 8.2-9.1, over only 24 generated tokens. No sustained decode benefit,
optimization adoption, quality result or objective completion is established.
The full upstream per-request artifact is copied byte-identically. Previous
raw-GEN results are not mixed with these HTTP measurements.

COPY1 selected the actual qualified copy-only group 1 before bench started.
Every request succeeded. The original server close wrote/flushed QUIT and both
engine and server exited zero without fallback signals. The anchored outer
owner reaped an empty session. Exact integer GPU health passed before and after;
all fresh kernel windows were fault-free, no devcoredump appeared, and source /
model / binary / provider identities remained unchanged. Five real CPU lifecycle
fixtures passed. No service, reset or device binding was changed.

The stock client's roughly-32K target produced 32,049 actual tokens. Workflow
guidance now compares this stock case by actual counts rather than imposing an
extra exact-32768 threshold that pristine calibration does not guarantee.
Historical receipts and their strict-threshold classifications remain immutable;
neither diagnostic arm is ranked. Separate 32768/64K/full physical 262144 claims
require their own actual evidence. Short suite stages and agent rows are auxiliary;
the persistent server's preceding suite stages warm allocations and graphs.

Nine new source reports are archived in registry v183 (881 completed reports).
Main rejected R349's unused-FP32-y deletion: the GDN normalization kernel reads
raw y, as R350 verifies. A qkv/y lifetime alias may reduce its subregion but may
save zero global bytes under the max-of-sets allocation; no change is adopted.
The phase-cache implementation exists in a separate worktree and has no build,
test, GPU or quality qualification. Its report's generic CUDA compiler suggestion
does not replace the machine's already qualified Intel-only contrib-llvm recipe.
Full-context probe coverage/finite/final-cell proof gaps remain open.

Next, execute matched quiet COPY0/1 stock suites from separate reviewed configs,
retain each long sample and judge prefill/decode separately. Main owns all new
phase-cache tests before GPU use. Successful per-token windows and duplicate
journal/console output are retired only after verified evidence extraction.
Original pinned receipts, individual upstream results and required fixtures stay.
