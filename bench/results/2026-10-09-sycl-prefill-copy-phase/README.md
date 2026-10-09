# Prefill copy resource lifetime experiment

This branch adds `STRATA_PREFILL_COPY_PHASE_RELEASE=1`, requiring
`STRATA_PREFILL_COPY_ENGINE=1` without transfer profiling. After a successful
prefill and optional state dump, the compute and copy queues finish, the host
stager joins, expert ring events are released, and the registered copy queue
owner is removed. The next prefill recreates those transfer resources.

Compute buffers, KV state, PLE uploads on the compute queue, decoder graphs and
weights remain live. The default is off. Removing the new guarded code reproduces
the entire previous prefill source byte for byte; decoder, CPU and common header
sources are unchanged from `6caa1421f9212750a425fe1729139ffdde6e9f9a`.

This is an unqualified experiment. Repeated 32K measurements of the previous
candidate found about 8% slower decode immediately after fresh prefill, so that
candidate is held back. Compare this experiment on/off in the same binary and
judge prefill and decode separately, including cold decode and restored decode.
First-use logged 32K numerical checks and complete physical 262144-cell lifecycle
checks are required before adoption. Logged runs are excluded from speed claims.
No production default is changed.


The repeated comparison stopped on its seventeenth process, `phaseon` block 6.
The engine exited normally with no recorded GPU fault. All 64 greedy token IDs
matched, but all 64 logprob rows differed from the numerical reference; MTP
accepted/offered counts were 39/72 instead of 41/66. This is a failed correctness
check. The eighteenth planned process was not started. Sixteen good processes
contain 192 measured restored requests; the five complete balanced blocks are
descriptive partial results, not a completed adoption comparison. Failed times
and the sixth unpaired baseline are excluded from paired analysis.

A subsequent fully logged first-head/all-66-state replay passed, as did three
reduced-log head-only replays. Logging and capture can change scheduling, so
these passes do not clear the intermittent failure. CPU-only sanitizer tests
of the extracted actual Stager fragment passed, including a non-PIE TSan run.
The initial TSan executable failed at startup because of its address layout;
both receipts are retained. These tests cover host buffer reuse and generation
handoffs with fake DMA, not SYCL, driver or model correctness.

The cause remains unresolved and this branch remains held. No physical 256K
qualification or production adoption is claimed. Receipts and the numerical
failure review are archived here; multi-gigabyte API logs and captured tensors
remain in the private paths listed in the manifest. Their logged timings are
excluded from performance conclusions.
