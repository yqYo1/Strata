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
