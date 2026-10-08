# Short direct-FP16 expert queue-wait check

The optional `STRATA_PREFILL_EXPERT_WAIT_BATCH=32` candidate completed four
actual normal-MTP requests on Arc B570 on 2026-10-07. The first, repeated,
different and restored prompts matched the preceding released-weight control
in every output ID, printed logprob and complete finite 248,320-float head.
Six draft-weight release/restore pairs completed. There were 377 recorded
32-expert queue waits. Engine and debugger exited normally, with no new xe
fault/reset or surviving process.

The candidate waits for its existing in-order SYCL queue before submitting
the next group of direct-FP16 experts. Weights, row order and reductions are
unchanged. The setting accepts integers from 0 through 256; its default 0
keeps the preceding schedule. MMQ grouping is unchanged. An expert group is
not a count of kernel jobs, and these waits do not prove a bound on the
driver's live job count.

The frozen executable SHA-256 is
`a63f66eb289afa8b406434ccd7996b171fa93965e10b06eb7ba5decf528dab1c`.
Only the prefill object changed among 148 objects. Native CPU pool task
factor remains 0. The build receipts and builder source preserve the actual
incremental build and object comparisons.

This short check used context 128, prefill chunk 32, compact mode 2,
layer-major mode 2, five CPU workers and the normal MTP drafter. It retained
the preceding control's direct-submission and copy-offload settings, with
full Level Zero entry/results, parameter validation, flushed UR tracing and
Strata progress. The diagnostic took 170.30 seconds; this is not a throughput
measurement. Its 1,045,519,355-byte stderr remains private. The archive retains
its SHA-256, API counts, Strata progress and final 64 KiB. The trace records
1,098,438 API entries and 948,551 successful result lines; other result counts
are preserved without claiming that every API query succeeded.

The comparator is the preceding actual `79a4b363…` released-weight control,
not a CPU arithmetic reference. Passing these four requests does not prove
full-context capacity, general stall prevention or improved engine speed.
The candidate's own full-cell CLI and normal-MTP serving checks remain
required. `run_full_expert_wait_serve.py` is archived as the prepared capacity
supervisor; its presence here is not a successful serving receipt.

`owned-expert-wait-32-short/` contains the terminal receipt, exact request
comparisons and diagnostic logs. `build-receipts/` contains the actual build.
`sources.json` maps the sources used in the short run and prepared follow-up.
`manifest.json` records every archived file's size and digest. Executables
and binary head/state dumps remain in private host state.
