# Short check waiting after each direct-FP16 expert

The frozen `a63f66eb…` executable completed four actual normal-MTP requests
with `STRATA_PREFILL_EXPERT_WAIT_BATCH=1` on Arc B570 on 2026-10-07. The first,
repeated, different and restored prompts matched the preceding released-weight
control in every output ID, printed logprob and complete finite
248,320-float head. There were 16,677 recorded one-expert queue waits and six
draft-weight release/restore pairs. Engine and debugger exited normally,
without new xe fault/reset or surviving process.

This changes only the optional setting on the same executable used for the
failed batch-32 capacity run. The short check retains context 128, chunk 32,
compact mode 2, layer-major mode 2, five CPU workers, normal MTP, direct
submission off and copy offload off. The native CPU task factor remains 0.
The comparator is the actual `79a4b363…` released-weight control, not a CPU
arithmetic reference. Weights, row order and reductions are unchanged.

Full Level Zero API entry/results, parameter validation, flushed UR tracing
and Strata progress were enabled. The diagnostic took 171.97 seconds; this
is not a throughput measurement. Its 1,052,787,352-byte stderr remains
private. The archive retains its SHA-256, counts, Strata progress and final
64 KiB. It records 1,114,739 API entries and 964,849 successful result lines;
other results are preserved without claiming that every query succeeded.

The updated owned debugger passed its four actual CPU/GDB checks before
this run. Its SHA-256 is
`61e1d206bf57bb320051cdd65943d23722bc523a602d80c54b7e7a2ad0d3f0d1`.
This successful model run did not need a stopped snapshot, so the CPU
checks provide the evidence for main-register capture.

The candidate's full-cell CLI and normal-MTP serving gates remain pending.
`run_full_expert_wait_one_serve.py` is the prepared capacity supervisor,
not a successful capacity receipt. It inspects 40 seconds of unchanged
progress before the application's 60-second watchdog, and reads CSR state
on a real crash stop too. EAGAIN attribution uses the explicitly marked
main-register section. It keeps finite request/overall/log limits and owned
cleanup. Captures and waits affect execution timing; no prevention or speed
claim is made from this short comparison.

`owned-expert-wait-1-short/` contains the terminal result and exact comparisons.
`build-receipts/` preserves the unchanged a63 engine's build identity.
`sources.json` and `manifest.json` preserve the sources and archived digests.
