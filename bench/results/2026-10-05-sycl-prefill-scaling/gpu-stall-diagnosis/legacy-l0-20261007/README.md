# Installed legacy Level Zero adapter short comparison

The installed legacy UR Level Zero adapter passed the unchanged small B570
GPU probe and four actual normal-MTP model requests on 2026-10-07. All three
probe rounds checked 16,384 exact integer words after H2D, kernel and D2H.
The model's first, repeated, different and restored requests matched the
preceding released-weight control in every output ID, printed logprob and
complete finite 248,320-float head. Six release/restore pairs completed.
Both tests exited normally without new xe fault/reset or surviving process.

The frozen engine remains `a63f66eb…`; its optional expert waits were set to
0. The legacy adapter SHA-256 is
`ec3e2a2672db65f99c0d55b233f42602a8415ab19c177c4330f2e5ed5a78eedf`.
`UR_ADAPTERS_FORCE_LOAD` selects the installed
`libur_adapter_level_zero.so.0`, with `UR_L0_USE_COPY_ENGINE=0`. The V2-only
copy-offload flag is removed. Loaded-adapter logging confirms legacy rather
than V2. Direct submission remains disabled and persistent caching remains
off. No driver, package, service or firmware change was performed.

The [official UR reference](https://oneapi-src.github.io/unified-runtime/core/LEVEL_ZERO.html)
documents value 0 as disabling copy engines. In the actual small probe,
legacy uses `zeCommandListAppendLaunchKernel`, while the preceding V2 probe
uses `zeCommandListAppendLaunchKernelWithArguments`. Both probes create
immediate command lists; this is not evidence that legacy automatically
batching all work prevented the model stall. An adapter comparison changes
more than one internal runtime path and does not isolate a root cause.

The model check retains context 128, chunk 32, compact mode 2, layer-major
mode 2, five CPU workers and normal MTP. Full API entry/results, parameter
validation, flushed UR tracing and Strata progress were enabled. The
diagnostic took 191.79 seconds, not a throughput measurement. Its complete
large stderr stays private with digest, counts, progress and final 64 KiB
archived. Other API query results are preserved. No independent CPU
arithmetic reference is claimed.

Legacy's own full-cell CLI and normal-MTP serving gates remain pending.
`run_full_legacy_l0_serve.py` is the prepared capacity supervisor, not a
successful capacity receipt. Its read-only CSR script also accepts the
completion-wait frames that the preceding submission-only script missed;
that prepared reader is not a live counter measurement in this short run.
No general prevention or speed claim is made.

`legacy-l0-health/` preserves the actual probe and loaded-library identity.
`owned-legacy-l0-short/` preserves the actual requests and comparisons.
`sources.json` and `manifest.json` preserve the sources and receipt digests.
