# Arc B570 prompt-length scaling

This measurement separates expert-weight transfer work from the way prompt
latency grows with input length. The device is an Intel Arc B570 10 GB, with a
Ryzen 5 5600X and 64 GB of system RAM. The model is the native IQ3_S
Qwen3.8-Flash-Next pack. The selected compute runtime is 26.35.39758.10.

The earlier 826- and 4,007-token measurements used different context limits and
chunk settings. The engine also defaults to a separate 256-token first chunk
to overlap PLE disk reads. That repeats the streamed expert-weight work and
makes those two measurements unsuitable for estimating a per-token cost.

[prefill_profile.py](../../../sycl/tools/prefill_profile.py) fixes the context
limit, cache capacity, prefill chunk, and input token prefix across lengths.
It sets `STRATA_PREFILL_FIRST=0` and checks that every run processes exactly
one chunk. It alternates the length order between repetitions. Model loading
and generation are outside the prefill timer. By default it omits the MTP
draft layer; `--mtp DIR` also measures the production draft-KV work.

The code-review fixture contains the beginning of three repository files from
revision `e274e17`. Its prefixes are diagnostic causal inputs, rather than
separately completed chat turns. The fixture metadata records the source
excerpts and tokenizer hashes. This avoids repeating one short sentence to
fill a long context.

| Measurement | Meaning |
| --- | --- |
| `expert_bytes`, `expert_copies` | Actual non-resident expert blobs copied to the GPU, including layer totals. Dense weights loaded once at startup are outside this count. |
| `dma_active_ms` | Sum of the memcpy events' device start-to-end durations. Excludes queue gaps. |
| `copy_span_ms` | Time from the first copy's device start to the last copy's device end. Includes gaps for routing, staging, and occupied ring slots. |
| `host_copy_worker_ms` | Sum of CPU staging/read durations across the worker threads, excluding their buffer waits. Parallel worker durations are not wall time. |
| `phase_ms` | Compute-queue intervals between phase markers. Includes launch gaps and waits; this is not a sum of kernel execution times. |
| `wall_ms` | Elapsed prefill time captured before querying and printing the profiling records. |
| `marginal_ms_per_token` | Difference in median wall time divided by the token-count difference for the two longest inputs. |

DMA, staging, and compute overlap. Their durations cannot be added to get
wall time, or subtracted from wall time to isolate computation. MoE routing
can change the transferred bytes with input length. Interpret the measured
latency slope alongside those bytes, rather than assuming the whole model
is copied for every input. Attention and other token-dependent work can also
make the slope change with context length.

Use `--mode profile` for phase markers and transfer events, `--mode transfer`
for transfer events alone, and `--mode wall` for elapsed time with both forms
of instrumentation disabled. The wall runs provide the speed comparison;
the profiled runs explain it. Compare first-logit hashes at matching lengths
to check that instrumentation leaves the model output unchanged.

The initial instrumentation check used the earlier 826-token, two-chunk
configuration, with MTP and 512 cache slots. It copied 24,468 expert blobs
(50,434,304,000 bytes), with 7,926.15 ms of active DMA and 2,619.90 worker-ms
of CPU staging. All 248,320 first-logit floats were bit-identical to the
unprofiled accepted AOT binary, and the eight generated token IDs matched.
The profiled prefill wall time was 13,830.73 ms; phase markers introduce
overhead, so this is not a speed improvement over the unprofiled baseline.
