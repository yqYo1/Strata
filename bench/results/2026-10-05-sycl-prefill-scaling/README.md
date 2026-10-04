# Arc B570 prompt-length scaling

This measurement separates expert-weight transfer work from the way prompt
latency grows with input length. The device is an Intel Arc B570 10 GB, with a
Ryzen 5 5600X and 128 GB of installed system RAM (125.72 GiB reported by
`MemTotal`). The model is the native IQ3_S
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

The first length sweep fixes the context at 8,192, the prefill chunk at 8,192,
and the cache at 128 slots, with MTP omitted. All three inputs fit one chunk.
The wall-time result is the median of three runs, ordered short-to-long,
long-to-short, then short-to-long. Transfer measurements are from one separate
profiled run per length, rather than from the uninstrumented speed runs.

| Input tokens | Median wall time | Overall prefill | Profiled expert bytes | Profiled active DMA |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 12.657 s | 161.81 tok/s | 42.496 GB | 6.674 s |
| 4,096 | 17.340 s | 236.22 tok/s | 45.402 GB | 7.129 s |
| 8,087 | 24.582 s | 328.98 tok/s | 46.989 GB | 7.374 s |

Between the two longest inputs, wall time grows by 7.243 seconds for 3,991
additional tokens: 1.815 ms/token, or 551.05 additional tokens per second.
This is a measured local slope, including routing, attention, and PLE reads;
it is not the GPU's pure arithmetic throughput. The transferred expert bytes
grow by 1.587 GB over the same interval. The 8,087-token profiled run reads
90,360 SSD pages for the PLE table; the prompt path reports 6.169 seconds of
PLE handling, most of it waiting for the row gather.

The model files are on ZFS (`rpool/USERDATA/yayoi`, 128 KiB records,
compression off, `primarycache=metadata`), and the direct reader uses
the Linux default of 16 blocking-read threads. This storage and reader setting
is part of the measurement condition. It affects the token-dependent row
reads as well as the GPU work. Future comparisons keep the file and GPU
settings fixed and report any change to the reader concurrency explicitly.

All nine wall runs produce finite first logits that match the corresponding
profiled run bit for bit across all 248,320 values. The generated token also
matches at each input length. See the [profiled record](profile/run.json),
[wall record](wall/run.json), and [parity checks](profile-wall-parity.json).

Increasing `STRATA_IO_THREADS` from 16 to 64 did not improve the longest-pair
slope: 1.833 ms/token versus 1.815. PLE handling times were also similar. All
nine runs retain the baseline's first-logit bits. The [64-thread record](io64/run.json)
is an experiment, rather than a new default.

The CPU-only [PLE probe](../../../sycl/tools/ple_read_probe.cpp) calls the
original table gather and decoder. Its [record](ple-probe/run.json) separates
cold reads from page-cache hits: the first 8,087-token mmap gather took
21.098 seconds, and the repeated gather took 0.157 seconds. The later direct
reads also benefited from the warmed pages. Their speed is not evidence that
64 threads beat 16. The embedding hashes agree in all modes and repetitions.

`--preload-ple` is an explicit diagnostic mode. It completes the same PLE
gather before the GPU/expert-transfer timer, reuses its embedding, and reports
the preload separately. The CLI's enclosing timer still includes it. It
requires one chunk, and does not disable the PLE or replace its values.
The [preloaded record](preloaded/run.json) has one profiled run per length:
GPU/expert-transfer stage times of 12.267, 14.705, and 18.598 seconds. The
4,096-to-8,087 local slope is 0.975 ms/token (1,025 additional tokens/s).
This diagnostic slope is not overall prefill throughput or a measured speed
improvement. All three first-logit arrays match the unmodified baseline bit
for bit. The extra phase markers separate PLE read waits from combine and
hyper-connection write/normalization time.
