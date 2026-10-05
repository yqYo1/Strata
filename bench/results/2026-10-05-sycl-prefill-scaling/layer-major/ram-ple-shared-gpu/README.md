# 32K RAM PLE diagnostic with the embedding server resident

The Arc B570, native IQ3_S and validated AOT build process 32,768 tokens
with compact HC scratch, layer-major traversal, a fixed 65,538-position
K8/V8 context, 4K chunks and the first 4K residual rows in VRAM. The
separate embedding service remains resident on the GPU. This is a
separate memory condition from the earlier 32K-residual, 8K-chunk study;
these observations do not establish a speed comparison with that study.

Both the profiled and normal-wall runs preserve all 248,320 finite head
values and generated IDs against the original control with the same
4K chunks. There are no diagnostic PLE preloads. This uses the existing
production RAM table option. The [summary](summary.json) records commands,
actual locking outcomes, startup loads, sampled memory and peer memory.

| Observation | Prefill seconds | Tokens/s | Whole process seconds | PLE startup seconds |
| --- | ---: | ---: | ---: | ---: |
| Phase and transfer markers | 87.901 | 372.78 | 119.623 | 1.5 |
| Normal wall, no markers | 79.063 | 414.46 | 112.304 | 1.5 |

These are single observations, not repeated medians. The table is loaded
but not locked. Both table paths used in this study resolve to the same
file inode (2,623,800) and size (28,800,138,432 bytes); it was fully touched
by the earlier RAM proof. These 1.5-second startup observations are warm,
whereas the first proof's 42.3-second table load was the initial full
load. Neither startup time is included in the prefill timer.

The profile reports PLE host waiting of 20.326 ms, PLE gather wall time
of 260.888 ms, expert-DMA active time of 8.097 seconds for 50.292 GB,
and host residual DMA of 17.460 seconds for 110.394 GB. Copies within
VRAM total 15.771 GB in 0.096 seconds. The residual-copy phase interval
is 17.721 seconds and the dequantization interval is 17.239 seconds;
attention is 9.209 seconds. Phase intervals include host submission gaps
and instrumentation overhead, rather than isolated kernel execution.
Transfer and compute overlap and their durations must not be added to
obtain wall time.

Sampled peak engine RAM is 75.081 GiB for the profile and 75.046 GiB for
the normal wall case. VRAM is 7.944 and 7.937 GiB, respectively. No process
swap is observed. Samples are at 1 Hz and can miss peaks; they do not
include the separate embedding server's VRAM in the engine values.
