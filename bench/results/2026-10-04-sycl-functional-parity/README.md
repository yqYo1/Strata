# Single-GPU functional ports

The comparison used upstream `99f3dbd`, engine 0.1.38, on an Intel Arc B570
with a Ryzen 5 5600X and 125 GiB RAM. Multi-GPU and Intel setup automation were
excluded from the requested work. The shared server and installer remain the
imported upstream code.

The SYCL ports add four native formats (Q5_1, IQ2_XXS, IQ2_XS and IQ1_M), routed
quantized MMQ for all 17 upstream MMVQ formats, fused Q2_0/native expert prefill,
matrix prompt attention for all four KV modes, and the optional Intel GPU image
encoder. Hybrid KV validation also found and fixed rejection of upstream's
folded K/V buffers; the SYCL append now gives each physical output one writer.

`run.json` records independent kernel references, full-vocabulary prefill
comparisons and real IQ3_S integration checks. The vision encoder comparison
uses a separately built CPU encoder and the same pinned projection model.
Real HTTP checks cover both image APIs, repeat caching, streaming, two-image
order and subsequent text/image requests.

These are functional fixtures. They do not establish general model quality,
maximum context support or new performance medians. MMQ and fused experts use
integer SIMD; matrix prompt attention uses Intel `joint_matrix`. CUDA-specific
instructions and scheduling are not reproduced, and unsupported matrix
devices/shapes use the split fallback. Background workstation loads were not
isolated.

Reviewed summaries and test source are committed. Raw logs, per-request JSON,
float dumps, temporary fixtures/configs and unsuccessful development attempts
remain outside Git in
`~/.local/state/strata-sycl/measurement-archive/2026-10-04-functional-parity/`.
The archive manifest identifies these files and their hashes. See the
[implementation report](../../../docs/SYCL_STATUS_2026-10-03.md),
[vision build instructions](../../../docs/SYCL_IMPLEMENTATION.md#intel-gpu-image-encoder-2026-10-04)
and [measurement storage policy](../README.md).
