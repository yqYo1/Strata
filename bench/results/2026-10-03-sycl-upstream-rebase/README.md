# Upstream sync and SYCL rebase validation

PR #1 merged upstream engine 0.1.38 into the fork main, then all 77 SYCL commits
were rebased onto that main. `run.json` is the reviewed integration record for
B570/5600X: 35 CTest checks, 139 server tests, 11 setup scripts, ordinary CPU
reference checks, a six-request persistent IQ3_S process, real HTTP checks and
the saved Q2_0 preset's 16-token regression.

The persistent rates are one fresh-process sample, not new paired medians.
Earlier IQ3_S tuning numbers remain in the adjacent historical summary. That
summary's CPU-reference label was corrected: `--expert-cache 0` with a profile
selected automatic sizing (2,530 slots), rather than disabling the cache. This
record includes a true disabled-cache condition without a profile. Original
numerical values were preserved; the two comparable full-vocabulary dumps also
remained bitwise equal after rebase.

The main sync contains upstream unchanged. Two test fixture portability issues
were fixed on the SYCL branch so the Windows archive fixture and golden config
normalizer work on Linux. They change no installer behavior. CUDA-only kernels
still need explicit SYCL ports; the peer device is rejected before model loading.

Raw logs, per-request JSON and outputs, float dumps, the learned profile and
local configs are outside Git at
`~/.local/state/strata-sycl/measurement-archive/2026-10-03-upstream-rebase/`.
See the [implementation report](../../../docs/SYCL_STATUS_2026-10-03.md) and
[measurement storage policy](../README.md).
