# Documented oneMKL GPU CNR experiment at32K

The same production binary/input/configuration and process-local counter conversion
setting as the parent are retained. MKL_CBWR=AUTO is the only additional setting.
Intel documents GPU CNR for SYCL BLAS level3/GEMM and recommends AUTO for Intel
GPUs; it is also supported on the Ryzen host. This experiment tests that numerical
mode, without assuming BLAS variation causes the earlier discrepancy.

The logged first process and two unlogged state/head processes all complete32,768
input and64 finite-logprob output tokens, normal exit, complete owned cleanup and
no new xe fault. The first unlogged process matches all66prefill state parts,
all248,320 first-head float bytes, all64IDs and every protocol logprob of the
logged control. Their whole-head SHA256 is
053ccfe54cabe2528f8f44570684a8dd0258a2b0630006b2a86dfb61f777c586.

The second unlogged process differs in state parts1,3 and56..65, first-head bytes,
IDs and logprobs. The first differing saved QSA ordinal is10 (model layer43).
That is a later observed state discrepancy than in the preceding experiment,
without identifying its producing operation. CNR alone does not resolve output
reproducibility. CNR can change reduction choices, so equality with the older
non-CNR output is not used as a correctness requirement.

The failed state/head gate prevents either clean speed job from launching.
These jobs are functional/equality evidence only: state/head copies and diagnostic
logs exclude every rate here from performance comparisons. Three successful exits
and no xe fault do not establish model correctness or hang prevention. No production
setting, source/binary, package or service is changed. Full262,144-cell gates and
PP1000/TG70 remain open. The actual-DMA-event candidate is evaluated separately
under these same process-local settings; no candidate is adopted in this record.

Primary references: [GPU reproducibility conditions](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/reproducibility-conditions.html),
[GPU CNR configuration](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/getting-started-with-conditional-numerical.html).
