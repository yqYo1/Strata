# Independent own prompt heads8 proposal

Source only, MIT, base commit `617647659f21487ff3bf61fa1fd4ba87ba2520bf`.
No build, compiler, test, GPU, profiler, environment mutation, service operation,
cleanup or Git mutation was performed. Main owns complete patch review, application,
and serialized validation after its running cohort closes. This package does not
change live source. `prepare.py` regenerates candidate files and diff from retained
hashed base bytes; it never applies them. All checks described here are UNRUN.

The only implementation references are current own `prompt_attention.cpp` and
`prompt_k_cellmajor.cpp`. The own CMake target declaration was read for wiring.
Root LICENSE is MIT and retained byte-for-byte. No XeStrata source, helper, test,
archive or old unverified proposal was read or used. R747/R748 and Main's resource
status are research/context only. No LGPL skill metadata was copied. The protected
prefill contract receipt was not read.

Exact `STRATA_SYCL_PROMPT_HEADS8=1` opts into Mode1, Direct1, Native=false,
KMapping0, HRows8/WG64. Other flag values retain the original route. On Mode1,
HEADS8=1 with Direct disabled or exact KMapping1/2 throws invalid_argument before
submission. Non-Mode1 ignores HEADS8 and retains its original dispatch; no expansion
of this experiment to FP16/Q4/mixed modes is claimed. Existing support gate remains
conservative (WG128 and original local-memory requirement); no support expansion.

Two groups cover each query/KV-head, headbase0/8 with global stride12. Both reduce
max-absolute query over all12 original heads, preserving normalization; no cross-group
reduction. Tail query rows and probabilities are zero, and tail output is suppressed.
QK uses two of four subgroups, PV all four; branch predicates are subgroup-uniform.
All original barriers remain unconditional. Original HRows16 arithmetic/layouts remain.
K/V tile, K/V scales, factor array and cell rows are unchanged and duplicated.

Independent Mode1 Direct launch-local byte derivation:

| Arrays | HRows16 | HRows8 |
| --- | ---: | ---: |
| qhi/qlo: 2*HRows*256*2 | 16384 | 8192 |
| kv: 32*256*2 | 16384 | 16384 |
| phi/plo: 2*4*HRows*32*2 | 8192 | 4096 |
| scores: HRows*32*4 | 2048 | 1024 |
| maxima/sums/alpha: 3*HRows*4 | 192 | 96 |
| ks/vs/factors/rows: 512+512+16+256 | 1296 | 1296 |
| final Direct placeholder | 4 | 4 |
| total | 44496 | 31088 |

R747's 1168 and 30960 are incorrect. Two groups total62176 launch-local requested
bytes versus44496 across one original group. Source DPAS totals are unchanged;
logical K/V gathers and cell/page work duplicate. No overall-byte reduction,
occupancy improvement, hardware cause or performance benefit is claimed. PMU
permissions are missing per R748; Main must use actual component timing.

The new EXCLUDE_FROM_ALL target `sycl_prompt_heads8_test` leaves the old fixture
and target untouched. It extends only the current own fixture. Small cases use actual
nq1/3, pages1/4/16, reversed pages/reordered IDs, negative IDs, mixed/all missing
pages, widths0/1/31/32/33/2051 and -1/2052. Full FP64 oracle checks every live
output; explicit logical enumeration checks unique 8+4 coverage. Query amplitudes
differ sharply between first8/last4 to exercise shared qmax. Prefix/tail output
guards, input immutability and batch-versus-single exact checks are included.
Benchmark nq128/256 repeats1..20 dumps full outputs plus tail guards, verifies
prefix guards/full finite/repeated exact/input unchanged; FP64 query subset0/1/2/last
is outside timed work and explicitly labeled. Full oracle applies to small cases.
Borrowed host transfers fail-stop on uncertain completion; per-run Drain waits and
aborts on drain failure before host/device owners unwind. No env mutation occurs.

Main's proposed commands, UNRUN (after applying reviewed patch and obtaining lock):

```
cmake --build <configured-build> --target sycl_prompt_heads8_test
STRATA_SYCL_PROMPT_DIRECT=1 STRATA_SYCL_PROMPT_K_CELL_MAJOR=0 STRATA_SYCL_PROMPT_HEADS8=0 <fixture> --dump <baseline.bin>
STRATA_SYCL_PROMPT_DIRECT=1 STRATA_SYCL_PROMPT_K_CELL_MAJOR=0 STRATA_SYCL_PROMPT_HEADS8=1 <fixture> --dump <candidate.bin>
cmp <baseline.bin> <candidate.bin>
```

Benchmark invocation: `<fixture> --bench 128|256 1..20 --dump PATH`, fresh process
per arm; Main records exact environment, binary/source hashes, individual durations,
fault checks and owned-process exit. Wrong combinations are source-review checks
only and remain UNRUN: Mode1 Direct0+HEADS8=1 and Direct1+HEADS8=1+KMapping1/2
must throw; Mode0/2/3 must retain original dispatch. These are not qualified results.
No raw captures exist to clean. Main owns retention of future comparison dumps
with a finite byte budget and retires them after fingerprints/exact comparisons.
