# Independent own prompt heads8 proposal v2

Source-only MIT package pinned to `617647659f21487ff3bf61fa1fd4ba87ba2520bf`.
Production kernel and CMake candidate bytes are identical to v1. Only this new
package was written; v1, live product sources, baseline fixture and reports remain
unchanged. All implementation qualification, compiler/build/test/runtime/GPU work
is UNRUN. Main owns complete review/application and all serialized validation.

Provenance: own current MIT `prompt_attention.cpp` and `prompt_k_cellmajor.cpp`
only as implementation references. v2 simplifies the own fixture and independently
improves coverage. R749/R750 and Main's instructions inform review requirements.
No XeStrata source/helper/test/archive or old unverified proposal was consulted.
Root LICENSE is retained byte-for-byte; no LGPL skill metadata was copied. Protected
prefill contract receipt was not read. `base/` preserves exact touched base bytes
and fixture provenance. `files/` holds candidate bytes; `prepare.py` regenerates
only the patch and manifest from these packaged bytes, without applying or testing.

Exact HEADS8=1 opts into Mode1 Direct1 Native=false KMapping0 HRows8/WG64.
Direct-off or exact KMapping1/2 rejects on Mode1 before submission. All other modes
retain their original dispatch; malformed/non-exact flags have no new validation
claim. Baseline HRows16 remains the default. Both groups reduce qmax over all12
heads. Global stride12, headbase0/8, zero padded query/probability rows and suppressed
tail stores preserve disjoint8+4 coverage. K/V/scales/rows and page work duplicate;
source DPAS work remains unchanged. No overall-byte saving, occupancy or causal
performance claim follows. PMU permissions are missing; use actual component time.

Local-byte arithmetic, correcting v1/R749's placeholder accounting:

| Source local declarations | baseline HRows16 | candidate HRows8 |
| --- | ---: | ---: |
| qhi/qlo | 16384 | 8192 |
| kv | 16384 | 16384 |
| phi/plo | 8192 | 4096 |
| scores | 2048 | 1024 |
| maxima/sums/alpha | 192 | 96 |
| ks/vs/factors/rows (512+512+16+256) | 1296 | 1296 |
| unused Direct final placeholder | 4 | 4 |
| declared sum | 44500 | 31092 |

Actual baseline UR launch arguments total44496B, excluding the optimized-away
placeholder. Candidate31088B is EXPECTED only if the compiler eliminates the same
4B placeholder; it is not measured. Candidate actual local args/static SLM/GRF
remain unknown. Two expected candidate groups total62176B, versus44496B across
one measured baseline group; these are not occupancy measurements.

The dedicated Mode1-only fixture removes unused FP16/Q4 pools, decode scratch and
old unrelated K-mapping proof. It keeps the independent FP64 quantized attention
oracle and logical unique8+4 output enumeration. Borrowed-host transfer uncertainty
aborts; per-run checked Drain destructs before allocation/host owners. No env writes.

`--smoke [--dump PATH]` runs exactly two bounded nq1/width33/page4 cases: valid
and mixed-missing pages, both with negative ID and strongly differing first8/last4
query amplitudes to exercise shared all12-head qmax. Every small-case live output
is checked by FP64 oracle. Default runs96 uniquely labeled initial cases:
nq1/3 × widths0/1/31/32/33/2051/-1/2052 × pages1/4/16 × masked0/1.
Each prints nq,width,page,masked,scaled and all gate outcomes. Smoke runs separately
and does not silently add cases to the96-case default.

Each small case checks initial full finite output, both prefix/suffix guards,
full FP64 oracle, batch/single exactness and unchanged Mode1 inputs. It then checks
all-pages-missing zero output, restores the original valid/mixed table, and tests
width0/-1/2052 independently. Each zero gate poisons the output first so omitted
stores cannot pass by inheriting zero. Both guards and expected-zero outputs are
checked after each gate; all inputs are compared with current expected table/steps
after each deliberate mutation. Single-query result readbacks also check both
guards. Initial invalid-width cases additionally use the full FP64 zero oracle.

`--bench 128|256 repeats1..20 --dump PATH` keeps all2051 widths. Initial output and
full guards are dumped once; the repeated sequence is compared in RAM with initial
full data and guards. No transfer/oracle/dump lies inside the repeated timing
interval. The benchmark's four-query FP64 oracle (0,1,2,last) is outside timing;
full finite/guard/input-unchanged checks cover the entire batch. Individual launches
inside the timed repeat sequence do not have intervening readbacks; the sequence's
complete observed result is checked afterward. No duplicate repeated or zero-gate
raw frames are dumped. Exact cross-process comparison remains Main's obligation.

Raw format and finite budgets: float32 native-endian, 16 prefix guards, nq*24*256
outputs, 16 suffix guards per frame; no header. Dump exactly one initial frame per
run() call in deterministic loop order. nq1 frame24704B; nq3 frame73856B.
Smoke:2 frames49408B. Default:96 frames4730880B (48 each nq).
Bench nq128:1 frame3145856B; nq256:1 frame6291584B, independent of repeat count.
The required controller raw-dump budget is7MiB (7340032B) per process. Text-log budget64MiB
per process plus a finite Main-selected deadline; the fixture does not enforce
controller supervision. Retain fingerprints/exact comparison before retiring raw.

Main's future serial gates, all UNRUN: after its cohort closes and complete patch
review, build only `sycl_prompt_heads8_test`; run smoke with Direct1/KMapping0 and
HEADS8=0/1 in separate immutable-environment owned processes, dump and cmp; then
run96 cases, then benchmark. Record actual commands/environments, binary hashes,
individual durations, drain/exit/fault results and journal boundaries. Source checks
for Mode1 Direct0+HEADS8=1 and Direct1+HEADS8=1+KMapping1/2 remain explicit future
separate-process rejection checks, not fixture PASS claims. Mode0/2/3 preservation
also remains source-only. No further approval is required for routine opt-in candidate
preparation; no runtime validation or adoption is implied by this package.
