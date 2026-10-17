# Private GDN chunk64 candidate

Current unfinished Main implementation in `src/prefill/sycl/gdn_chunk64.cpp`
and `include/strata/prefill/gdn_chunk64.hpp`; this directory owns the fixture.
Repository MIT license applies. This
is independently authored SYCL code, not an external kernel port or adoption.
It preserves the loaded16/48head order, uses BF16 activation/scratch, FP32 carry,
a full64 FP32 inverse rounded once and genuine FP16 output for the existing
consumer. The schedule is not the SGLang block16/XPU numerical contract.

The previous compile stopped because fixture.cpp lacked the runtime.hpp include;
that include is now explicit. Host DMA buffers remain owned through failure-path
drain. An unused BF16 V copy was removed, saving 96 MiB of scratch at 8192 tokens
(workspace now 1,024,983,040 bytes). Exact dyadic T=1/3 and
63/64/65/127/128/129 checks passed on B570 with no bit differences, including
FP32 carry, BF16 readout, FP16 output, guards, inputs, rejection and repeat.
No framework numerical qualification, model speed/quality, full262144 or RESTORE
result exists yet. General cases compare against an independent sequential
FP64 mathematical reference and test aligned-call carry handoff; their error
metrics are diagnostic, not framework-contract or model-quality acceptance.
The first device run passed exact FP32 carry, but strict FP16 bit equality
rejected signed zeros (41 differences with zero absolute error at T=1).
Prospective smoke criteria now ignore only the sign of zero for BF16 readout
and FP16 output, requiring exact bits for every nonzero value. This is not a
general numerical tolerance, and original failed outcomes remain in Git.
The next diagnostic isolated a separate BF16 raw mismatch (maximum absolute
error 9.31323e-10): scaling the two cancelling FP32 readout terms separately
introduced residuals near zero. The candidate now adds before applying the
common scale, saving one multiply and avoiding that cancellation error.
The earlier raw-pointer matrix API error is fixed by explicit local/global
multi_ptr arguments. Superseded private source copies and build transcripts are
not required to continue this implementation.

Use the maintained measurement owner/guards in tools/sycl and set the shared
first-check environment before exec. Main serializes all builds/tests/GPU jobs.
See the retained GDN contract and current tuning documentation for the next
comparison; do not infer persistent framework pool dtype from track_state.

Commit the exact sources first, then run these separately under Main ownership:

```sh
python3 bench/sycl/gdn_chunk64/qualify.py build
python3 bench/sycl/gdn_chunk64/qualify.py smoke
python3 bench/sycl/gdn_chunk64/qualify.py boundaries
python3 bench/sycl/gdn_chunk64/qualify.py general
python3 bench/sycl/gdn_chunk64/qualify.py producer-smoke
python3 bench/sycl/gdn_chunk64/qualify.py producer
python3 bench/sycl/gdn_chunk64/qualify.py producer-large
python3 bench/sycl/gdn_chunk64/qualify.py component-check
python3 bench/sycl/gdn_chunk64/qualify.py component
```

The helper uses the shared measurement lock, bounded process supervision,
immutable validation/tracing environment and qualified runtime ELF identities.
It refuses to overwrite a previous candidate or reuse a changed source tree.
Current compact evidence goes to `bench/results/2026-10-15-pr6-recovery-audit/`
as `gdn-chunk64-current.json`; temporary full logs stay outside Git.

The native producer passed the 1/3 and 63/64/65/127/128/129 synthetic cases
on B570 at source `f31cdbd786e3d65c76045f6282a207a55fd9f8a4` without new GPU
faults. Conv output differences against the CPU formula were zero; history,
guards, inputs, rejection, repeat, split conv and aligned conv/carry handoff
passed. At source `68e58db6`, those cases plus 8192 also passed the shared arena
reuse and partial-call relayout, with zero conv CPU-formula differences and no
new faults. The lifetime-sharing layout at `ed1a9693` also passed all these cases,
including the exact disjoint/aliased pair comparison, partial-alias rejection and
rejection before conv history mutation. This is component safety evidence, not
framework/model qualification.
It reads BF16 projected QKV,
retains existing FP32 conv history, accumulates conv/SiLU locally in FP32,
rounds the activation to BF16, then normalizes Q/K with FP32 local reductions
and writes BF16 directly. It never materializes a full FP32 activation tensor.
The producer fixture checks a separate CPU formula (two BF16 ULP or absolute
1e-7 for its bounded synthetic inputs), exact history updates and split-call
invariance, and direct BF16 recurrence with aligned conv/carry handoff. This
diagnostic bound is not a general framework/model-quality tolerance. Projection
GEMM rounding, full-model execution, memory budgeting and quality remain separate
integration gates. Source now includes an unqualified opt-in long-prefill dispatch
with `STRATA_SYCL_PREFILL_GDN_CHUNK64=1`, requiring native16 BF16 and the existing
layer-major admission. The default and short-input/decode paths retain the prior
implementation. The second diagnostic model run completed a 39,260-token
official bench.sh request, thinking on, with8192 chunks and no new GPU faults.
It measured546.2 PP and14.1 decode tok/s with parameter-validation/trace logging;
this cold single sample is not a clean speed comparison or quality qualification.
Clean matched long-input timing and production adoption remain pending.

The shared arena planner is used by Prefill's size accounting/carving and the
fixture. Projection QKV dies after conv/history update and is reused by FP32
lower-KKT and the disjoint inverse. Normalized input dies after pack and shares
history storage; vb dies after W/U and shares FP16 output storage. Only those
two exact same-base recurrence aliases are admitted; partial aliases and other
overlaps are rejected. Preflight occurs before conv history changes.
The new fixture checks this reuse and partial-chunk rebinding against disjoint
buffers, with exact state/output/history and outer guards. The 8192 arena is
1,467,486,720 bytes for the original disjoint layout, excluding external
state/history, weights, KV, residuals and GEMM scratch. That layout made model
setup shrink chunk8192 to6144 and fail the existing MoE-window admission, with
no GPU fault. The lifetime-sharing layout passed requalification at
1,148,718,848 bytes (about1.070 GiB), fitting the existing
1,180,698,624-byte legacy GDN reservation at8192. These synthetic copies are test setup,
not transfers or host arithmetic in the inference path.

`python3 tools/sycl/build_engine.py` builds the current comparison engine after
verifying committed project/dependency inputs, using the known-good engine's
configure recipe. Main owns build/model runs; the build helper never runs the
model. Keep that engine only for the imminent native-GDN comparison.

For model checks, Main prepares exact committed inputs using
`python3 tools/sycl/bench_engine.py prepare check`, commits the generated compact
plan, then uses `run check`. After that check passes, the `base` and `candidate`
modes use the same engine/config/workload, changing only the GDN startup flag.
Each clean process runs four fresh >=64K prompts (first plus three subsequent),
thinking on with24 generated tokens. That cap qualifies only prefill mechanics
and timing; agent-session, sustained generation, quality and full-context/RESTORE
checks remain separate gates.

On B570, the resident8192 comparison uses identical decoded input values,
one warmup and five rotating samples per arm. Clean processes omit tracing and
validation layers. Queue intervals include submission gaps; individual stage
events measure device service. Uploads, resets and readbacks are outside them.

| Schedule | Median queue interval, ms |
| --- | ---: |
| Current FP32 recurrence | 23.428125 |
| Independent chunk64, original schedule | 21.436146 |
| Coefficients retained in unused inverse SLM | 21.097604 |
| Packed inverse, 16-column carry slices | 18.266458 |

The last schedule reduces this component's interval by 22.0%. Packed inversion
retains FP32 coefficients in the unused upper triangle with an XOR address
permutation, adding no SLM. Narrow carry slices request 14 KiB instead of 28 KiB
SLM and hold 16 instead of 32 FP32 state elements per workitem, doubling the
workgroup count. Carry service fell from 7.35 to 4.53 ms. Additional W/K reads are
a tradeoff; actual occupancy, physical traffic and hardware utilization were
not measured. These results establish a device-local resource/schedule gain,
not an end-to-end prefill speedup or a PCIe bandwidth result.

Both optimizations preserve the original candidate's complete 8192-token FP32
state and FP16 output bits. All dyadic/general safety checks passed without new
driver faults. Against the current FP32 recurrence, the candidate's 8192-token
relative L2 errors are 0.2954% for carry and 0.3751% for output; these are diagnostics,
not accepted model-quality limits. The full64 solve and rounding contract still
differ from SGLang's XPU block16 path. Native BF16 conv production, model
integration and framework/quality/physical-context/RESTORE gates remain open.
