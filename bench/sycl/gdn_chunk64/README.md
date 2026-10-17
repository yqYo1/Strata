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
python3 bench/sycl/gdn_chunk64/qualify.py component-check
python3 bench/sycl/gdn_chunk64/qualify.py component
```

The helper uses the shared measurement lock, bounded process supervision,
immutable validation/tracing environment and qualified runtime ELF identities.
It refuses to overwrite a previous candidate or reuse a changed source tree.
Current compact evidence goes to `bench/results/2026-10-15-pr6-recovery-audit/`
as `gdn-chunk64-current.json`; temporary full logs stay outside Git.

The native producer is currently unqualified. It reads BF16 projected QKV,
retains existing FP32 conv history, accumulates conv/SiLU locally in FP32,
rounds the activation to BF16, then normalizes Q/K with FP32 local reductions
and writes BF16 directly. It never materializes a full FP32 activation tensor.
The producer fixture checks a separate CPU formula (two BF16 ULP or absolute
1e-7 for its bounded synthetic inputs), exact history updates and split-call
invariance, and direct BF16 recurrence with aligned conv/carry handoff. This
diagnostic bound is not a general framework/model-quality tolerance. Projection
GEMM rounding, full-model dispatch, memory budgeting and quality remain separate
integration gates. CMake compiles the module but the model does not dispatch it.

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
