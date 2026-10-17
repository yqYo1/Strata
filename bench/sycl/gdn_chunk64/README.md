# Private GDN chunk64 candidate

Current unfinished Main implementation; repository MIT license applies. This
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
No general numerical qualification, speed, model quality, full262144 or RESTORE
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
```

The helper uses the shared measurement lock, bounded process supervision,
immutable validation/tracing environment and qualified runtime ELF identities.
It refuses to overwrite a previous candidate or reuse a changed source tree.
Current compact evidence goes to `bench/results/2026-10-15-pr6-recovery-audit/`
as `gdn-chunk64-current.json`; temporary full logs stay outside Git.
