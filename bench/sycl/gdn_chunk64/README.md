# Private GDN chunk64 candidate

Current unfinished Main implementation; repository MIT license applies. This
is independently authored SYCL code, not an external kernel port or adoption.
It preserves the loaded16/48head order, uses BF16 activation/scratch, FP32 carry,
a full64 FP32 inverse rounded once and genuine FP16 output for the existing
consumer. The schedule is not the SGLang block16/XPU numerical contract.

The previous compile stopped because fixture.cpp lacked the runtime.hpp include;
that include is now explicit. Host DMA buffers remain owned through failure-path
drain. An unused BF16 V copy was removed, saving 96 MiB of scratch at 8192 tokens
(workspace now 1,024,983,040 bytes). No GPU test, general numerical qualification,
speed, model quality, full262144 or RESTORE result exists yet. Run the bounded
dyadic smoke before general chunk-boundary and framework-contract comparisons.
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
```

The helper uses the shared measurement lock, bounded process supervision,
immutable validation/tracing environment and qualified runtime ELF identities.
It refuses to overwrite a previous candidate or reuse a changed source tree.
Current compact evidence goes to `bench/results/2026-10-15-pr6-recovery-audit/`
as `gdn-chunk64-current.json`; temporary full logs stay outside Git.
