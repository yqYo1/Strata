# Private GDN chunk64 candidate

Current unfinished Main implementation; repository MIT license applies. This
is independently authored SYCL code, not an external kernel port or adoption.
It preserves the loaded16/48head order, uses BF16 activation/scratch, FP32 carry,
a full64 FP32 inverse rounded once and genuine FP16 output for the existing
consumer. The schedule is not the SGLang block16/XPU numerical contract.

The latest compile stopped because fixture.cpp lacks the runtime.hpp include.
No GPU test, general numerical qualification, speed, model quality, full262144
or RESTORE result exists. Fix/build this source, then run the bounded dyadic
smoke before general chunk-boundary and actual-framework-contract comparisons.
The earlier raw-pointer matrix API error is fixed by explicit local/global
multi_ptr arguments. Superseded private source copies and build transcripts are
not required to continue this implementation.

Use the maintained measurement owner/guards in tools/sycl and set the shared
first-check environment before exec. Main serializes all builds/tests/GPU jobs.
See the retained GDN contract and current tuning documentation for the next
comparison; do not infer persistent framework pool dtype from track_state.
