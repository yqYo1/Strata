# CPU low-bit MoE methods: curated retrieval note

**Use:** retrieve only if a Main-owned CPU profile or calibration makes codebook/dequant work a live candidate. This is method evidence, not a Zen 3 speed result. Source context is Strata `dfb3b73344f7f9cb3f7494268e33841179702a44`; actual shapes are H=2560, FF=640, NT=1..4.

## Findings that transfer

- The model's IQ GU weights use nonuniform codebooks, signs and format-specific scale metadata. Treating the packed 2/3-bit fields as ordinary signed integers changes the represented weights. A bit-exact LUT/decomposition is conceivable only if it reproduces the actual decoded values and preserves activation quantization, scale application, and accumulation/rounding.
- T-MAC's binary-plane subset tables and `pshufb`/`tbl` lookups are a design contrast, not a drop-in IQ kernel. Its reported TM-base reached at most 17% slower than llama.cpp before its layout/table optimizations on the paper's M2 Ultra cases; the device, formats, and matrices do not predict this model. The paper also reports register/table pressure and numerical effects from table quantization.
- LUT-GEMM shows that a chosen nonuniform quantizer can be represented through binary coding, but its published implementation is GPU-oriented. It does not show a cheap exact transform for existing ggml IQ blocks or a Zen 3 benefit.
- QIGen's paper supports shape-aware AVX2 kernel generation for GPTQ-like LLaMA/OPT inference. The currently visible repository root has no license file; use its paper as a method reference and do not reuse repository code unless a license is resolved.
- Exact predecode/reordering is a possible lower-risk alternative only if it preserves each decoded weight. Expanded bytes, load-time conversion, cache pollution and replacement cost must be included; int8 expansion can multiply packed weight storage and may hurt the 32 MiB shared L3.

## Applicability and counterevidence

The seven observed GU/Down layer pairs are IQ3_XXS/IQ4_NL (14), IQ2_S/IQ4_NL (15), IQ3_S/IQ4_NL (9), IQ2_S/Q2_0 (5), IQ3_XXS/Q2_0 (3), IQ3_S/Q2_0 (1), and IQ4_XS/IQ4_NL (1). These are layer counts, not routed job counts. The existing source already vectorizes the production Q2_0 Down42 route. Earlier gather/index/sign-expansion prototypes on IQ3_XXS/IQ3_S matched outputs but did not consistently win for current 2–4-token groups. Neither the six-process comparison nor a component kernel establishes that CPU math, cache misses, or RAM bandwidth limits model decode.

Before testing an exact method, use real expert blocks and production H/FF/group shapes, compare each GU family and both Down formats at NT=1..4, and check finite/canary-safe outputs against the existing contract outside timing. Keep a direct kernel result separate from the pool path; only a component pass can motivate a frozen production-entrypoint comparison.

## Provenance and sources

No external code, helper, fixture, or implementation was copied. T-MAC's root project declares MIT, but its submodules/dependencies and derived llama.cpp files need per-file review before any code reference. The LUT-GEMM repository declares Apache-2.0; its paper is the relevant method reference here. ggml format links below are mutable; consult the exact dependency revision before relying on byte-layout details. The existing local `iq_single_avx2.cpp` states that it adapts ggml AVX2 code and points to `third_party/ggml/LICENSE` (MIT); this is known provenance for that local helper, not permission to transplant other files.

- ggml format definitions: [quantization code](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c), [common definitions](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h)
- [T-MAC paper](https://arxiv.org/abs/2407.00088) and [official repository](https://github.com/microsoft/T-MAC)
- [LUT-GEMM paper](https://arxiv.org/abs/2206.09557) and [official repository](https://github.com/naver-aics/lut-gemm)
- [QIGen paper](https://arxiv.org/abs/2307.03738) and [repository](https://github.com/IST-DASLab/QIGen)

Retrieval tags: `zen3-avx2-lowbit`, `iq-nonlinear-codebook`, `exact-lut`, `predecode-cost`, `no-model-speed-claim`, `license-review-before-code`.
