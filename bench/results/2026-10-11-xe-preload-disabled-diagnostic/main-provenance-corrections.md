# Main review of provenance and superseded conclusions

Original returned reports are preserved byte-identically; this note does not
edit their statements or hashes.

`research-xe-production-mma-minimal-probe-eligibility-next-v1.txt` links to four
`MistVVK/XeStrata/blob/fa3e20e842fbed6986a69bade8afe3edabedee07/…` pages. That revision
is the local candidate published in the user's fork, not a claimed MistVVK
upstream revision. The corresponding owner is `yqYo1/Strata`:

- https://github.com/yqYo1/Strata/blob/fa3e20e842fbed6986a69bade8afe3edabedee07/src/kernels/xe/mma_gemm.cpp
- https://github.com/yqYo1/Strata/blob/fa3e20e842fbed6986a69bade8afe3edabedee07/tools/test_prefill_dense_gemm.cpp
- https://github.com/yqYo1/Strata/blob/fa3e20e842fbed6986a69bade8afe3edabedee07/src/core/device.cpp
- https://github.com/yqYo1/Strata/blob/fa3e20e842fbed6986a69bade8afe3edabedee07/CMakeLists.txt

The mapped-provider report separates a BiF symbol present in a library from the
actual bundle selected by compilation. No source string search establishes the
root cause or a runtime-install recommendation.

The N64 API-volume and original group-join reports refer to an older standalone
component qualification state. The v2 model-width addendum corrects that scope:
the same IQ kernel source had a qualified 18-format byte-comparison experiment
and seven paired n=16 component measurements. Those resident-input component
curves still do not predict the model's group-width distribution or model latency.

`research-xe-native-dequant-v6-n64-group-census-review-next-v1.txt` was delivered
after its original N64-only review. Its statement that a new 32K ON result is
needed is superseded by the new v6 32K OFF/ON PASS pair in this archive. Its
N64 counts and limits remain valid. The old source census result's 269.159 tok/s
is not a matched reference for the new source/input/diagnostic recipe.

The stream-tail report's 4096 candidate adds two full model passes. Main retains
the subsequent six-pass 6140 feasibility report as a narrower unmeasured candidate;
neither setting has a new measured result. The exact-context arithmetic is not
successful full-262144 inference, and no output can fit after an exactly full
prompt without a separately defined capacity behavior.

The published preload guard changes only source publication identity after the
historical builds. It is not a new build or new model result. Extra mapped
compiler/libc files are hashed provenance; filtering them from the original
core validator is not a claim that those extra files received its qualification.
