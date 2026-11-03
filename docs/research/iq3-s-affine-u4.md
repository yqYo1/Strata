# IQ3_S affine-U4: AVX2 feasibility

**Classification:** decision support. **Verdict:** conditionally feasible: the affine mapping is exact, and per-four-value-lane correction can preserve the fork's integer and FP32/FMA schedule. A scalar correction cannot. Practical low-NT benefit is unknown; this is not a speed or bottleneck claim.

Pinned scope: native recipe `479a6fdac8cbaa35a9bf3a41d2b6e23be5260e54`, engine `552b1616d21ad1b9aa6a317cb940d375c003246b896a255a935e09292804c96a`, source `af378e88d` (AGENTS unchanged). Local `iq_avx2.cpp` and `iq_codebooks.hpp` match supplied SHA-256 pins. GGML reference is `3cf03257f219afbe7334045ff7c6a06ac68c627d`.

## Mapping and representation

All 512 `iq3_s_grid` words contain four odd magnitudes in `{1,3,5,7,9,11,13,15}` (`include/strata/artifact/iq_codebooks.hpp:1039-1104`); dequantization establishes eight 32-value groups with per-group scales (`:1106-1129`). The fork decoder maps group `j*2+half` to low grid indices at `src/kernels/cpu/iq_avx2.cpp:164-177`: 8 low bytes at `2+16*j+8*half`, high-index bits at `66+2*j+half`, that half's 32 sign bits at `74+8*j`, and scale `1+2*nibble` from low/high nibble of `106+j`. `sgn_vec` maps set sign bits to negative (`:17-28`). For magnitude `m`, encode `q=(15-m)/2` if sign is set, else `(15+m)/2`; then `q∈[0,15]` and signed odd level `w=2q-15` exactly. Preserve this byte, bit and group order.

The 128 packed U4 bytes replace `qs[64]+qh[8]+signs[32]`; retaining `d[2]+scales[4]` gives 134 B/block vs 110 (+24 B, 21.8%). Pack the earlier `q` in each byte's low nibble. With `d` first, U4 starts at offset 2 and 134-byte stride is not 32-byte aligned: use unaligned loads; padding or retaining both copies costs more. Entry points pass a blob pointer (`src/kernels/cpu/native_expert.cpp:175-204`); no transformed-buffer owner or conversion/freeing policy is present.

## Exact dot and schedule

For each existing four-value int32 lane compute

`p = s * (2 * Σ(q[i] a[i]) - 15 * Σ(a[i]))`,

where `s∈{1,3,...,31}` is the IQ3_S group scale and `a` is Q8_K `qs`. AVX2 can unpack nibbles, use unsigned-U8 × signed-I8 `VPMADDUBSW` for pairwise `q*a`, then `VPMADDWD` to form four-value dot lanes. Form activation-pair sums separately and apply `VPMADDWD` with `15*s` (max 465, signed-int16); subtract this lane correction from twice the scaled dot. This adds nibble unpack and a sum/correction path per 32 values, offset by removed sign decode. The [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) documents pairwise saturation for `VPMADDUBSW` and adjacent-word sums for `VPMADDWD`.

Q8_K's quantizer uses `iscale=-127/max`, `nearest_int`, and clamps the upper side at 127, giving valid `qs∈[-127,127]` (GGML `ggml-quants.c:2593-2628`; local contract `include/strata/kernels/quantize_act.hpp:39-51`). Thus each U4/Q8 pair sum is bounded by `2*15*127=3810`, below int16 saturation; a four-value weighted lane is bounded by `4*15*127*31=236220`, within int32. `block_q8_K.bsums` contains 16-element sums (`third_party/ggml/ggml-common.h:373-375`; GGML quantizer `:2619-2624`), not the eight four-value sums needed per 32-value half. It can correct an aggregate scalar, but not preserve each lane. Since lane values are independently converted to FP32, a whole-block correction can change rounding and the later FMA result.

The fork adds each corrected 32-value contribution into `acci`, then converts the eight lanes once per Q8_K 256-value block and FMA-accumulates with `h2f(d)*q8.d` (`iq_avx2.cpp:203-226`). Exact per-lane correction before those same adds preserves this integer-to-float/FMA boundary. Float correction or correction after lane reduction does not. The pinned GGML AVX2 IQ3_S implementation confirms odd scale multiplication and integer-to-FP32 accumulation (`[quants.c:3095-3135](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/src/ggml-cpu/arch/x86/quants.c#L3095-L3135)`); its accumulator grouping differs from the fork.

## Low-NT alternatives and counterexamples

- **Native IQ3_S, 110 B:** no conversion/expansion. The fork already shares grid and sign decode across NT tokens (`iq_avx2.cpp:202-220`); likely the strongest baseline at NT=1, but not measured here.
- **Expanded U4, 134 B:** removes per-use codebook lookup/sign expansion, but adds 24 B to every weight block read. Persistent conversion can amortize setup; transient conversion may pay repeatedly. Grouped NT can share each row's weight decode; NT=1 cannot. Dispatch supports NT=1..8 (`iq_avx2.cpp:348-375`; caller `native_expert.cpp:183-204`), but source says nothing about actual NT distribution, reuse, cache/DRAM residency or CPU/RAM bottleneck.
- **Magnitude U4 + sign bits, 166 B:** keeps sign handling. **Signed int8, 262 B:** avoids nibble unpack but has much larger weight traffic. Neither wins by format alone; include setup and bytes served.

Falsifiers: a grid outside the eight odd levels; any sign/index/scale mapping mismatch; failure to match each original four-value `acci` lane over metadata and Q8_K boundary cases; or no viable owner/lifetime for the expanded bytes. For the speed hypothesis, a future setup-inclusive matched CPU comparison with no repeatable gain at relevant NT, or whose working-set cost erases saved decode work, rejects adoption without refuting the algebra.

## Minimum future owned discriminator (not run)

Main pins source, conversion owner and recipe; compares native 110 B with pre-expanded 134 B on identical rows/activations at NT=1 and available grouped NT>=2, retaining current dispatch. Include conversion read/write and a named reuse horizon as well as repeated warm service; separate requested bytes from measured service. Before timing, exhaustively check all 512 grids/sign cases and scale nibbles plus Q8_K `[-127,127]`/zero, comparing all eight four-value lanes and post-block FP32/FMA results to the native schedule. Scalar-total equality is insufficient. This can decide arithmetic and setup-inclusive CPU utility, not model quality, bottleneck or route qualification.
