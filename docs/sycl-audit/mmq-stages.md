# MMQ gather, activation and connected products

The isolated port now includes all five surrounding operations from pinned
Strata `src/prefill/moe_mmq.cu`: native gather, grouped native gather, Strata Q2
repacking, SwiGLU and identity IDs. This is component evidence, not full-engine
parity or a speed measurement.

## Source correspondence

| Upstream operation | SYCL implementation and retained behavior |
| --- | --- |
| `gather_native` | All five pointers and both sizes select either 16-byte vector copies or single-byte copies. Gate precedes up; down has a separate destination. |
| `gather_native_group` | Same `[first,n)` range, maximum 16 entries, absolute expert slot indices, offsets, strides and alignment rejection. Invalid groups return false and launch nothing. The pointer table is captured by value. |
| `gather_strata_q2` | Same fixed 1280×2560 gate/up and 2560×640 down geometry. Each 64-value block gets its original 2-byte scale followed by 16 code bytes. Gate/up rows remain interleaved. |
| `swiglu` | Same FP32 expression and split/interleaved indexing. `sycl::native::exp` maps CUDA's approximate `__expf` to the Intel operation. Their bit patterns have not been compared on a CUDA device. |
| `iota` | Same int32 identity IDs and positive-length guard. |

The component interfaces use an explicit in-order SYCL queue, device-accessible
pointers and caller-owned storage. They allocate no memory and introduce no host
waits. Event-returning operations expose completion; grouped gather retains the
upstream boolean admission contract. The [public adapter and context](mmq-context.md) now provide the original
entry-point symbols with an explicit SYCL queue contract; the full runtime is
still separate work.

## Validation

Intel Arc B570, Linux, oneAPI 2026.1.1, Level Zero, 2026-10-04:

- Native gather checks the aligned path, each pointer independently misaligned,
  and non-multiple-of-16 sizes, including leading/trailing destination guards.
- Grouped gather starts at expert 3 of 16, uses matrix gaps and destination
  strides, and checks untouched earlier experts and padding. Twelve rejected
  cases independently cover invalid ranges and every alignment field/pointer;
  both outputs remain untouched. A valid group is also accepted afterward.
- Strata Q2 repacking compares every output byte, including scales, codes and
  destination guards, against host byte indexing into the original planes.
- Both SwiGLU layouts use three rows of width 257, with gate values from -120 to
  120. References use double `exp`; the bound is `3e-6 * abs(reference) + 1e-6`.
- Iota checks lengths 0, 1, 255, 256, 257 and 4007, including output guards.
- Ten connected cases queue gather → iota → quantize → gate/up product → SwiGLU
  → quantize → down product without any intermediate host synchronization.
  The nine default native weight formats use width 512, seven tokens, three
  experts including an empty expert, input row selection, output permutation and
  padded strides. A tenth case uses the actual Strata Q2 packed geometry with
  one token. Both products use normal stream-K planning (160 groups here) and
  reuse one scratch buffer in queue order.
- Every active output of both products is checked using the pinned upstream CPU
  weight dequantizer and a double dot product. The inputs to each reference are
  the actual quantized intermediate tensors; this validates each stage, not an
  independently rounded whole-chain CPU or CUDA inference implementation.
  The quantizer itself retains its separate 546-case byte-oracle test.
- Each connected SwiGLU output is checked against the same double reference and
  tolerance. Final row padding and intermediate/final tail guards are checked.

JIT and `bmg_g21` AOT pass the new test and existing quantizer/product checks.
The new test performs 1,410,304 explicit byte comparisons and 135,046 numerical
comparisons. Maximum observed product error divided by the sum of absolute
reference products is `1.17884e-7` (bound `3e-6 * L1 + 1e-6`). Maximum observed
SwiGLU absolute error is `0.000973148`; all values satisfy the relative-plus-absolute
bound above, including larger synthetic activations.

Build/source hashes and test logs are in [mmq-stages-results.json](mmq-stages-results.json).
Existing product and exact fixup-order checks remain unchanged. These tests do not
establish CUDA fast-math/FTZ agreement, NaN/Inf parity, routing/top-k equivalence,
full layer/session behavior, or full-model performance. Optional MMQ formats and
the complete runtime/engine integration remain open audit items.
