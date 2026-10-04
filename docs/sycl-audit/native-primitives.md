# Original native state and attention primitives

Twelve complete pinned CUDA files now build as native SYCL components: GR
normalization/postops, MoE combine, GDN recurrence/preprocessing, QSA norm/gate,
router, RoPE, PLE postops, short-context attention, QSA indexer and score.
Original source files and public headers remain unchanged. The source inventory
contains 29 kernel definitions and 41 launch sites. The original portable F32
score branch executes on SYCL; the NVIDIA PTX tensor score branch remains
inactive and unbound. This closes these component bindings, not full engine
execution or whole-model state parity.

Both JIT and B570 AOT builds pass 21 related CTest programs, including the
previous complete GEMM/product/group regressions. The new test covers 944 cases
and 649 graph replays per build, with 9,107,948 numerical, 4,481,680 bit and
81,392 guard comparisons. Full logs, source/object/binary hashes, loader evidence
and earlier failures are in [native-primitives-results.json](native-primitives-results.json).
All 256 original files checked against the pinned inventory remain unchanged.
No new PP/TG measurement has been established.

## Native execution and capture

`tools/sycl/lower_cuda_primitives.py` checks the pinned hashes and retains each
whole original translation unit. The existing product lowerer now accepts the
multiple static F32 shared arrays used by indexer and attention kernels. Each
array receives its own offset in one local accessor; the accessor has a
16-byte-aligned base. Unsupported shared declarations or later explicit aligned
fields fail the lowering. The original workgroup dimensions, subgroup XOR trees,
array extents, templates, guards and host format choices remain present.

Launch arguments now evaluate on the host before kernel submission. The
previous lowering placed argument expressions inside the device lambda, which
made the original `mrope_table()` host registry getter a device call. Captures
now retain argument values and pointers; pointee contents remain live on replay.
A RoPE graph test changes the registered pointer after capture and verifies
that the graph continues to use the captured pointer with its updated contents.
The same correction regenerates all four earlier product files and their
regressions pass in both builds.

The router's original seven unused warps exit before its one `__syncthreads()`.
Only warp zero remains and communicates through registers. SYCL requires a
workgroup barrier to be reached by the whole workgroup. The lowering therefore
uses a subgroup barrier at this one site. Integer shuffles preserve router IDs;
float shuffles retain the original reduction trees. No CUDA architecture or
compiler macro is invented to expose device-only helpers: a generated overlay
of `mrope.hpp` adds an explicit SYCL branch around their unchanged bodies.

## Numerical differences found during validation

The first indexer reference used FP64 powers and positions throughout the
analytic angle. That missed the original F32 power/angle boundaries. Restoring
those boundaries in the independent CPU reference did not remove the failure:
mode None, position 27, pooled row 6/channel 1 still returned `0.162851` versus
`0.162861` within the unchanged numerical gate.

A separate GPU arithmetic diagnostic with base `0x1.7ff222p-1`, exponent 1 and
position 40 returned a plain power one or two ULPs above the base in different
compiled diagnostic variants. Its angle and trigonometry changed accordingly.
Intel IMF high-accuracy power returned the base and F32 angle `0x1.dfeeaap+4`,
matching the CPU operands. This diagnostic measured the power expression, not
the indexer's complete reduction. Binding the same original F32 operands to
[IMF high-accuracy power](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2026-0/imf-device-library-power-functions.html)
at all three analytic device sites then passed the original indexer/RoPE
checks. The independent reference still evaluates trigonometry in FP64 after
the F32 angle boundary. Its tolerance was not increased.

PLE single and batch calls initially disagreed by one ULP in normalized keys:
`-0x1.0b3a24p+0` versus `-0x1.0b3a26p+0`. Their mean divisions use a runtime
width and a constant width respectively. Both original mean sites now use
explicit F32 RN division with the same operands. The unchanged batch/sequential
bit comparison then passes for keys, gates, gated values, normalized rows,
residual results and final history. This is evidence from changing those two
sites; no diagnostic exported the original mean intermediates.

The first indexer diagnostic used the selected 26.35 driver. The later power
and PLE debugging runs used the system-default driver; those logs retain this
distinction. Final 21-program suites explicitly select the 26.35 driver and
disable persistent SYCL caching. A new JIT loader trace verifies the selected
GPU driver, FCL, IGC, GMM and SYCL library paths; their hashes are recorded.
Original CUDA builds use `--use_fast_math`. These SYCL bindings use
`-fno-fast-math -ffp-contract=fast` and explicit RN/high-accuracy operations at
the sites above. CUDA device bit identity, NVIDIA FTZ and native exponential
identity remain unverified.

## What the new test covers

| Component | Executed checks |
| --- | --- |
| Norm and GR | Widths 1/127/128/256/512/1023/1024/2560 around the 1024-thread branch; full/broadcast gamma, zero rows, exact aliases, fused/unfused gating, scaled SiLU and residuals |
| MoE/router | All 1..15 expert counts, optional shared expert, single/multi bit comparisons; 512 logits/top 10, ties, permuted IDs and negative logits |
| QSA gate | Three geometries including 24 heads of 256; query/gate half selection, extreme sigmoid inputs and in-place/separate outputs in changed-input graphs |
| GDN | Three head shapes including the model's 16 key/48 value heads, width 128; five connected state/history updates per shape in a seven-kernel graph |
| RoPE | None/Linear/YaRN, widths 128/256, M-RoPE on/off, registered angle table on/off, aliases and captured old registry pointer with changed device contents |
| Indexer/score | Twelve scaling/table/M-RoPE combinations, 48 chronological cells each; FP16-rounded tail, spare, completed pool/metadata, every batch helper, nonaligned chunks, score ReLU/bias and exact incomplete-tail bias |
| Attention | Original Q24x256/KV2x256 shape, widths 1/3/4/127/128/129/255/256, optional F16 additive mask, poisoned inactive KV/mask cells, changed-step graphs and invalid-step status/output poison |
| PLE | Original 2560x4 shape and nine-row history, all query-normalized/hidden-result alias combinations, changed-input six-kernel graphs, batches 1/3/10 in seven-kernel graphs versus sequential stored GPU bits |

GDN's CPU oracle evaluates the delta recurrence and readout in FP64, with F32
stores at the original state boundaries. It checks every persistent state
element, modulo head mapping, query scaling, chronological convolution history,
softplus threshold and closing RMS/gamma/gate. The shape comes from the original
layout; all weights and input values in these tests are synthetic.

Indexer input rounding and attention/PLE F16 weights use actual pinned GGML CPU
conversion routines. Score and PLE stage references use actual stored GPU inputs
to distinguish one stage's arithmetic from upstream error. Independent FP64
products, RMS, sigmoid, softmax and convolution references do not execute the
translated GPU body. Each graph checks its actual kernel-node count and verifies
that instantiation leaves outputs/state untouched. Fixture uploads synchronize
to preserve temporary CPU-vector lifetimes; this test makes no asynchronous
upload or throughput claim.

The numerical gate is `3e-5 * max(abs(reference), supplied_L1) + 3e-6`; the
reported maximum is `error / (max(abs(reference), supplied_L1) + 1)` and reaches
`4.41821e-7` in both final builds. This metric is not an L1-only relative error.
All batch/sequential comparisons described as bit comparisons remain exact.

## Reproduce and remaining work

On Intel Arc B570 / Ryzen 5 5600X, oneAPI 2026.1.1 and the recorded driver paths:

```sh
source /opt/intel/oneapi/setvars.sh
export LD_LIBRARY_PATH=/home/yayoi/.local/opt/strata-compute-runtime-26.35.39758.10/usr/lib/x86_64-linux-gnu:/home/yayoi/.local/opt/strata-compute-runtime-26.35.39758.10/usr/local/lib:$LD_LIBRARY_PATH
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_CACHE_PERSISTENT=0
cmake -S tests/sycl_upstream -B build-upstream-sycl \
  -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx \
  -DSTRATA_GGML_DIR=/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned
cmake --build build-upstream-sycl -j3
ctest --test-dir build-upstream-sycl -V \
  -R 'original_|mmq_public_stages|mmq_context_lifetime|runtime_'
```

Use a separate build directory with `-DSTRATA_SYCL_DEVICE_ARCH=bmg_g21` for AOT.
The final suites took 61.94 seconds JIT and 10.52 seconds AOT. Those durations
include compilation/startup and correctness checks; they are not inference
benchmarks. AOT warnings still report spills in some original product kernels.

Remaining work includes native program/device metadata, the other original
kernel families, the NVIDIA tensor score algorithm's native binding, fixed BLAS
workspace binding, CPU ISA bindings, full engine linking and whole-model state
validation. The host syntax census remains the earlier 19/20 checkpoint; these
CUDA components are not extra passes in that census. The unchanged standalone
large MMQ and historical real-model dequant sample tests were not repeated in
this phase. Windows, other devices and multi-GPU execution remain untested.
PP1000/TG70 remain unachieved.
