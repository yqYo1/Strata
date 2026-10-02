# Battlemage operations available through SYCL

Investigated on 2026-10-02 for this PC's Intel Arc B570 (`8086:e20c`,
Battlemage G21 / Xe2). This catalog includes operations that Strata is unlikely
to use. It is an implementation reference, not a performance ranking.

## Scope and evidence

“Every operation” here means the finite API surface of SYCL 2020, the installed
Intel compiler's extensions, and the numerical/algorithm libraries described
below. Arbitrary C++ kernels can compose these into indefinitely many algorithms;
there is no finite list of all possible programs or third-party SYCL libraries.
Overloads differing only in vector width, address space, or pointer/buffer form
are grouped. Restrictions on types, shapes and execution models are retained.

The coverage has four layers:

1. The operation-family tables in this document, including unsupported paths.
2. [Every named upstream Intel extension specification](sycl/EXTENSION_INDEX.md),
   including proposals, retired extensions and facilities for other devices.
3. An [installed Intel function-name index](sycl/INTEL_API_NAMES.md), including
   the large Intel math interface and experimental ESIMD functions. This is a
   declaration index, not proof that every entry runs on B570.
4. [Header paths and SHA256 hashes](sycl/INSTALLED_HEADERS.tsv),
   [compiler feature macros](sycl/FEATURE_MACROS.json), and the complete
   [device capability query](sycl/B570_CAPABILITIES.txt). These make version
   changes and omissions auditable without copying vendor headers into Strata.

Evidence labels used below:

| Label | Meaning |
| --- | --- |
| **V** | The stated small case executed and matched a reference on this B570 |
| **Q** | This driver/device reports the capability; execution of all cases is not established |
| **D** | Defined by the standard or documented for the relevant Intel GPU path; exact workload still needs validation |
| **H** | Declaration or experimental specification found; B570 execution not established |
| **N** | Unavailable through the specified interface in the inspected configuration, or explicitly excluded by its support table |

These labels are deliberately independent. A feature macro, an aspect, a
successful compile, native hardware support and competitive throughput are
five different observations. **V never means all inputs, layouts or overloads
passed a conformance test.** No timings or theoretical speed ratios are inferred
from the correctness samples.

### Version boundary and local checks

| Component | Inspected configuration |
| --- | --- |
| OS/kernel | Ubuntu 24.04.5; after reboot, `7.0.0-38-generic` |
| Compiler | Intel oneAPI DPC++/C++ 2026.1.1 |
| Backend | `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`; normal driver settings |
| Driver | `1.17.39395+14`, package `26.31.39395.14-1~24.04~ppa1` |
| Libraries whose headers were indexed | oneMKL 2026.1; oneDPL 2022.13; oneDNN 2026.0 |
| Other installed libraries inspected | oneDAL 2026.1; oneCCL 2022.1 |
| Compiler specification reference | Intel LLVM `16fff99e152f8485cda1905fcd9bbd18b347849c` |
| Execution limits reported | Subgroups 16/32; workgroup at most 1,024 work-items; local memory 131,072 bytes |

The installed headers determine the usable spelling, not a newer upstream
proposal. B570 observations must not be silently extended to BMG G31 or future
architectures. Source specifications sometimes group them together; driver
recognition and available combinations still need to be queried on that device.

The following checks passed after the reboot. Inputs to the DPAS checks vary
by row, column and reduction coordinate; they are not all-one matrices.

| Check | Result / limited coverage |
| --- | --- |
| Scalar FP64 FMA/sqrt, FP32 FMA, FP16 multiply, integer popcount/mul_hi | Expected values matched |
| Subgroup reduction | 16 lanes, integer sum 0 through 15 = 120 |
| ESIMD unsigned INT8 DPAS | M=4, N=16, K=32, initial accumulator 3; 64 outputs matched CPU |
| ESIMD unsigned INT4 DPAS | M=4, N=16, K=64, packed operands; 64 outputs matched CPU |
| ESIMD unsigned INT2 DPAS | M=4, N=16, K=64, packed operands; 64 outputs matched CPU |
| oneMKL FP32 GEMM | Row-major 2x2 product matched CPU |
| oneDPL sort | Four device-accessible integers sorted correctly |

Sources: [operations probe](../tools/sycl/operations_probe.cpp),
[library probe](../tools/sycl/library_probe.cpp), and
[recorded output](sycl/B570_OPERATIONS_PROBE.txt).
The earlier [matrix probe](../tools/sycl/matrix_probe.cpp) also tests one FP16
8x16x16 tile with FP32 accumulation; its constant-input test is weaker than a
layout test with varying data. Graph replay and CPU/GPU handoff observations
remain in [SYCL_RESEARCH.md](SYCL_RESEARCH.md) and
[SYCL_USM_ATOMICS.md](SYCL_USM_ATOMICS.md).

## 1. Execution models and arithmetic representations

Intel's [B580/B570 architecture presentation](https://download.intel.com/newsroom/2024/client-computing/Intel-Arc-B580-B570-Media-Deck.pdf)
identifies SIMD16/32 vector execution, FP64, and XMX INT2/INT4/INT8/FP16/BF16/TF32.
This describes hardware, not a promise that every SYCL abstraction exposes every
format. In particular, XMX INT2/INT4 are reachable through ESIMD DPAS even though
this runtime's `joint_matrix` combination query does not list them.

| Model | Entry points and operation shape | B570 status / limits |
| --- | --- | --- |
| Ordinary SYCL SPMD | `single_task`, `parallel_for(range)`, `parallel_for(nd_range)`; 1D/2D/3D indexing, scalar and vector C++ expressions | D/V samples; compiler maps work-items to SIMD lanes |
| Hierarchical parallelism | `parallel_for_work_group`, `parallel_for_work_item`, `h_item`, `private_memory` | D legacy SYCL model; not the model used by the probes |
| Explicit SIMD | `SYCL_ESIMD_KERNEL`, `SYCL_ESIMD_FUNCTION`, `simd<T,N>` and `simd_view` | Q/V DPAS; an ESIMD work-item controls a vector, with subgroup size 1 in this execution model |
| Mixed SPMD/ESIMD | `invoke_simd`, uniform arguments and explicit vector results | H/D extension; requires its calling convention and uniformity rules; not arbitrary calls between the two models |
| Cooperative subgroup matrices | `joint_matrix` and collective matrix functions | Q/V FP16; lane ownership and legal tiles are API-defined |
| Host-launched library operation | oneMKL, oneDPL, oneDNN SYCL queues/streams | D/V selected libraries; may launch many kernels and need workspace |
| Device-callable library operation | oneMKL device RNG/VM subset, oneDPL kernel API, Embree traversal | Separate device APIs; host-launched GEMM cannot simply be called inside a kernel |
| Runtime-generated kernels | Kernel bundles; experimental kernel compiler for SYCL/OpenCL C/SPIR-V | Q online compiler/linker; source-language support and installed tools must be checked separately |

`vec<T,N>` is not a subgroup. Its supported standard element counts are
1, 2, 3, 4, 8 and 16; `marray<T,N>` has a compile-time array length. Neither alone
guarantees a particular hardware instruction or coalesced memory transaction.
A `simd<T,N>` length is also not a workgroup size.

| Representation | Operations / conversions | Evidence and important distinction |
| --- | --- | --- |
| `bool`; signed/unsigned 8/16/32/64-bit integers | Arithmetic, comparison, bitwise logic, shifts, loads/stores, supported built-ins | D; integer divisions and wide operations may lower to multiple instructions |
| `sycl::half` / FP16 | Scalar/vector arithmetic and math; FP16-to-FP32 conversion; XMX operands | Q/V small multiply and earlier matrix case |
| `float` / FP32 | Full standard math surface, FMA, vector operations | D/V samples; ordinary FP32 and TF32 matrix math are not numerically identical |
| `double` / FP64 | Arithmetic, FMA, sqrt and standard FP64 functions | Q/V small FMA/sqrt; hardware presentation also lists FP64. No native throughput ratio measured |
| `ext::oneapi::bfloat16` | Storage, conversions, operators, experimental BF16 math and matrix input | D/Q matrix; scalar operations can widen to FP32. Not every BF16 operation is a native BF16 ALU instruction |
| `matrix::precision::tf32` | FP32-backed matrix operand with reduced mantissa; `round_to_tf32`; ESIMD experimental `tfloat32` | Q matrix combinations; rounding must be specified/tested for accuracy |
| Packed signed/unsigned INT4/INT2 | Pack/unpack with bit operations; ESIMD DPAS precisions `s4/u4/s2/u2` | D; V unsigned same-width DPAS only. There is no ordinary standard `sycl::int4_t` scalar |
| Packed INT1 | Bitwise/popcount implementation of binary math | D composed operations; ESIMD `u1/s1` precision enumerators are explicitly reserved/unsupported, so N direct DPAS |
| FP8 / FP4 / MX formats | Software storage/conversion and higher-level library formats are separate from native matrix support | N native B570 XMX formats in the inspected hardware/API tables; upstream FP8/FP4 proposals are not enabled by this installed compiler. Do not infer native support from oneDNN data-type enums |
| Complex | Pairs of real values; experimental SYCL complex math, supported C++/oneDPL kernel APIs, library complex BLAS/FFT | D/H by interface; no separate complex hardware ALU is implied |
| 128-bit integers, extended `long double`, arbitrary precision | Can be composed from supported words or libraries | No standard SYCL/B570 native guarantee; compiler-specific spellings need separate compilation/accuracy checks |

The capability log reports denormals, infinities/NaNs, FMA and nearest/zero/
infinity rounding modes for half, float and double; correctly rounded divide/
sqrt is reported for float/double. That is not a guarantee after enabling
fast-math, flush-to-zero, reduced-precision matrix modes or native math.
Signed C++ overflow, invalid shift counts and invalid casts retain their language
constraints; GPU execution does not make them defined.

## 2. Complete standard SYCL built-in operation families

The following names are the SYCL 2020 built-in surface, grouped by operation.
The installed `sycl/detail/builtins/*_functions.inc` files were cross-checked with
Khronos references. Scalar/vector/marray overload restrictions still apply;
`half` and `double` require their aspects, both present here. These are **D**;
only the explicitly listed probe cases are **V**.

### Floating-point math

| Operation | Function names |
| --- | --- |
| Trigonometric | `sin`, `cos`, `tan`, `sincos`, `sinpi`, `cospi`, `tanpi` |
| Inverse trigonometric | `asin`, `acos`, `atan`, `atan2`, `asinpi`, `acospi`, `atanpi`, `atan2pi` |
| Hyperbolic and inverse | `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` |
| Exponentials and logarithms | `exp`, `exp2`, `exp10`, `expm1`, `log`, `log2`, `log10`, `log1p`, `logb`, `ilogb` |
| Powers and roots | `pow`, `pown`, `powr`, `rootn`, `sqrt`, `rsqrt`, `cbrt`, `hypot` |
| Error/gamma functions | `erf`, `erfc`, `tgamma`, `lgamma`, `lgamma_r` |
| Arithmetic and extrema | `fabs`, `fdim`, `fma`, `mad`, `fmax`, `fmin`, `maxmag`, `minmag` |
| Rounding and decomposition | `ceil`, `floor`, `trunc`, `round`, `rint`, `fract`, `modf`, `frexp`, `ldexp` |
| Remainders | `fmod`, `remainder`, `remquo` |
| Representation-sensitive | `copysign`, `nextafter`, `nan` |

Pointer-output functions such as `sincos`, `frexp`, `modf`, `remquo` and
`lgamma_r` have writable-address-space and output-type requirements. `fma` has
a fused rounding contract; `mad` is not an interchangeable accuracy guarantee.
`pow`, `powr` and integer-exponent `pown` also have different domains/contracts.
[Khronos math reference](https://github.khronos.org/SYCL_Reference/iface/math-functions.html).

Both `sycl::native` and `sycl::half_precision` provide:

```text
cos divide exp exp2 exp10 log log2 log10 powr recip rsqrt sin sqrt tan
```

These are reduced/implementation-defined accuracy interfaces as specified for
each namespace. **`half_precision` is not the FP16 type.** Its arguments are
float-based. Native math is not permission to assume IEEE special-value or
full-range behavior without its contract.
[Khronos native math](https://github.khronos.org/SYCL_Reference/iface/native-prec-math-func.html),
[half-precision math](https://github.khronos.org/SYCL_Reference/iface/half-prec-math-func.html).

### Integer, common, geometric and relational functions

| Family | Function names |
| --- | --- |
| Integer magnitude/extrema | `abs`, `abs_diff`, `min`, `max`, `clamp` |
| Saturating and averaged integer arithmetic | `add_sat`, `sub_sat`, `mad_sat`, `hadd`, `rhadd` |
| Integer products | `mul_hi`, `mad_hi`, `mul24`, `mad24` |
| Bit operations | `clz`, `ctz`, `popcount`, `rotate`, `upsample` |
| Common real-valued functions | `clamp`, `degrees`, `radians`, `min`, `max`, `mix`, `step`, `smoothstep`, `sign` |
| Geometric | `dot`, `cross`, `distance`, `length`, `normalize`, `fast_distance`, `fast_length`, `fast_normalize` |
| Floating comparison | `isequal`, `isnotequal`, `isgreater`, `isgreaterequal`, `isless`, `islessequal`, `islessgreater` |
| Floating classification | `isfinite`, `isinf`, `isnan`, `isnormal`, `isordered`, `isunordered`, `signbit` |
| Predication and bit selection | `any`, `all`, `bitselect`, `select` |

`mul24`/`mad24` have 24-bit input constraints; they do not multiply arbitrary
32-bit inputs under the same contract. Vector comparison masks and `select`
use their specified mask representation; a scalar C++ `bool` is not a universal
replacement. Geometric functions have their own allowed scalar/vector shapes.
[Khronos integer reference](https://github.khronos.org/SYCL_Reference/iface/integer-functions.html),
[common functions](https://github.khronos.org/SYCL_Reference/iface/common-functions.html),
[geometric functions](https://github.khronos.org/SYCL_Reference/iface/geometric-functions.html),
[relational functions](https://github.khronos.org/SYCL_Reference/iface/relational-functions.html).

### Language, vector and representation operations

| Family | Operations |
| --- | --- |
| C++ expressions | Unary/binary arithmetic, remainder for integers, comparisons, Boolean logic, bitwise logic, shifts, ternary selection; loops, branches, device functions and templates |
| Vector manipulation | Element access, named components, swizzles, `lo`/`hi`/`even`/`odd`, `shuffle`, `shuffle2`, vector load/store |
| Type conversion | C++ conversions; `vec::convert` with `automatic/rte/rtz/rtp/rtn` rounding as applicable; `vec::as`; `sycl::bit_cast` |
| Pointers | Generic/global/local/private address spaces, `multi_ptr`, `address_space_cast`, pointer arithmetic within valid objects |
| Values and function objects | `marray`, supported device-copyable structs; `plus`, `multiplies`, bit/logical operations, `minimum`, `maximum`, known identities |
| Kernel configuration | Specialization constants, named/free-function kernels, kernel bundles, kernel parameters and dynamic local storage |

Host allocation, exceptions, filesystem/network I/O, arbitrary virtual dispatch,
recursion and the whole host C++ standard library are not automatically device
features. Device virtual functions have a separate extension/aspect (Q here).
Device `printf`/`stream` and assertions are special facilities, discussed below.

## 3. Collectives, reductions, sorting and synchronization

### Standard collectives

| Operation | API family | Evidence / constraints |
| --- | --- | --- |
| Broadcast | `group_broadcast` | D; source lane/id must be valid and agreed on |
| Lane exchange | `shift_group_left`, `shift_group_right`, `permute_group_by_xor`, `select_from_group` | D subgroup operations; do not assume CUDA warp width 32 |
| Predicates | `any_of_group`, `all_of_group`, `none_of_group`; `joint_any_of`, `joint_all_of`, `joint_none_of` | D; the latter operate on a common pointer range |
| Reduction | `reduce_over_group`, `joint_reduce` | D/V 16-lane integer sum; operator/type restrictions and non-associative floating summation matter |
| Scan | `inclusive_scan_over_group`, `exclusive_scan_over_group`, `joint_inclusive_scan`, `joint_exclusive_scan` | D; inclusive/exclusive and initial-value variants |
| Whole-range reduction | `sycl::reduction`, reducer `combine`, supported operators, multiple reducers and identities | D; may use multiple kernel launches; experimental user-defined/reduction properties extend the interface |
| Group rendezvous | `group_barrier`; legacy `nd_item::barrier`, memory fences | D; all participating work-items must reach required collectives in converged control flow |
| Group copies | `async_work_group_copy`, `device_event`, `wait_for` and legacy group/nd_item forms | D; not a promise of a CUDA-style asynchronous copy pipeline or independent progress |

[Khronos group algorithms](https://github.khronos.org/SYCL_Reference/iface/group-algorithms-library.html).
Full-kernel synchronization generally uses dependent kernel submissions.
A normal workgroup barrier does not synchronize different workgroups.

### Extended collectives and group memory

| Extension | Operations and implementation consequence | Evidence |
| --- | --- | --- |
| Subgroup mask | `group_ballot`, `sub_group_mask` Boolean operations, bit tests/sets, count and bit search/extraction/insertion | D; a mask is not a memory fence |
| Older subgroup interface | Subgroup shuffle/up/down/xor, broadcast, reduce/scan, block load/store forms | D compatibility interface; use current group algorithms when equivalent |
| Non-uniform groups | Fixed-size, ballot and opportunistic groups; installed fragment/chunk/tangle facilities | Q for fragment/chunk/tangle; formation/membership and collective rules differ from full subgroups |
| Group load/store | `group_load`, `group_store`; blocked/striped placement and alignment properties | D/H; layout affects lane-to-memory mapping |
| Group sorting | `joint_sort`, `sort_over_group`, `sort_key_value_over_group`; default/radix sorters and scratch memory | D; scratch-space query and key/comparator restrictions apply |
| Workgroup local storage | `group_local_memory`, local accessors, `work_group_memory`, `work_group_static`, `work_group_scratch_memory` | D/H; runtime vs static size and object construction rules differ |
| Private dynamic storage | `private_alloca` extension | Q; per-work-item storage can spill and is not an unlimited heap |
| Root group | Cooperative whole-kernel groups, root barriers and collectives | H; must query kernel/cooperative limits and use the required launch property. Not tested; no global spin-barrier assumption |
| Forward-progress queries | Query execution-scope progress guarantees | D/H; relevant to persistent kernels and producer/consumer designs |
| CUDA cluster / async barrier / masked CUDA shuffles | CUDA-specific extensions | N for this backend; installed CUDA headers/macros do not make them B570 operations |

The [extension index](sycl/EXTENSION_INDEX.md) links each exact specification,
including older/removed proposals which were replaced by standard operations.

### Atomics: operation support is not allocation support

`sycl::atomic_ref<T,Order,Scope,AddressSpace>` provides the following families,
with overloads constrained by `T`:

| Family | Operations |
| --- | --- |
| All supported atomic types | `load`, `store`, `exchange`, `compare_exchange_weak`, `compare_exchange_strong`, lock-free/alignment queries |
| Integer arithmetic | `fetch_add`, `fetch_sub`, increment/decrement, compound add/subtract |
| Integer bitwise | `fetch_and`, `fetch_or`, `fetch_xor` and compound forms |
| Integer and floating extrema | `fetch_min`, `fetch_max` where defined |
| Floating arithmetic | Floating `fetch_add`/`fetch_sub` and associated forms; native implementation is not guaranteed |
| Pointer arithmetic | Atomic pointer add/subtract and increment/decrement where defined |
| Ordering without an RMW | `atomic_fence`; group barriers and events at their respective scopes |

The device reports `atomic64=1` and **all five** SYCL memory orders
(relaxed/acquire/release/acq_rel/seq_cst), and work-item/subgroup/workgroup/device/
system scopes. These are **Q**, not proof of every type/op/address-space
combination being a single hardware instruction. Unsupported native RMWs can be
implemented with compare-exchange loops. Floating atomics can have a distinct
floating-point environment from ordinary expressions.

16-bit atomics are **N**: `ext_oneapi_atomic16=0`; its compiler feature macro is
also zero. Do not reinterpret `half` arithmetic support as FP16 atomic support.

Host-USM and shared-USM **concurrent CPU/GPU atomics remain N** in this normal
configuration. GPU device memory and local-memory atomics are a separate issue.
Ordinary system allocations have a different advertised capability and were
previously tested, but their page-migration cost makes them a separate design
choice. System scope alone does not make every allocation legal. Detailed
allocation/capability distinctions and the failed driver override are in
[SYCL_USM_ATOMICS.md](SYCL_USM_ATOMICS.md). No overrides were used for this catalog.

## 4. XMX matrix arithmetic

### `joint_matrix`

Namespace: `sycl::ext::oneapi::experimental::matrix`.
Headers: `sycl/ext/oneapi/matrix/matrix.hpp` and its Intel extension.

The runtime returned **53 combinations**, including result types which the older
probe did not print separately. The compact table below preserves all of them.
`C` is the input accumulator, `D` the result; the mathematical operation is
`D = A * B + C`.

| A / B | C and D | Reported (M,N,K) shapes | Evidence |
| --- | --- | --- | --- |
| INT8/UINT8, all four signedness pairs | C=D=INT32 | M up to 8, N=16, K=32 | Q; four combinations |
| FP16 / FP16 | Independently FP16 or FP32 for C and D | M up to 8,N=16,K=16; (16,16,16); (1,64,16); (32,64,16); (1,64,32); (32,64,32) | Q; 24 combinations; V only earlier 8x16x16 FP32 accumulator/result case |
| BF16 / BF16 | Independently BF16 or FP32 for C and D | Same six shape families as FP16 | Q; 24 combinations |
| TF32 / TF32 | C=D=FP32 | M up to 8,N=16,K=8 | Q; one combination |

The query's `max_msize` with zero exact `msize` expresses a maximum rather than
one exact tile. Zero maximum fields for other dimensions do not mean zero-sized
matrices: the corresponding exact fields apply. Legal subgroup sizes and
compiler restrictions must also be satisfied; do not form arbitrary combinations
of tile shapes and subgroup sizes. The raw enum mapping in the log is
BF16=0, FP16=1, TF32=2, FP32=3, INT8=5, INT32=7, UINT8=9.

| Matrix operation | API | Notes |
| --- | --- | --- |
| Tile objects | `joint_matrix`, `use::a/b/accumulator`, row/column/packed layout | Distributed across lanes, not an ordinary contiguous per-thread array |
| Load and store | `joint_matrix_load`, `joint_matrix_store` | D; pointer/stride/layout/alignment requirements |
| Initialize | `joint_matrix_fill` | D; uniform participation |
| Multiply and accumulate | `joint_matrix_mad` | Q/V stated case; only queried/documented types and tiles |
| Elementwise epilogue | `joint_matrix_apply` | D; callback over tile elements; Intel overload supplies coordinates |
| Tile copy/conversion | `joint_matrix_copy` | D/H for exact source/destination combination |
| Prefetch | `joint_matrix_prefetch` | D/H; hint, not data availability or completion |
| Bounds-checked tile I/O | Intel `joint_matrix_load_checked`, `joint_matrix_store_checked`, `joint_matrix_fill_checked` | H/Q Intel matrix aspect; installed header exposes these, but the newer specification has additional checked-API query concepts not all present in this compiler |
| Operand-tile stores | Intel extended `joint_matrix_store` | D/H; separate from standard accumulator store |
| TF32 rounding | `round_to_tf32` | D; makes operand rounding explicit |

For the BMG group, the pinned matrix specification requires base pointers
aligned to at least four bytes, byte strides divisible by eight and at most
2^24; offset overloads add row-coordinate restrictions for 8/16-bit elements.
Checked block I/O has further surface, pitch and alignment rules: satisfying
ordinary tile I/O requirements is not enough.
[Matrix specification](https://github.com/intel/llvm/blob/16fff99e152f8485cda1905fcd9bbd18b347849c/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc),
[Intel matrix additions](https://github.com/intel/llvm/blob/16fff99e152f8485cda1905fcd9bbd18b347849c/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_intel_matrix.asciidoc).

FP64 GEMM, full-precision FP32 GEMM, INT4/INT2 matrix inputs and arbitrary
mixed FP16/BF16 inputs are **not** entries in this `joint_matrix` query. They
must not be selected just because some other interface or GPU supports them.
Ordinary FP32/FP64 kernels and library GEMM remain available.

### ESIMD `xmx::dpas`

`sycl::ext::intel::esimd::xmx::dpas` exposes the packed operands directly. It
has forms with and without an input accumulator. DPAS uses the XMX engine;
ESIMD `dp4a` is a different integer-dot API.

| Operand precision | Storage and arithmetic | B570 evidence |
| --- | --- | --- |
| `u8`, `s8` | Packed 8-bit operands; INT32/UINT32 accumulation with legal combinations | D; V u8/u8 |
| `u4`, `s4` | Packed nibbles; integer accumulation | D; V u4/u4 |
| `u2`, `s2` | Packed two-bit values; integer accumulation | D; V u2/u2 |
| Mixed integer precisions/signedness | Explicit A/B precision template parameters; not every pairing is legal | H/D per DPAS restrictions; not covered by the same-width unsigned probes |
| `fp16`, `bf16` | 16-bit matrix operands and documented accumulator/output types | D/Q hardware path; ESIMD floating variants not executed here |
| `tf32` | Reduced-mantissa FP32-backed operands | D/Q hardware path; not executed here |
| `u1`, `s1` | Enumerators marked reserved/unsupported in installed header | N |
| FP8, FP4, FP64 DPAS | No B570 path established by installed XMX API/support tables | N as a selectable B570 DPAS path |
| `dpasw` | Paired-thread wide DPAS entry remains declared | H, **not validated for BMG**. Do not port an older-generation `dpasw` kernel by assuming availability |

Installed DPAS validation requires systolic depth 8 and repeat counts 1 through
8. Its integer precision declarations cover the 6x6 Cartesian product of
`u2/s2/u4/s4/u8/s8` for A and B, subject to target/compiler restrictions;
only three unsigned same-width cases were executed here. For execution size 16,
FP16 operands permit independently FP16/FP32 C and result types; BF16 operands
permit independently BF16/FP32 C and result types; TF32 requires FP32 C/result.
The installed `dpasw` validator permits execution size 8 only, so it cannot
replace the size-16 DPAS used in these B570 probes.
The probes use repeat count 4 and execution size 16. For these same-width integer cases,
`K = 8 * min(32 / bits, 8)`; therefore both INT2 and INT4 use K=64 in these
probes. This does not imply identical throughput.

A and accumulator/result are horizontally represented; B is packed vertically
in the required VNNI form. The callable argument order is **C, B, A** for the
accumulating form. Packing, row/column order, signedness and repeat count must
be validated together. Quantization scales/zero-points and nonlinear codebooks
are separate arithmetic: a stored Q2/Q4 model tensor is not automatically a
valid DPAS INT2/INT4 operand.

Sources: installed `esimd/xmx/{common,dpas}.hpp`,
[Intel DPAS vISA description](https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/instructions/DPAS.md),
and the committed varying-input probe. The vISA description covers multiple
architectures; use its platform restrictions, not merely its encoding enum.

## 5. ESIMD vector, bit, memory and synchronization operations

ESIMD is important beyond matrix multiplication. This section includes the
less likely image/media and raw-message interfaces. **Q** `ext_intel_esimd`
means the execution model is available; it does not validate each low-level
instruction variant. The [name index](sycl/INTEL_API_NAMES.md) contains the full
set of namespace-level function names found in the selected installed headers.

### Vector and mathematical operations

| Family | Installed API names / operations | Evidence and limits |
| --- | --- | --- |
| Register vectors and regions | `simd`, `simd_mask`, `simd_view`, `select`, `bit_cast_view`, element access, region read/write, `merge`, `replicate`, `replicate_vs_w_hs`, `iselect`, `iupdate`, `copy_from`, `copy_to` | D; compile-time region/stride restrictions; views are register regions, not memory buffers |
| Elementwise operators | Arithmetic, integer bitwise/shifts, comparison masks, predicated merge, `convert`, `saturate`, `clamp`, `abs`, `min`, `max` | D; saturation mode and type promotion are explicit |
| Reductions | `reduce`, `hmin`, `hmax`, mask packing/ballot | D; vector reduction inside one ESIMD work-item, not SPMD group reduction |
| Fast hardware math | `inv`, `log2`, `exp2`, `sqrt`, `rsqrt`, `sin`, `cos`, `pow` | D; type and accuracy restrictions differ from standard SYCL |
| IEEE-oriented helpers | `sqrt_ieee`, `div_ieee` | D; specific overloads, not a blanket IEEE mode for the kernel |
| Additional math | `exp`, `log`, `rndd`, `rndu`, `rnde`, `rndz`, `floor`, `ceil`, `trunc` | D; ordinary math may be synthesized |
| Bit counting and search | `cbit`, `fbl`, `fbh`; experimental `popcount`, `clz`, `ctz`, `lzd` | D/H exact variants; zero-input and signedness contracts differ |
| Shifts and rotates | `shl`, `shr`, `lsr`, `asr`, `rol`, `ror` | D; saturating forms where provided |
| Three-input Boolean logic | `bfn` with a compile-time truth-table/control value | D; useful for bit unpacking and Boolean circuits |
| Carry/borrow | `addc`, `subb` | D; explicitly expose carry/borrow for multiword arithmetic |
| Mask/memory compaction | `pack_mask`, `unpack_mask`, `ballot`, `mask_compress_store`, `mask_expand_load` | D/H; lane mask and memory capacity must agree |
| Four-byte integer dot | `dp4a` | D; packed 8-bit products accumulated into 32 bits, signedness/saturation selected by types/mode |
| Experimental bit fields | `bf_reverse`, `bf_extract`, `bf_insert` | H; template width/offset and type constraints |
| Experimental extra math | `imul`, `frc`, `dp4`, `fma`, `fmod`, `frem`, `sin_emu`, `cos_emu`, `sincos`, `acos`, `asin`, `atan`, `atan2`, `atan2_fast`, `tanh`, `tanh_cody_waite` | H; implementation and approximation contracts are not interchangeable with standard functions |
| Stochastic rounding | Experimental `srnd(float-vector, uint16-vector)` to half | H; caller supplies randomness; no statistical quality or B570 execution test here |

Older ESIMD prose mentions facilities such as `plane` that are not present as a
public function in this installed header set. Such prose is not evidence of a
callable installed API. Conversely, the installed name index includes APIs
which a short introductory ESIMD guide does not enumerate.

### Memory, messaging and barriers

| Family | API names / operations | Evidence and restrictions |
| --- | --- | --- |
| Contiguous I/O | `block_load`, `block_store`, scalar load/store; `simd::copy_from/copy_to` | D/V copy operations in the DPAS probe; size/alignment/cache-policy restrictions depend on overload |
| Scatter/gather | `gather`, `scatter`, RGBA variants, masks and pass-through values | D; offsets are generally **bytes**, not typed element indices; verify the chosen overload |
| 2D block operations | `load_2d`, `store_2d`, `prefetch_2d`; transpose and transform/VNNI load variants | D/H exact configuration; surface width/height/pitch encoding, base alignment, block dimensions, element size and padding are coupled |
| Prefetch and cache control | `prefetch`, L1/L2 cache-hint properties; cached/uncached/streaming/read-invalidate or write-back/write-through where allowed | D; load/store/atomic hint combinations differ; hints are not synchronization |
| Local memory | `slm_init`, `slm_allocator`, `slm_block_load/store`, `slm_gather/scatter`, scalar/RGBA variants, local accessor forms | D; must fit device and kernel-specific SLM limits |
| Vector atomics | `atomic_update`, `slm_atomic_update` | D; add/sub/inc/dec, signed/unsigned min/max, exchange/CAS, and/or/xor, float add/sub/min/max/CAS, load/store as supported by type and memory form |
| Fences | `fence`, memory kind and flush/scope selections | D; global/image/local memory; not every enum value is legal for every message/path |
| Workgroup barrier | `barrier` | D; ESIMD thread-group participation differs from a 16-lane SPMD subgroup |
| Named barriers | `named_barrier_init`, `named_barrier_signal`, `named_barrier_wait`; experimental `named_barrier_allocate` | D/H; producer/consumer counts, barrier IDs and signaling protocol must agree |
| Split barriers and dependency waits | Experimental `split_barrier`, `wait` | H; not equivalent to a SYCL event or arbitrary CPU/GPU synchronization |
| Legacy surface/media operations | `get_surface_index`, `media_block_load/store`, RGBA surface operations | H; stateful/image route and architecture/backend restrictions; no media codec acceleration is implied |
| Experimental LSC interfaces | `lsc_block_load/store`, `lsc_gather/scatter`, `lsc_prefetch`, `lsc_load_2d/store_2d/prefetch_2d`, SLM forms and atomic updates | H/D; lower-level alternatives to current stable memory APIs; require valid data-size/order/cache parameters |
| Raw sends | `raw_send`, `raw_sends`, result/no-result forms, predicates and descriptors | H; caller owns hardware message encoding, register layout and target restrictions. Not exercised on the live GPU |
| Assembly/native values | ESIMD inline assembly and native vector/type facilities | H; compiler/vISA/ISA-specific, no universal access to every GPU instruction |
| Hardware queries | `rdtsc`, experimental hardware-thread/subdevice IDs | H; timestamp domain and identifier semantics are not a portable profiling clock |

The installed stable and experimental memory headers are both indexed. APIs
with the same short name in those namespaces are not assumed to have identical
contracts. ESIMD cannot freely call the full SPMD device library, use every
accessor/image form, or use `sycl::stream` as if it were an ordinary kernel.
[ESIMD execution/API specification](https://github.com/intel/llvm/blob/16fff99e152f8485cda1905fcd9bbd18b347849c/sycl/doc/extensions/supported/sycl_ext_intel_esimd/sycl_ext_intel_esimd.md),
[ESIMD memory-operation constraints](https://github.com/intel/llvm/blob/16fff99e152f8485cda1905fcd9bbd18b347849c/sycl/doc/extensions/supported/sycl_ext_intel_esimd/sycl_ext_intel_esimd_functions.md).

## 6. Additional scalar math and conversion interfaces

### Intel device math (`sycl::ext::intel::math`)

Headers: `sycl/ext/intel/math.hpp` and `math/imf_*.hpp`. The installed AST index
contains **547 distinct function names** in this namespace, plus 26 names in
each of `ha`, `la` and `ep`. Those counts collapse overloads and include
conversion/packed-operation spellings; they are not counts of GPU instructions.
The [complete name list](sycl/INTEL_API_NAMES.md) avoids omitting rarely used
migration-oriented operations.

| Family | Operation coverage |
| --- | --- |
| Extended real math | Standard-like elementary math plus inverse error functions, scaled complementary error function, normal CDF/inverse, Bessel functions `j0/j1/jn/y0/y1/yn`, modified Bessel I0/I1, reciprocal cube root, reciprocal hypot/norm, 3D/4D norms, sigmoid |
| Accuracy variants | `ha`, `la`, `ep` variants of 26 transcendental/math names; check each accuracy contract, not just its name |
| Explicit rounding | Float/double add, subtract, multiply, divide, reciprocal, square root and FMA with `rd/rn/ru/rz` variants where provided |
| Numeric conversion | Float/double/half/BF16 and signed/unsigned integer conversions with explicit rounding suffixes; integer width and range remain part of the contract |
| Representation conversion | Float/integer and double/64-bit bit reinterpretation, high/low double words, half/BF16 bit representations, packing/unpacking pairs |
| Half/BF16 arithmetic | `hadd/hsub/hmul/hdiv/hfma`, pair variants, saturation/ReLU forms, negation/absolute value, comparisons and NaN-sensitive extrema where provided |
| Integer utilities | Bit reverse, leading-zero/first-set/popcount, byte permutation, high-half multiply, 24-bit multiply, sums of absolute differences, signed/unsigned half-add and rounded half-add |
| Packed 2x16 / 4x8 integer math | `vabs`, absolute difference, saturating add/sub/negate, averages/half-add, min/max, comparisons returning masks or set values, sums of absolute differences |
| Compound packed integer operations | Three-way min/max, add+min/max, ReLU variants, comparison-result variants (`viadd*`, `vib*`, `vimax*`, `vimin*`) |

Evidence is **H/D** for the installed Intel device-library interface, not V for
all 547 names. Some operations are sequences or library calls. These are useful
for expressing exact conversion and CUDA-compatible arithmetic contracts, but
availability does not establish superiority over ordinary SYCL expressions.
They also are not automatically callable from ESIMD functions.

### BF16, packed dot, complex and other extensions

| Interface | Operations | Evidence / scope |
| --- | --- | --- |
| `ext::oneapi::dot_acc` in `dot_product.hpp` | Four INT8 products plus INT32 accumulator; all signed/unsigned operand pairings; packed 32-bit or four-byte vectors | D/H installed interface; compare against ESIMD `dp4a` for code generation |
| BF16 storage built-ins | BF16 representation conversions and operations provided by `bf16_storage_builtins.hpp` | H; explicit storage type matters |
| `experimental` BF16 math | `isnan`, `fabs`, `rsqrt`, `fmin`, `fmax`, `fma`, `ceil`, `cos`, `exp`, `exp10`, `exp2`, `floor`, `log`, `log2`, `log10`, `rint`, `sin`, `sqrt`, `trunc`, `tanh` | D/H; several installed implementations widen to FP32 and narrow back |
| Experimental complex | Construction, real/imag access, arithmetic, comparison, conjugate/projection, magnitude/phase/norm/polar, powers, roots, exponential/logarithmic, trigonometric/hyperbolic and inverses | H; inspect permitted half/float/double and scalar/marray overloads; `std::complex` device support is a separate contract |
| Experimental native math | Additional native/reduced-accuracy overloads described by `sycl_ext_oneapi_native_math` | D/H; distinct from standard namespace guarantees |
| Kernel floating-point controls | Intel FP-control kernel properties for rounding and denormal behavior | H; kernel/backend/type restrictions; not globally enabled for this study |

## 7. Memory movement, images and resource operations

These are included because excluding them would miss much of the useful SYCL
surface, even though allocation or a DMA copy is not an arithmetic instruction.

### Buffers, USM and transfer operations

| Family | Operations | Evidence / limits |
| --- | --- | --- |
| Buffers/accessors | Allocate/manage typed buffers; device/local/host accessors; access modes; sub-buffers; reinterpretation; explicit/implicit dependency tracking | D; accessor modes affect hazards and transfers |
| USM allocation | `malloc_device`, `malloc_host`, `malloc_shared`, aligned forms, allocator and allocation queries, `free` | Q/V selected allocations; allocation type determines residency/access rules |
| System allocations | Ordinary CPU allocations accessible to device under the system-USM aspect | Q; residency and migration costs differ from device USM |
| Transfer and initialization | Queue/handler `memcpy`, typed `copy`, `memset`, `fill`, accessor copies, `update_host` | D/V selected copies; dependencies and lifetimes remain explicit for USM |
| 2D transfers | `ext_oneapi_memcpy2d`, `copy2d`, `memset2d`, `fill2d` extension forms | D/H; pitch/width/height contracts; not a tensor-transpose arithmetic API |
| Placement advice | `prefetch`, `mem_advise`, device-read-only and pinned-host properties | D/H; implementation may treat advice as hints |
| Annotated pointers/allocations | Allocation alignment, cache controls and pointer/kernel-argument properties | H; attaching a property does not bypass the runtime's allocation capabilities |
| Async allocation | Memory pools, enqueue allocation/free operations | Q `ext_oneapi_async_memory_alloc`; allocation lifetime is dependency-ordered |
| Virtual memory | Reserve/free address ranges; physical memory create/map/unmap; access permissions and granularity | Q `ext_oneapi_virtual_mem`; not automatic unlimited VRAM or a paging-performance guarantee |
| Memory export and IPC | Exportable device memory, inter-process handle export/import/release | Q relevant aspects; external ownership/lifetime and same-device restrictions |
| Device globals | Persistent device-global objects and host copy interfaces | D/H; context/device/image scope and initialization restrictions |
| Peer access / composite devices | Peer enable/disable/query and component/composite-device interfaces | H/N for this single B570 setup; composite/component aspects both false |
| Host-memory registration proposals | Register existing host memory for device access | H upstream proposal/interface only unless installed support is established; not a replacement for tested `malloc_host` |

The [USM investigation](SYCL_USM_ATOMICS.md) remains the reference for concurrent
host/device atomic access. A 64-bit pointer or global fence does not upgrade
host/shared allocation concurrency.

### Images, sampling and external graphics synchronization

| Route / operation | Local evidence | Consequence |
| --- | --- | --- |
| SYCL 2020 sampled/unsampled image API | **N** `aspect::image=0`; compiler warns this API is unsupported | Do not use standard-2020 image support as the basis of a port |
| Intel legacy image API | **Q** `ext_intel_legacy_image=1` | SYCL 1.2.1-style image/accessor/sampler route is separate |
| Bindless image allocation, handle creation/destruction, copy, read/write/sample/fetch | **Q** `ext_oneapi_bindless_images=1` | Device-accessible handles and explicit lifetime; exact format/filter/dimension restrictions still apply |
| Bindless 1D/2D USM image storage | **Q** both true | Does not imply shared-USM backing or every sampling operation |
| Bindless shared-USM image storage | **N** | Use permitted image/device-memory allocation forms |
| Sampled-image fetch, 1D/2D images and USM, 3D images | **Q** five fetch aspects true | Fetch support is not the same as filtered sampling from USM |
| Sample from 1D/2D USM backing | **N** both `bindless_images_sample_*_usm` false | Do not conflate with the corresponding fetch aspects |
| Image arrays | **Q** | Layering is distinct from cubemaps |
| Mipmap creation, level reference and anisotropy | **N** all three aspects false | Hardware graphics support does not establish this SYCL implementation path |
| Cubemap/seamless cubemap sampling | **N** | No advertised bindless cubemap path here |
| Image gather | **N** `bindless_images_gather=0` | Not equivalent to ordinary memory gather |
| sRGB extension | **N** `ext_oneapi_srgb=0` | Do not assume implicit sRGB conversion |
| Per-dimension distinct addressing modes | **N** `unique_addressing_per_dim=0` | Sampler configuration must stay within supported combinations |
| External memory / semaphore import | **Q** both true | Enables specified graphics/compute interop; handle type, format, synchronization and ownership must be checked |

Relevant computations include integer/float texel read/write, normalized channel
conversion, nearest/linear filtering, coordinate normalization and addressing
(clamp/repeat/mirror where permitted). Their supported channel order/type and
sampler combinations are format-dependent. The local query reports legacy
image limits 16,384x16,384 for 2D and 2,048 per dimension for 3D; do not substitute
these for bindless allocation-specific limits.

Source: [bindless image specification](https://github.com/intel/llvm/blob/16fff99e152f8485cda1905fcd9bbd18b347849c/sycl/doc/extensions/experimental/sycl_ext_oneapi_bindless_images.asciidoc),
installed image headers and all image-related aspect values in the capability log.
Image execution and external semaphore/memory import were not exercised.

## 8. Submission, graphs, diagnostics and execution controls

| Family | Operations / controls | Evidence / constraints |
| --- | --- | --- |
| Queues and events | Ordered/unordered submission, `depends_on`, waits, status, async-error propagation and profiling timestamps | D/Q; separate queues need explicit USM dependency edges |
| Queue extensions | Barriers, priority, emptiness query, discard events, last/in-order event facilities, immediate command lists and queue index | D/H; queue index/priority does not by itself guarantee a dedicated copy engine or overlap |
| Command graph | Explicit nodes/edges and queue recording, finalize, execute/replay, update/dynamic facilities supported by the installed graph version | Q; earlier V simple replay only, not every node type or update pattern |
| Native command enqueue / interop | `get_native`, `make_*`, `interop_handle`, enqueue native commands; Level Zero/OpenCL integration | D/H; native command lifetime and completion must be reflected in SYCL dependencies |
| Host task | CPU work as a node, optionally using native handles | D; CPU execution, not a B570 arithmetic unit |
| Event modes and profiling tags | Experimental event configuration and queue profiling markers | Q profiling-tag aspect; remaining forms H |
| Timing | Queue profiling; experimental device clock | Q subgroup clock only; workgroup/device clock aspects false. Do not compare unrelated clock domains |
| Device wait | Extension for supported waiting behavior | Q aspect; not a substitute for an arbitrary persistent GPU spin protocol |
| Kernel queries | Workgroup/subgroup limits, local/private/spill memory, maximum registers/workgroup, compute-unit and architecture queries | Q/D/H depending on descriptor; query the actual compiled kernel |
| Register/occupancy controls | Intel GRF-size/automatic-GRF properties, kernel execution properties, subgroup/workgroup requirements/hints, `kernel_args_restrict` | H/D; changes can trade occupancy against spills; no optimal setting measured |
| Cache controls | Kernel cache configuration, annotated pointer L1/L2 policy, ESIMD per-message hints | H/D; not all controls compose, and cache bypass is not coherence |
| Compilation and bundles | Online compile/link, specialization constants, source/SPIR-V bundle creation, raw kernel args, free-function kernels, device-image content and SYCL binary facilities | Q online compile/link; H/D specific extensions; these change delivery, not the arithmetic set |
| Virtual calls | Device virtual-function extension | Q aspect; indirect-call restrictions and definition visibility apply |
| Diagnostics | `sycl::stream`, experimental `printf`, device assertions, asynchronous exceptions | Q native assertion; not ESIMD-general or a normal high-throughput logging path |
| Device management queries | PCI/UUID, memory information, topology, clocks/throttle/fan/power metadata | Q most installed Intel descriptors; these are observability, not new compute operations |
| Khronos extensions in installed headers | Work-item queries, group interface, static/dynamic address-space casts, free-function command submission, granular include headers | H/D by header; do not overlook these because they live under `sycl/khr`, not `sycl/ext` |

The runtime reports both graph and limited-graph aspects true. This is not a
contradiction: do not interpret the limited flag as negating the full one.
Buffer use, host tasks, library calls, dynamic updates and allocation nodes
have their own graph-version limitations. New graph APIs must be checked against
the installed version instead of a newer specification example.

FPGA pipes/LSUs/datapath/register and FPGA memory-placement extensions, CUDA
texture-cache/cluster/barrier primitives and HIP-specific interfaces are not
B570 implementation choices. Intel LLVM's removed and deprecated documents
are retained in the extension index so they cannot be mistaken for current APIs.

## 9. oneMKL numerical operations

Installed oneMKL 2026.1 has SYCL headers for BLAS, LAPACK, Sparse BLAS, DFT, RNG,
VM, statistics and experimental data fitting/distributed DFT. **A header in this
list is not evidence that the domain runs on GPU.** Intel's 2026.0 public device
support tables are the documented baseline below; the installed 2026.1 headers
also expose newer entries, explicitly marked H where appropriate.

USM interfaces usually accept dependency events and return an event; buffer
interfaces express dependencies through accessors. Most are host submission
APIs. Device RNG/VM are separate headers. Row/column-major layout, leading
dimensions, precision, transpose/conjugation, workspace and batch form must be
selected together. A tiny correct GEMM does not measure XMX utilization.

### Dense BLAS and extensions

The documented Intel GPU BLAS table supports levels 1, 2, 3 and BLAS-like
extensions (**D**); only FP32 2x2 GEMM is **V** here.

| Family | Routine names / complete operation groups |
| --- | --- |
| Level 1 | `asum`, `axpy`, `copy`, `dot`, `dotc`, `dotu`, `iamax`, `iamin`, `nrm2`, `rot`, `rotg`, `rotm`, `rotmg`, `scal`, `sdsdot`, `swap` |
| General matrix-vector / rank update | `gemv`, `gbmv`, `ger`, `geru`, `gerc` |
| Symmetric matrix-vector / rank update | `symv`, `sbmv`, `spmv`, `syr`, `syr2`, `spr`, `spr2` |
| Hermitian matrix-vector / rank update | `hemv`, `hbmv`, `hpmv`, `her`, `her2`, `hpr`, `hpr2` |
| Triangular matrix-vector / solve | `trmv`, `tbmv`, `tpmv`, `trsv`, `tbsv`, `tpsv` |
| Level 3 | `gemm`, `symm`, `hemm`, `syrk`, `syr2k`, `herk`, `her2k`, `trmm`, `trsm` |
| Additional arithmetic | `axpby`, `gemm_bias`, `gemmt` |
| Batched forms | `axpy_batch`, `copy_batch`, `dgmm_batch`, `gemm_batch`, `gemv_batch`, `syrk_batch`, `trsm_batch`; strided/grouped forms where declared |
| Matrix movement/arithmetic | `omatcopy`, `imatcopy`, `omatadd`, `omatcopy_batch`, `imatcopy_batch`, `omatadd_batch` |

Real/complex and single/double precision are operation-specific. GEMM also has
installed FP16, BF16 and INT8 overloads, including selected mixed result types.
Compute-mode selection can permit reduced-precision products. It is not safe to
use “FP32 GEMM” as shorthand for an unexamined math mode, or to assume every
level-1/2 routine has the same low-precision overloads as GEMM.
[BLAS support table](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2026-0/blas-functionality.html).

### LAPACK

| Group | Documented Intel GPU routines (D) |
| --- | --- |
| LU factorization, solve, inverse | `getrf`, `getrs`, `getri` |
| Cholesky factorization, solve, inverse | `potrf`, `potrs`, `potri` |
| QR and orthogonal/unitary Q | `geqrf`, `orgqr`, `ungqr`, `ormqr`, `unmqr` |
| Triangular solve | `trtrs` |
| Symmetric/Hermitian eigensystems and tridiagonal reduction | `syev`, `heev`, `syevd`, `heevd`, `syevx`, `heevx`, `sytrd`, `hetrd`, `steqr` |
| Generalized symmetric/Hermitian eigensystems | `sygvd`, `hegvd`, `sygvx`, `hegvx` |
| Singular values and bidiagonal reduction | `gesvd`, `gebrd` |
| Batched factorization/solve/Q | `getrf_batch`, `getrs_batch`, `getri_batch`, `potrf_batch`, `potrs_batch`, `geqrf_batch`, `orgqr_batch`, `ungqr_batch` |
| Workspace queries | Corresponding `_scratchpad_size` functions; allocate the requested element count for the routine's type |

The documented GPU exclusions (**N for these SYCL routines**) are `gerqf`,
`ormrq/unmrq`, `sytrf/hetrf`, `orgtr/ungtr`, `ormtr/unmtr`, `orgbr/ungbr` in
that summary table. A CPU LAPACK symbol in the same product does not establish
a SYCL GPU implementation. The summary is not exhaustive: installed headers
also declare `geinv_batch`, `gels`, `gels_batch`, `gesv`, `gesvda_batch`,
`getrfnp`, `getrfnp_batch`, `getrsnp_batch`, `ormqr_batch`, `unmqr_batch`,
`trtri` and `trtrs_batch`. Keep these in the candidate inventory (**H**, exact
backend/overload checks outstanding), rather than marking them unsupported
because the summary omits them. The individual
[`geinv_batch` page](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2026-0/geinv-batch-group-version.html)
explicitly documents GPU support (**D**) for real and complex float/double.
[LAPACK support table](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2026-0/lapack-functionality.html).

### Sparse arithmetic and formats

| Operation | Documented GPU formats / limits |
| --- | --- |
| `sparse::gemv` | CSR, COO, CSC, BSR |
| `sparse::symv`, `trmv`, `gemvdot`, `trsv` | CSR; do not promote CPU COO support to GPU support |
| `sparse::gemm` | CSR and COO sparse matrix times dense matrix |
| `sparse::trsm` | CSR triangular multiple-right-hand-side solve |
| `sparse::omatadd` | CSR sparse addition |
| `sparse::matmat` | CSR sparse product producing sparse output |
| `sparse::matmatd` | CSR sparse product producing dense output |
| `sparse::omatcopy`, `omatconvert` | Documented CSR/COO copy/transpose/conversion paths; direction-specific restrictions apply |
| `sparse::sort_matrix` | CSR; COO must meet its input contract rather than assuming a GPU sort API |
| Setup/optimization | Handle create/release, set CSR/COO/CSC/BSR data, matrix properties, `optimize_gemv/trmv/trsv/gemm/trsm`, matmat descriptors, `update_diagonal_values`; `omatadd`/`omatconvert` buffer-size, analyze and get-nnz stages |

These are **D**, unexecuted here. Sparse vector/level-1 routines are **N** in
this SYCL API. General sparse LU/Cholesky/QR, sparse eigensolvers, Poisson and
trust-region solvers are also **N** in the documented oneMKL SYCL solver surface;
CPU PARDISO/FEAST availability is a different interface.
[Sparse support and per-format table](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2026-0/sparse-blas-sycl-functionality.html),
[solver exclusions](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2026-0/sparse-solvers-functionality.html).

### Transforms, vector math, RNG, statistics and interpolation

| Domain | Operations | GPU evidence / exclusions |
| --- | --- | --- |
| DFT/FFT | Descriptor configuration/commit, forward/backward transforms, real/complex, 1D/2D/3D, batches, in/out of place, strides, scaling | D within descriptor restrictions; public table excludes direct 4D–7D transforms. Higher-dimensional transforms can be composed, which is a different implementation |
| Distributed DFT | Distributed descriptor and process-distributed FFT | H/D library path; requires distributed infrastructure, not tested on this single-GPU target |
| Vector Math | Array arithmetic; elementary, transcendental, rounding, special-function, power/root, magnitude and representation operations | D real FP32/FP64; N complex VM in the public GPU support table. Installed FP16 overloads are H for exact operation support |
| RNG host submission | Engine construction/state, `generate`, skip-ahead/leapfrog where supported, distributions into buffers/USM | D only the engine/distribution combinations below |
| RNG inside kernels | Device engines and distributions, `generate`, `skip_ahead`, engine state/counter access | Separate D/H interface; can fuse random generation into a custom kernel |
| Summary statistics | Min/max, sums and central sums, means/moments, skewness, kurtosis, variation | N GPU in the public SYCL statistics support table, despite installed headers |
| Data fitting | Linear and cubic Hermite spline construction and 1D interpolation | D; public table excludes other spline types, integration and standalone cell-search/callback operations |

References: [DFT](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2026-0/dft-functionality.html),
[VM](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2026-0/vector-math-functionality.html),
[statistics](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2026-0/summary-statistics-functionality.html),
[data fitting](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2026-0/data-fitting-functionality.html).

For VM, the installed USM declaration file contains these 91 distinct names.
This includes complex-only operations that remain subject to the exclusion above:

```text
abs acos acosh acospi add arg asin asinh asinpi atan atan2 atan2pi atanh atanpi
cbrt cdfnorm cdfnorminv ceil cis conj copysign cos cosd cosh cospi div erf erfc
erfcinv erfcx erfinv exp exp10 exp2 expint1 expm1 fdim floor fmax fmin fmod frac
hypot i0 i1 inv invcbrt invsqrt j0 j1 jn lgamma linearfrac ln log10 log1p log2
logb maxmag minmag modf mul mulbyconj nearbyint nextafter pow pow2o3 pow3o2 powr
powx remainder rint round sin sincos sincospi sind sinh sinpi sqr sqrt sub tan
tand tanh tanpi tgamma trunc y0 y1 yn
```

The public **host-submission** RNG GPU engine list is MRG32K3A, MT2203, MT19937,
PHILOX4X32X10, SOBOL, MCG59 and MCG31. ARS5, NIEDERR, WH, SFMT19937, R250 and
NONDETERM are excluded there. GPU distributions listed are uniform real/integer,
raw bits, Gaussian, lognormal, Poisson, Bernoulli, exponential, geometric,
Gumbel, Laplace, Rayleigh, Weibull and Cauchy. Beta, gamma, binomial, chi-square,
hypergeometric, multinomial, negative-binomial, PoissonV and GaussianMV are
excluded from that table.
[RNG support table](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2026-0/random-number-generators-functionality.html).

**Do not apply that host table to the separate device RNG interface.** Installed
device engines are `philox4x32x10`, `mrg32k3a`, `mcg31m1`, `mcg59`,
`pcg64_dxsm` and `count_engine_adaptor`. Device distributions include `uniform`,
`gaussian`, `lognormal`, **`beta`, `gamma`**, `uniform_bits`, `bits`,
`exponential`, `poisson`, `bernoulli` and `geometric`. The installed headers
establish these declarations; engine/distribution/type combinations still need
validation. In particular the newer PCG entry should not be inferred from an
older generic RNG support page.

## 10. oneDPL algorithms and device utilities

oneDPL launches algorithms with a SYCL device policy; it also provides code
callable inside kernels. Its algorithms can be composed into selection,
compaction, routing and sorting, but no workload-dependent speed advantage is
assumed. The installed policy-based declaration files enumerate the following
families (**D**; only the four-element `sort` case is **V**).

| Family | Named algorithms |
| --- | --- |
| Predicate and traversal | `all_of`, `any_of`, `none_of`, `for_each`, `for_each_n` |
| Search and count | `find`, `find_if`, `find_if_not`, `find_end`, `find_first_of`, `adjacent_find`, `count`, `count_if`, `search`, `search_n` |
| Comparison | `equal`, `mismatch`, `lexicographical_compare` |
| Copy/move/transform | `copy`, `copy_n`, `copy_if`, `move`, `swap_ranges`, `transform`, `transform_if` |
| Initialize/replace | `fill`, `fill_n`, `generate`, `generate_n`, `replace`, `replace_if`, `replace_copy`, `replace_copy_if` |
| Remove/unique | `remove`, `remove_if`, `remove_copy`, `remove_copy_if`, `unique`, `unique_copy` |
| Reorder | `reverse`, `reverse_copy`, `rotate`, `rotate_copy`, `shift_left`, `shift_right` |
| Partition | `is_partitioned`, `partition`, `partition_copy`, `stable_partition` |
| Sort/select | `sort`, `stable_sort`, `sort_by_key`, `stable_sort_by_key`, `partial_sort`, `partial_sort_copy`, `nth_element`, `is_sorted`, `is_sorted_until` |
| Merge and sets | `merge`, `inplace_merge`, `includes`, `set_union`, `set_intersection`, `set_difference`, `set_symmetric_difference` |
| Extrema/heap inspection | `min_element`, `max_element`, `minmax_element`, `is_heap`, `is_heap_until` |
| Numeric | `reduce`, `transform_reduce`, `adjacent_difference`, `inclusive_scan`, `exclusive_scan`, `transform_inclusive_scan`, `transform_exclusive_scan` |
| Object lifetime | `uninitialized_copy`, `uninitialized_copy_n`, `uninitialized_move`, `uninitialized_move_n`, `uninitialized_fill`, `uninitialized_fill_n`, `uninitialized_default_construct`, `uninitialized_default_construct_n`, `uninitialized_value_construct`, `uninitialized_value_construct_n`, `destroy`, `destroy_n` |

Additional interfaces include segmented/by-key reductions and scans, batched
`lower_bound`/`upper_bound`/`binary_search`, histograms, permutation/counting/zip/
transform/discard iterators, C++20 range algorithms, experimental ranges,
asynchronous algorithm forms, and dynamic device selection. They do not all
share identical algorithm coverage or return/completion semantics.
The kernel-template headers provide radix sort (SYCL and ESIMD variants) and
single-pass scan with tunable parameters. These are **H/D**, not measured here.
[oneDPL documentation and API divisions](https://uxlfoundation.github.io/oneDPL/index.html).

Kernel utilities include supported subsets of `array`, `tuple`, `optional`,
`complex`, `cmath`, `cstring`, `functional`, `numeric`, `limits`, type traits,
iterators and utility functions. “Supported subset” must not be expanded to
host containers, filesystem or arbitrary allocation inside a device kernel.
Device RNG includes linear-congruential, subtract-with-carry and discard-block
engines, minstd/ranlux aliases and scalar/vector forms; experimental Philox
facilities are version-sensitive. Distribution and seed/discard operations
must preserve reproducibility across work-item mapping changes.

## 11. oneDNN primitive, graph and training operations

The installed oneDNN 2026.0 supports a SYCL GPU engine/stream path (**D** at the
engine level). Creating a primitive descriptor for the exact device, data types,
layout, dimensions, attributes and propagation kind is the operation-level
availability test; `unimplemented` is a valid outcome. Header presence and
successful engine construction do not prove all combinations work.

| Family | Primitive operations, including training paths |
| --- | --- |
| Linear algebra | `matmul`, inner product forward, backward-data and backward-weights; batched/broadcasting forms where supported |
| Convolution | Convolution and deconvolution forward, backward-data and backward-weights; grouped/depthwise, stride/dilation/padding variants |
| Elementwise | Abs, round, square/sqrt, exponential/log, linear/power, ReLU/leaky ReLU, ELU, tanh, logistic, soft-ReLU, GELU variants, swish, mish, clip, hard-swish/hard-sigmoid; derivatives where supported |
| Binary | Add/subtract/multiply/divide, min/max, comparisons and select, with broadcasting |
| Activation with learned slope | PReLU forward/backward |
| Probability normalization | Softmax and log-softmax forward/backward |
| Statistical normalization | Batch, group and layer normalization forward/backward, scale/shift/statistics variants; local response normalization forward/backward |
| Spatial operations | Max/average pooling forward/backward; nearest/linear resampling forward/backward |
| Recurrent | Vanilla RNN, LSTM, GRU, linear-before-reset GRU, AUGRU and linear-before-reset AUGRU; forward/backward, direction and state variants |
| Reductions | Sum/product/min/max/mean and Lp-norm variants; data types and reduced dimensions constrained |
| Layout/composition | Reorder, concat, weighted sum, channel shuffle and backward shuffle |
| Attributes/fusion | Bias, scales, zero points, post-op sum/eltwise/binary, fpmath/accumulation/rounding modes, scratchpad and applicable deterministic/dropout controls |
| Graph | Operator graph construction, partition/fusion, compile and execute with SYCL interop; layouts, tensors and constant caches |
| Microkernel API | BRGEMM and data-transform descriptors are present in the product |

The microkernel API is not evidence of a B570 callable GPU tile primitive;
check its backend support separately. Use joint_matrix/ESIMD for the direct GPU
programming paths documented above.

**Installed-version issue:** Intel's 2026.0 release notes explicitly list
sporadic incorrect results for **FP32 matmul on Arc B-series**, plus FP16 matmul
performance regression on Xe2 and BF16 convolution performance regression on
B-series. This catalog therefore does not treat oneDNN 2026.0 as an unquestioned
correctness oracle for B570 matmul. A later release may fix an issue, but that
requires checking the selected version and running reference comparisons.
[oneDNN 2026.0 release notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/onednn/2026.html).

Low-precision memory types, weight decompression, per-group scales, FP8/FP4/MX
formats and fused attention are library features with architecture-specific
implementations. They do not extend the B570 native XMX format table. Similarly,
SDPA/GQA or gated-MLP graph fusion is a compound library implementation, not an
additional standard SYCL instruction.
[Matmul constraints](https://uxlfoundation.github.io/oneDNN/dev_guide_matmul.html),
[oneDNN operation and fusion documentation](https://uxlfoundation.github.io/oneDNN/dev_guide_inference_int8.html).

### Installed oneDNN graph operation-name inventory

The following is the installed `dnnl_graph.hpp` operation enum, not a claim that
all graph partitions containing these operations compile for B570. Sentinel and
wildcard operations are retained so the inventory does not hide non-computational
entries. Newer online documentation may list additional operators.

```text
Abs AbsBackward Add AvgPool AvgPoolBackward BatchNormForwardTraining
BatchNormInference BatchNormTrainingBackward BiasAdd BiasAddBackward Clamp
ClampBackward Concat Convolution ConvolutionBackwardData ConvolutionBackwardWeights
ConvTranspose ConvTransposeBackwardData Dequantize Divide DynamicDequantize
DynamicQuantize Elu EluBackward End Exp GELU GELUBackward GroupNorm HardSigmoid
HardSigmoidBackward HardSwish HardSwishBackward Interpolate InterpolateBackward
LayerNorm LayerNormBackward LeakyReLU Log LogSoftmax LogSoftmaxBackward MatMul Maximum
MaxPool MaxPoolBackward Minimum Mish MishBackward Multiply Pow PReLU PReLUBackward
Quantize Reciprocal ReduceL1 ReduceL2 ReduceMax ReduceMean ReduceMin ReduceProd
ReduceSum ReLU ReLUBackward Reorder Round RMSNorm Select Sigmoid SigmoidBackward
SoftMax SoftMaxBackward SoftPlus SoftPlusBackward Sqrt SqrtBackward Square
SquaredDifference StaticReshape StaticTranspose Subtract Tanh TanhBackward TypeCast
Wildcard GenIndex GreaterEqual LastSymbol
```

## 12. Other SYCL-accessible library and hardware domains

These are included even though they are outside Strata's initial inference path.
Their APIs cannot be treated as extra intrinsic instructions.

| Domain | Operations and path | Status for this study |
| --- | --- | --- |
| oneDAL analytics | Statistics, covariance/correlation, PCA; k-means and initialization; DBSCAN; k-nearest-neighbor; linear/logistic regression; decision forests; SVM; linear/polynomial/RBF/sigmoid kernels; distance/finiteness calculations | Installed 2026.1, D Intel GPU library family, H exact B570 algorithm/method/precision. Training/inference/partial-compute/finalize and dense/sparse variants have separate support |
| oneDAL graph/other headers | Connected components, Louvain, shortest paths, subgraph isomorphism, triangle counting, Jaccard; additional optimization/objective/distance headers | H declarations; do not imply GPU support for CPU-only algorithms just because oneDAL also has SYCL APIs |
| oneCCL | `allreduce`, `allgather/allgatherv`, `alltoall/alltoallv`, `broadcast`, `reduce`, `reduce_scatter`, barrier, point-to-point communication and reduction operators through SYCL streams | Installed 2022.1; H B570. The product's public GPU requirements list Data Center GPU Max; no B570 collective execution validated here |
| Intel SHMEM / distributed PGAS | Put/get, atomics, collectives and synchronization from device code where supported | Not installed/validated here; do not assume B570 support from general SYCL compatibility |
| Embree | Scene/BVH construction and ray intersection/occlusion traversal, geometry/instance/motion-blur forms through its SYCL GPU path | D Intel Arc GPU path; H exact B570 build/workload. A route to ray-tracing hardware, not a matrix accelerator API |
| Open Image Denoise | Neural denoising filters, auxiliary image inputs, SYCL device/queue integration | D SYCL interface; H selected B570 build/filter. No denoiser execution performed |
| SYCL-TLA and kernel frameworks | Tiled GEMM and fused tensor kernels built on SYCL/ESIMD | D documented B580 target in SYCL-TLA; H B570 configurations. Useful additional kernel-design references, not measured optimization baselines |
| Other numerical frameworks | Tensor, sparse solver, FFT and graph libraries can compose the same SYCL operations | Outside the finite installed/vendor API inventory; no claim that every third-party release supports B570 |
| Video encode/decode, display, rasterization and graphics shaders | Dedicated media/graphics APIs and imported/exported memory/semaphores | N as standard SYCL compute commands; interop does not expose codec/raster/display blocks as ordinary kernel instructions |
| CPU AMX, CUDA tensor cores and architecture-specific sparse/tensor instructions | Other hardware paths appearing in cross-vendor headers/specifications | N for B570; similarly this Ryzen CPU has no AMX |

References: [oneDAL 2026 requirements](https://www.intel.com/content/www/us/en/developer/articles/release-notes/onedal/2026.html),
[oneDAL algorithms and DPC++ examples](https://uxlfoundation.github.io/oneDAL/),
[oneCCL requirements and API](https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneccl.html),
[Embree SYCL ray tracing](https://www.embree.org/),
[Open Image Denoise SYCL device integration](https://www.openimagedenoise.org/documentation.html),
[SYCL-TLA target support](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/build/building_with_sycl_support.md).

### Less prominent compiler facilities and unimplemented proposals

The extension index additionally covers slicing (`cslice`), product-of-range
helpers (`prod`), property lists, weak object references, device/context
selection, platform indices, uniform-value annotations, architecture-conditional
code, and submission shortcuts. They either compose earlier operations or manage
objects/dispatch; they do not add an independent B570 arithmetic unit.

Upstream documents also describe append-and-shift, joint-for, generic barrier,
cache-size/local-static-memory/launch queries, named subgroup sizes, device-if,
address-cast, reusable events, maximum-register controls, range types and binary
bundle proposals. An upstream document's status and the installed macro/header
must be checked individually. Some concepts already have an installed spelling
under another extension or `sycl/khr`; other proposed spellings are absent.
The [143-document index](sycl/EXTENSION_INDEX.md) records these instead of silently
classifying every proposal as usable. The installed header manifest is the
version boundary for declarations not described by a feature macro.

## 13. Consequences for Strata implementation

This inventory does not select one implementation for every kernel. It provides
these concrete options and boundaries for subsequent measurements:

| Strata work | Candidate operations to compare | Conditions to preserve |
| --- | --- | --- |
| Dense/prefill GEMM | oneMKL, joint_matrix, ESIMD DPAS, selected oneDNN versions, tiled custom kernels | Precision/math mode, tile layout, scratchpad, tail handling, actual model accuracy |
| Quantized decode/GEMV and MoE experts | Vector load/bit extraction/BFN, DP4A/dot_acc, packed INT2/4/8 DPAS, FP16/BF16 dequantization+XMX | Model-specific codebooks/scales/zero-points; packing overhead and low-batch utilization |
| Norm, softmax, router and sampler | Standard/ESIMD math, subgroup/workgroup reductions/scans, group sort/oneDPL, fused kernels | Stable reductions, approximation error, tie-breaking, RNG determinism |
| RoPE, complex-style rotations, activation functions | Ordinary arithmetic, sincos/native/Intel math choices, BF16/FP16 conversion | Range reduction and accuracy; do not enable fast math without output checks |
| KV/cache/expert movement | USM device/host copies, 2D copies, block/scatter/gather I/O, cache hints, graph scheduling | This machine's measured PCIe path, transfer overlap, lifetime and allocation legality |
| CPU/GPU coordination | Events and dependent commands first; specialized synchronization only with new evidence | No shared/host-USM concurrent atomic assumption; no forced KMD-migration override |
| Advanced pipelines | ESIMD 2D loads/prefetch, SLM, named barriers, GRF tuning | Exact launch/resource/participation constraints and bounded progress tests |

Do not use llama.cpp throughput or kernel design as a B570 optimization ceiling.
It may help with porting contracts, but the candidate set here comes from the
SYCL/Intel interfaces and device evidence. Performance comparisons must include
actual Strata shapes, quantization layouts, CPU overlap and end-to-end execution.

The following remain deliberately **unmeasured**, rather than silently assumed:
signed/mixed-precision DPAS variants, every joint_matrix tile and accumulator
combination, checked/2D matrix I/O, advanced barriers, atomics by every type and
address space, image formats/samplers, indirect calls, runtime-generated kernels,
all library precision/layout combinations, and their throughput/latency. Their
APIs and constraints are cataloged above; choosing one for production requires
its focused validation. Unsupported paths remain in the catalog as exclusions.

## 14. Reproduce and update the evidence

These programs do not change the driver configuration. They use small bounded
kernels with no CPU/GPU spin protocol. Run them on a healthy GPU, with normal
settings, and stop on device loss rather than repeatedly submitting work.

```sh
source /opt/intel/oneapi/setvars.sh
mkdir -p /tmp/strata-sycl-catalog

icpx -std=c++20 -fsycl -DSYCL_DISABLE_IMAGE_ASPECT_WARNING \
  tools/sycl/capability_inventory.cpp -o /tmp/strata-sycl-catalog/caps
ONEAPI_DEVICE_SELECTOR=level_zero:gpu timeout -k 2 20 \
  /tmp/strata-sycl-catalog/caps

icpx -std=c++20 -fsycl -O2 tools/sycl/operations_probe.cpp \
  -o /tmp/strata-sycl-catalog/ops
ONEAPI_DEVICE_SELECTOR=level_zero:gpu timeout -k 2 30 \
  /tmp/strata-sycl-catalog/ops

icpx -std=c++20 -fsycl -O2 \
  -I/opt/intel/oneapi/mkl/2026.1/include \
  -I/opt/intel/oneapi/dpl/2022.13/include \
  tools/sycl/library_probe.cpp \
  -L/opt/intel/oneapi/mkl/2026.1/lib \
  -lmkl_sycl_blas -lmkl_intel_lp64 -lmkl_sequential -lmkl_core \
  -o /tmp/strata-sycl-catalog/libraries
ONEAPI_DEVICE_SELECTOR=level_zero:gpu timeout -k 2 30 \
  /tmp/strata-sycl-catalog/libraries
```

For the header inventory, compile an AST without executing device code. The
selected headers are in [api_index.cpp](../tools/sycl/api_index.cpp): standard
SYCL, Intel math, stable ESIMD and experimental ESIMD math/memory. The JSON AST
is about 464 MiB in this compiler; it stays temporary. The committed output
contains names/hashes, not the compiler's full AST or vendor source bodies.

```sh
icpx -std=c++20 -fsycl -fsycl-host-only -fsyntax-only \
  -Xclang -ast-dump=json -Xclang -ast-dump-filter=sycl::ext::intel \
  tools/sycl/api_index.cpp > /tmp/strata-sycl-catalog/intel-ast.json
icpx -std=c++20 -fsycl -fsycl-host-only -dM -E \
  tools/sycl/api_index.cpp > /tmp/strata-sycl-catalog/macros.txt

python3 tools/sycl/catalog_inventory.py \
  --include-root /opt/intel/oneapi/compiler/2026.1/include/sycl \
  --include-root /opt/intel/oneapi/mkl/2026.1/include/oneapi/mkl \
  --include-root /opt/intel/oneapi/dnnl/2026.0/include/oneapi/dnnl \
  --include-root /opt/intel/oneapi/dpl/2022.13/include/oneapi/dpl \
  --intel-ast /tmp/strata-sycl-catalog/intel-ast.json \
  --output docs/sycl
```

To regenerate the extension index, supply the pinned upstream
`sycl/doc/extensions` directory to the same command with
`--spec-root PATH --spec-revision 16fff99e152f8485cda1905fcd9bbd18b347849c`
and `--feature-macros /tmp/strata-sycl-catalog/macros.txt`.
The downloaded document paths and hashes are in
[EXTENSION_SOURCES.json](sycl/EXTENSION_SOURCES.json). The study fetched these
pinned raw documents and did not clone or alter another repository.

For a toolchain/driver update: regenerate the header/macro/aspect inventories,
compare added/removed API names and matrix combinations, then rerun only the
relevant bounded correctness cases before benchmarking. Re-querying a capability
is not equivalent to passing its execution tests.
