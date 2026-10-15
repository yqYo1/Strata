# Own K pitch40 proposal — unbuilt and untested

Only this proposal directory was written. No active source edit, build, test, GPU/runtime initialization, environment/service change, cleanup or Git operation occurred. All proposed correctness and timing gates are UNRUN. This is own-project implementation based on the current kernel, R725, R735 and R736; no XeStrata source or other implementation was read or used. Project skill/AGENTS and artifact retention policy remain applicable.

## Exact base and package

`base/` contains exact working-source bytes captured together before implementation; `snapshot.json` pins them, and `manifest.json` records base/proposed hashes plus diff hash. This deliberately does not assume HEAD identifies Main's working changes. The fixture base includes Main's `(void)scratch` strict-warning fix (SHA-256 `05147854dab32c93c8630d35584c7671f3ebc572298d33e227c5690e17899102`). Kernel base SHA-256 is `a22ae0834ead98e0b129b3f31a15cb06db99fe855e96b03c10f216df713f07eb`. CMake base/proposed bytes match `2544f5a426f3565ade25f3860c0f6317d1770fdc9e225cb0439ff5218ede14d2`.

`files/` contains complete proposed files, including unchanged actual SYCL CMake resolver. `source.diff` changes only kernel and CPU logical-tile proof in the existing fixture. `prepare.py` regenerates proposed files/diff/manifest from hash-verified `base/`, never from changing live source. Before application Main must compare actual bytes and review/rebase the complete patch, preserving concurrent work.

## Evidence and purpose

Main reports the completed plain cell-major qualification: full 4-arm/16-case component outputs exact, fault0; independent clean nq256x20 order0,1,1,0,0,1 gives OFF396.634/398.779/393.245ms and ON442.666/441.227/441.371ms, all six dumps exact. Plain cell-major is approximately11.28% slower and rejected for adoption, with no long-model trial. This proposal does not alter that status. R735/R736 predate these supplied results.

Pitch40 is a hypothesis to change the transpose store pattern after the negative pitch32 result. The cause of that regression is NOT measured; no bank-conflict, occupancy, sector-efficiency or speed claim follows from source strides. More local memory may regress further. Existing fine observer intervals are nonexclusive and do not identify K gather in isolation. No600/1000tok/s, model quality, RESTORE or physical262144 qualification is claimed.

## Arm selection and admission

Use the existing immutable-before-exec environment `STRATA_SYCL_PROMPT_K_CELL_MAJOR`:

- Unset,0,empty or other nonexact values: mapping0, original dimension-major gather and half accessor.
- Exact1: mapping1, already measured unpadded cell-major pitch32.
- Exact2: mapping2, cell-major K pitch40.

Candidate1/2 select only Mode1 K8V8 with existing Direct1 semantics, Native=false. Modes0/2/3 and Direct0 ignore the candidate switch; their dispatch remains unchanged. A compile-time integer0/1/2 replaces the bool; there is ONE extra device specialization (`<1,true,false,2>`), bringing total launch specializations10 to11. Existing symbols change template type, so compiled code/resource identity for baseline is a gate, not an assertion.

The existing actual matrix/subgroup/device-size support check still occurs before dispatch. Candidate2 additionally checks device-reported local memory >=64976 bytes (existing conservative Mode1 floor60880 +4096). If that added bound fails, it throws the explicit bounded error `prompt K pitch40 requires local memory >=64976 bytes`; it never silently runs mapping0/1. Unsupported matrix hardware still returns the existing API false before dispatch. The manual fixture treats false as failure. The additional query uses the same existing SYCL device-info API, with no new external GPU API. Compiler allocation can exceed declarations; the floor does not prove successful launch or occupancy.

## Layout, alignment and language contract

Mapping2 alone selects one-element `local_accessor<PaddedK>` where `struct alignas(4) PaddedK { sycl::half values[256*40]; }`. Static assertions require >=4 alignment, exact20480-byte size, standard layout, trivial copying and SYCL device copyability. Standard-layout first-member placement plus the explicitly aligned enclosing object gives the first half at a4-byte-aligned address. This is one genuine10240-element half array, not arrays of small row objects and not uint32 storage viewed through half pointers. Pointer arithmetic stays inside that single array, without aliasing/type-punning or `assume_aligned` promises. These static assertions and local object support must pass the configured compiler; they are not yet compiled evidence.

Within the device kernel, `&kv[0].values[0]` points into that local array and `sycl::address_space_cast<local_space,decorated::no>` constructs the local `multi_ptr<half>` used by the existing matrix API. The address-space cast reflects known local storage; it does not create alignment or change object type. Installed compiler support for this exact accessor/member-pointer/address-space lowering remains a compile/runtime gate.

For mapping0/1, the conditional type is still the original half local accessor of8192 elements; no second allocation or padded object is declared for those specializations. A common local half pointer aliases that same accessor. Mapping2 adds4096 bytes: K tile20KiB versus16KiB, nominal Direct1 accessor sum48596 versus44500 bytes. Actual compiler SLM/resources remain unknown.

Mapping2 writes K only at `dim*40+c`, dim0..255/c0..31. Max logical index10231 is below10240. Every QK B pointer changes to `kvp+k*KP+cell`, strideKP; k=0,16,...240 and cell=0/16. A16x16 load therefore consumes dimensions k..k+15 and cells cell..cell+15, never padding32..39. Its largest consumed index10231 is within the array. Each pointer byte offset `(k*40+cell)*2` is divisible by4. Row stride40 half =80bytes, divisible by8 and below2^24, satisfying the R735 primary documented BMG-G21 rules. Pitch33 is not proposed. Main should match that experimental specification to the installed compiler, as R735 requires.

V reuses only the first8192 half elements with unchanged `c*256+dim` stores; PV pointers remain `k*256+dim`, stride256 (512bytes). Their offsets are also4-byte aligned (k0/16, dim multiples16), and maximum footprint index8191 fits the array. The K-to-V barrier and all other barriers remain identical and uniform. K padding need not be initialized: QK cannot consume it, and V overwrites every element PV later reads. Query storage, selection/row resolution, scales, V values, numerical operations, CH32/WG128 and arithmetic order remain unchanged. Existing selected-ID/page/pool bounds preconditions are preserved.

## Fixture and future gates

The existing actual-SYCL `sycl_prompt_k_cellmajor_test` manual target continues linking `strata_kernels`, whose configured resolver explicitly contains `src/kernels/sycl/prompt_attention.cpp`. No CUDA test is used. Full normal fixture and nq128/256 benchmark paths are otherwise unchanged, including Main's strict-warning fix, immutable environments, limited benchmark oracle labels, full output dumps, warmup, timing/final drain, finite/tail checks and fail-stop local drain guard. Allocation cleanup already uses checked drain/fail-stop before free.

Only CPU logical-tile proof is extended: pitch32 and40 logical operands exact for all256 dimensions/32 slots, counts0/1/13/31/32, both KV heads, page indirection/invalid rows; pitch40 padding retains a sentinel with zero writers; all actual QK16x16 footprints are enumerated to prove bounds,4-byte relative offsets and exact consumed operands without padding. This is a logical/source index proof, not inspection of actual device SLM.

Main future sequence, serially supervised and NOT run here: review exact snapshot/diff, compile/link with actual configured SYCL toolchain and strict warnings, check static assertions and address-space/matrix overload support, then full normal fixture separate arms0/1/2 Direct1 with all outputs byte-exact and FP64/finite/bounds/fault gates; Direct0 and other modes must retain results. Next run independent immutable-process benchmark arms0/1/2 with identical nq256/counts/repetitions and exact full dumps. Example candidate command:

```
env STRATA_SYCL_PROMPT_DIRECT=1 STRATA_SYCL_PROMPT_K_CELL_MAJOR=2 <build>/sycl_prompt_k_cellmajor_test --bench 256 20 --dump <owned>/arm2.bin
```

Preserve all samples/source/binary/environment/process/fault identities; no per-expert events. Optional R734 offline AOT/resource inspection does not establish live JIT allocation or bank conflicts. Reject on parity, admission, compilation or performance failure; do not promote a legal stride into a speed claim. Only a successful matched component screen could justify Main's later actual78453 whole-prefill qualification. Normal full oracle and benchmark's limited queries0/1/2/last remain distinct. Main owns deadlines, log budgets, cleanup serialization and adoption decisions.
