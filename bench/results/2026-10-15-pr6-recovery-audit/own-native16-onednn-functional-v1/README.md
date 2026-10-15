# R804 standalone oneDNN faithful BF16 comparison

Source proposal only: **build, load, tests, GPU work, profiling and measurements
are UNRUN**. `OneDnnFaithfulBf16` has a real public-API execution path, with no
unconditional denial/ISA-proof gate. Actual unsupported engine/context, API,
geometry, descriptor or temporary-memory requirements throw before GEMM
execution. There is no CPU retry, alternate queue, backend switch or install.
This does not establish that the installed library can run on this PC's actual
queue; Main must compile and qualify that combination.

The dispatch pins project base `66c4e6c17c4168657746eff827b81d3d95e90d19`.
`proposal.patch` adds four new files only; no old-file replacement or production
CMake hook is proposed. Main must check these additions against that exact base
and review the complete patch before integration. The existing `Gemm*_out16`
ABI, its exact-RNE contract, its 68 exact custom-GEMM tests and prefill dispatch
are unchanged. Existing-file read identities are recorded separately from the
dispatch's immutable base; this worker did not run Git commands to reconstruct
or modify that base.

## Contract and actual execution

`Y[T,N] = X[T,K] * transpose(W[N,K])`. Source, weight and destination descriptors
are exactly `bf16`, with explicit strides `{K,1}`, `{1,K}`, `{ldy,1}`. `ldy=0`
means `N`. The wrapper requires positive dimensions, `ldy>=N`, overflow-safe
extents and a registered device allocation covering the full `T*ldy` rectangle
for Y. It validates 2-byte alignment, pointer arithmetic, output/input overlap,
registered extents and the actual Runtime context/device before preparing the
primitive. Inputs may alias each other. Positive and negative zero beta are
accepted; every nonzero/nonfinite beta is rejected. No sum, bias, post-op,
reorder, FP32 destination or separate cast kernel is installed.

The wrapper explicitly requests and reads back strict FP32 accumulation, strict
floating math, user scratchpad and environment rounding. The comparison uses
oneDNN's **faithful** destination-conversion contract. It does not substitute
that contract for exact RNE in an existing Strata API. The official
[accumulation documentation](https://uxlfoundation.github.io/oneDNN/v3.11/dev_guide_attributes_accumulation_mode.html)
defines strict accumulation as FP32 for this floating primitive. The
[data-type documentation](https://uxlfoundation.github.io/oneDNN/v3.11/dev_guide_data_types.html)
guarantees faithful destination conversion and describes the default SYCL
rounding/denormal behavior. Those statements do not prove a universally
correctly-rounded BF16 destination or the selected instruction sequence.

`submit` calls `dnnl::sycl_interop::execute` on the actual matmul primitive and
returns its event. `prepare` reports implementation name, device/driver, queue
backend, runtime source hash, primitive-cache capacity, actual strict attributes,
BF16 descriptor equality and temporary sizes. The wrapper creates its engine
from the caller's explicit GPU device/context, then verifies `kind::gpu`, the
device and context. It wraps and verifies the caller's in-order queue. It never
uses a CPU engine, ordinal-based library engine selection or a replacement
queue. `make_engine`'s public signature has no engine-kind parameter; the exact
oneDNN C implementation selects GPU kind from `device.is_gpu()`.

GPU-engine identity and `impl_info_str` are observables, not native-XMX proof.
The library may choose GPU reference/emulation implementations. The generated
installed config does not disable GPU reference kernels. This proposal makes
no claim that a selected kernel performs all arithmetic on GPU or uses native
BF16/XMX; Main must inspect actual execution and CPU-time attribution. There is
no host GEMM or numerical fallback in the wrapper.

X, W and Y remain two-byte BF16 tensors at the public boundary. `--chain`
submits a BF16 producer followed immediately by a BF16 consumer using the same
device output allocation as its next source, with no host round trip or whole
hidden-state FP32 restoration. Necessary internal FP32 accumulation and temporary
storage are allowed. This is a standalone GEMM component, not a claim that all
prefill operators now consume BF16.

## Installed API and Level Zero evidence

The inspected installation is `/opt/intel/oneapi/dnnl/2026.0`, headers 3.11.4,
generated source identity `0291f89430882d350585d276cc0d0eda623906e9`, Intel SYCL
GPU runtime. Static ELF inspection is recorded in `installed-evidence.json`;
it does not load the library. The public SYCL engine/stream/memory/execute and
accumulation symbols are exported. The library needs `libsycl.so.9`; matching
the configured Strata compiler/runtime ABI remains UNRUN. Runtime version and
source identity must match the headers this proposal was compiled against.

`DNNL_WITH_LEVEL_ZERO` is **undefined** in the installed generated config.
R804 does not equate that fact with compatibility or incompatibility. The exact
source identity was retrieved through the official GitHub API/raw endpoint:

- `src/xpu/sycl/capi/capi_engine.cpp`, engine-create path: derives kind from
  the supplied device, obtains its backend-specific device index and calls the
  factory with the supplied context.
- `src/xpu/sycl/engine_factory.cpp`, explicit device/context overload: validates
  consistency and routes Intel devices to the Intel SYCL GPU engine.
- `src/xpu/sycl/engine_impl.hpp`, initialization: explicitly admits `level0`
  among supported backend kinds before device/context checks.
- `src/xpu/sycl/utils.cpp`, backend mapping and device-index paths: explicitly
  maps SYCL Level Zero and has a Level Zero index workaround.
- `src/gpu/intel/sycl/utils.cpp`, OpenCL compiler-engine utility: contains a
  Level Zero branch that matches a device and creates an OpenCL compilation
  context. This library-internal path is not an application queue switch and
  does not prove that every compiled kernel can execute in the caller domain.
- `src/gpu/intel/sycl/engine.hpp/.cpp`: GPU engine and kernel creation paths
  exist. Inspected paths do not make their support conditional on the generated
  macro. This is a bounded source observation, not a whole-library audit.

`source-evidence.json` records URLs, hashes, exact read sections and limits.
The initial browser fetches failed; direct read-only HTTPS retrieval succeeded.
No third-party source is copied into the package or used to implement a kernel.
The only compatibility decision in the wrapper is the actual public API result
and returned domain/descriptor properties. A real unsupported Level Zero
context, strict attribute or geometry produces a failure, with API stage and
oneDNN status, rather than a skipped/pretend-success comparison.

## Temporary memory, cache and lifetime

The default and absolute temporary budget is **128 MiB per object**, and callers
can lower it. The wrapper queries the actual user scratchpad descriptor size
and the primitive-owned-memory query before creating/executing a primitive.
The C query's status is checked: the C++ `query_s64` helper silently returns zero
on a failed query, so that helper is deliberately not used for accounting.
An unavailable query, excessive size or sum above the budget fails explicitly.

One reusable registered GPU allocation backs user scratch for every cached
shape. No cache entry owns a separate scratch allocation. Scratch growth drains
queued work and memory wrappers, releases old storage, then allocates its
replacement, preventing a doubled allocation peak. Same-queue order plus the
previous oneDNN completion event serialize scratch reuse. Per-object accounting
includes live scratch and the sum of reported primitive-owned memory across
cached entries. Entries are evicted as necessary before creating a primitive.
The cache holds at most 32 primitive entries, including pending entry owners;
an eviction whose entry is still in flight first drains the queue.

The scratch size is an aggregate byte count: public APIs do not expose each
scratch region's datatype or whether it holds a full FP32 output/intermediate.
The fixture reports `full_fp32_output_equivalent_bytes=4*T*N` only as a comparison
scale, never as measured usage. **Actual full FP32 intermediate use and cost are
UNKNOWN until Main identifies/inspects the selected implementation.** If present,
report its size, transfers and timing honestly; zero user scratch is not proof
of zero hidden internal memory. A source-inspected matmul candidate nests the
selected GEMM's scratch registry; selection itself remains a runtime observation.

The wrapper's ownership and byte limits do not reconfigure oneDNN's process-wide
primitive/JIT caches or driver allocations. Their actual cache capacity is
reported. For an isolated memory comparison Main can freeze
`ONEDNN_PRIMITIVE_CACHE_CAPACITY=0` in the native child **before exec**, then
verify the reported capacity. This worker neither reads environment switches
to select an arm nor mutates any environment or library-global cache setting.

The object retains a `shared_ptr<Runtime>` and a value copy of the actual queue.
Its stream, engine, cached descriptors/primitives, shared scratch and all
per-call memory wrappers outlive queued work. No global wrapper cache, raw queue
pointer or static queue owner is used. At most 512 pending argument-owner nodes
are retained. Completed events can release nodes; reaching capacity drains at
a safe boundary. Nodes, maps and dependency vectors are allocated before
submission. A possibly partial enqueue failure poisons the object and retains
owners until a checked drain; execution is never retried on another backend.

`wait` drains the library stream and exact queue, then checks asynchronous and
Runtime sticky errors. `close` drains before releasing wrappers, primitives,
scratch, stream and engine. It is idempotent. If completion is proved but a
sticky async error is observed, it safely closes and rethrows. A wait outcome
that cannot establish completion retains owners. The noexcept destructor calls
checked close; an error fail-stops before unsafe member destruction. Strata's
existing `Allocation` release also uses its checked Runtime drain, including
the Runtime's other queues; that synchronization cost is not claimed to be zero.
Caller X/W/Y allocations remain caller-owned through completion. Host use of one
object must be serialized. The failure/hang path needs Main-owned supervised
fault qualification; it is not exercised by this source-only delivery.

## Standalone fixture and Main's qualification order

The optional CMake file is OFF by default and only creates a separate static
library and manual executable. It requires an explicit existing installation
root and does not call external package managers. Main may add an optional
include after `strata_sycl_runtime` exists in the actual configured SYCL CMake
resolver, after reviewing the add-only patch. It does not link production
`strata_prefill` to this candidate or install a prefill hook.

After that source review, a Main-owned build can use the existing toolchain and
`-DSTRATA_SYCL_ONEDNN_FAITHFUL_BF16=ON`
`-DSTRATA_ONEDNN_ROOT=/opt/intel/oneapi/dnnl/2026.0`, then explicitly build
`sycl_onednn_faithful_bf16_test`. No configure/build command was executed here.
Freeze each separate native child's SYCL/driver/provider/diagnostic environment
before exec and use Main's existing serialization, deadlines, log budget and
fresh fault checks. No test mode changes environment after initialization.

Fixture modes (all UNRUN):

| Mode | Proposed evidence |
| --- | --- |
| `--smoke` | One real small BF16 matmul, selected identity, strict attributes, faithful oracle, guards and checked close |
| `--inspect` | Actual primitive/identity/attributes/memory only; prints execution UNRUN |
| `--negative` | Dimensions/overflow, beta, alignment/nulls, all pointer extents, overlap, host/foreign USM, context/queue-order rejection; no primitive preparation |
| `--numerics` | 20 real cases: dyadic finite dots including nonmultiple/tail shapes, signed zero beta, midpoint neighbors, cancellation, exact subnormals, subnormal halfway results, NaN, signed Inf, opposite Inf, signed overflow and zero; stride/input/weight/guard checks |
| `--chain` | Two queued BF16 producer/consumer matmuls with the intermediate staying BF16 in device memory |
| `--lifecycle` | 40 shape submissions and stride key, 1025 repeated submissions, per-step bounds, independent object/queue, original queue-wrapper destruction, explicit close and destructor with queued work |
| `--scratch` | Actual scratch/owned size and zero-budget rejection when that actual shape needs nonzero temporary memory; explicitly NOT_EXERCISED if it does not |

The numerical oracle independently decodes BF16 into double and uses FP64
products/sums plus an integer search for the nearest BF16 encoding. It does not
convert BF16 subnormals through host float arithmetic; host FTZ/DAZ therefore
cannot silently turn a required subnormal into an accepted oracle zero. Finite
fixtures deliberately have exact FP32 sums regardless of reduction
association, so the test isolates accumulation precision and destination
conversion without a broad summation tolerance. An exactly representable BF16
oracle value must match **exactly**, including subnormals/zeros. Otherwise the
result must be one of the two BF16 values bracketing the exact sum and at most
one BF16 rank from its RNE oracle. Quiet NaN classification, signed Inf and
overflow are checked separately; none receive a finite relative/absolute error
budget. Payload preservation for NaNs is not required. A GPU that flushes an
exact BF16 subnormal fails. General long-K cancellation/reduction error and model
quality need separate Main-owned evidence; these small exact fixtures do not
prove universal numerical behavior.

Main's first step is a small traced smoke check and fresh fault evidence. Then
run all fixture modes and retain the existing 68 exact custom-GEMM tests without
changes. Reject or fix actual API/compiler/runtime/numerical failures rather than
turning them into skip statuses. Then pair component benchmarking against custom
BF16 and oneMKL under the same shapes/data/binary/toolchain, capturing all memory,
CPU computation/submission/wait and GPU work. Follow with the same 128-token model
comparison, then long-input and quality gates before any adoption. The task's
current single 78,453-token sample is FP32 346.5 versus custom BF16 123.8 tok/s in
the same binary; this is a regression and the custom BF16 route is not adopted.
That supplied observation is not an R804 measurement and predicts nothing about
oneDNN's speed. Main's R803 paired-benchmark source and active coarse profile are
separate work; this package does not change them.

## Provenance and retention

All new wrapper, fixture, CMake and report text is independently authored for
this Strata task and follows the repository MIT license. R799 is the permitted
own-code structural reference, not an adopted foreign implementation. R804
replaces its zero-scratch/universal-RNE/always-false policy with a distinct
faithful comparison contract and bounded real execution. The installed oneDNN
headers/library and inspected official source remain Apache-2.0 dependencies;
neither their headers, kernels, samples nor templates are copied or relicensed.
No XeStrata implementation, helper, diagnostic or archived proposal was read or
used. No third-party implementation is proposed for project licensing.

`manifest.json` inventories every payload file, byte count, SHA256 and provenance.
`manifest.sha256` pins the manifest without a self-hash cycle. `proposal.patch`
is generated from `/dev/null` additions using Python, without Git commands.
`installed-evidence.json` and `source-evidence.json` pin all referenced local
dependency files and fetched exact-source bytes/read sections; full external
files are not retained. `read-inputs.json` pins instruction, R799 and project API
inputs and records exactly what was read. The entire package must remain below
300 KiB. No raw run traces, tensors, binaries or captures were created, and no
cleanup or deletion was performed. Main owns integration, commit/push, all actual
validation, measurement and adoption decisions.
