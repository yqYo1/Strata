R803 paired GEMM operator proposal — SOURCE ONLY / UNRUN

This package adds a private benchmark translation unit and a Main-owned build/run
controller. It changes no production API, Gemm ABI, source resolver, CMake target,
archive or model path. Main owns review, compilation, all tests, GPU execution and
any later adoption. The worker did not compile, run a syntax check or test, launch
a GPU/profiler, inspect processes, clean artifacts, or run Git mutations. The
explicitly requested `git show` reads used immutable base
`31ad125b7223668cc17bc9435b02021d5b8d32a9`.

The dispatch asked for operator evidence before tuning. The earlier whole-model
FP32 sample and the BF16 run's state at dispatch are not evidence that GEMM is the
bottleneck. No new speed or quality result is claimed here. R799 oneDNN is an
unadopted candidate with no accepted backend result and is absent from this fixture.

| Route | Existing implementation called | Stored output | Global FP32 intermediate |
| --- | --- | --- | --- |
| `old_f32_direct` | `Gemm::bf16` or `Gemm::f16`, oneMKL | FP32 | None beyond its FP32 output |
| `old_f32_plus_gpu_bf16_cast` | Same oneMKL call, then one GPU BF16 conversion kernel | BF16 | The oneMKL FP32 output |
| `native_bf16_output` | `Gemm::bf16_out16` or `Gemm::f16_out16`, `output_bf16=true` | BF16 | None |

The second route isolates storage-only output conversion overhead. It does not
reconstruct the original model pipeline. The native wrapper has only 16-bit
tensor pointers and calls the existing joint-matrix producer; FP32 accumulation
and epilogue arithmetic stay local inside that production kernel. The process
also allocates the other routes' FP32 output, but the native route does not read,
write or stage through it. No oneDNN, new GEMM implementation, fallback, clamp,
global FP32 expansion or per-route environment switch is introduced.

Every route uses the same immutable X and W buffers, `Gemm`, GPU device, context
and `Runtime::compute()` queue. Runtime's existing transfer queue is unused by
this fixture. All tensor storage uses registered `Allocation(MemoryKind::Device)`;
uploads and readbacks explicitly use the same compute queue. Both existing span
bounds and queue/context/device registry checks are retained. All outputs have
64-word prefix/suffix guards. Live outputs start as NaNs for beta zero, are reset
before each call outside timing, and must become entirely finite. X, W and their
guards are compared byte-for-byte after every invocation.

| Shape ID | T | N | K | Interpretation |
| --- | ---: | ---: | ---: | --- |
| 0 | 160 | 1280 | 2560 | Requested representative routed projection |
| 1 | 160 | 2560 | 640 | Requested representative routed projection |
| 2 | 8192 | 1280 | 2560 | Requested representative long projection |

These are the three requested choices, not a newly verified census of model HC
operators. All routes use `ldy=N`, `beta=0`. There is one owned process per shape
and input type, in fixed order `(0,fp16), (0,bf16), (1,fp16), (1,bf16), (2,fp16),
(2,bf16)`. Every process first warms direct/cast/native once, then measures three
rounds in orders direct/cast/native, cast/native/direct, native/direct/cast.
Every route receives exactly three individual measured samples. The controller
checks the declared shape, order, ordinal, output type and all validations.

Wall intervals start immediately before the selected route and end after
`compute().wait_and_throw()` and the runtime's sticky-error check. They include
submission, entry validation, all route work and completion, including the cast
route's second submission. They exclude uploads, resets, readbacks, host oracle,
hashes and validation. No GPU timestamp queries, event profiling collection,
instrumentation or full profiler are added. The inherited compute queue already
has a profiling property; its event timestamps are never used. These are isolated
operator wall measurements, not LLM throughput, task quality or model acceptance.

The independent host oracle uses a separable dense rank-two pattern. Let
`A=r%7-3`, `B=(r/7)%5-2`, `D=c%11-5`, `E=(c/11)%7-3`, and let `s(k)` be eight
positive ones followed by eight negative ones, repeated. Integer divisions here
are floors on nonnegative row/column indices. Set

```
X[r,k] = (2*A + B*s(k)) / 32
W[c,k] = (2*D + E*s(k)) / 64
Y[r,c] = (K*(4*A*D + B*E) + sum(s)*(2*A*E + 2*B*D)) / 2048
```

All X/W values are exactly representable in FP16 and BF16. The sum of signs is
computed from `K%16` in constant time; all selected K, including smoke K=48,
have sum zero. Products are integer multiples of `2^-11`. Even the sum of
absolute integer product numerators is at most `104*K`, which is below `2^24`
for these shapes. Thus FP32 products and intermediate sums are exact regardless
of reduction order. Every FP32 output must equal the closed form exactly, and
every BF16 output must equal its independent host RNE conversion (either sign
of exact zero is allowed). The host converter uses FP64 exponent decomposition
and mathematical ties-to-even rounding; it calls no SYCL, MKL or Strata bit
converter. All outputs are checked with O(T*N) oracle work, with no CPU dot
products. Input preparation and immutable-input readback are O((T+N)*K).
FNV-1a word-stream hashes identify generated tensors/results diagnostically;
source, binary, runtime and artifact pins use SHA256.

This pattern checks shape, sign, indexing, exact accumulation and output rounding
for these bounded inputs. Its low rank and repeated row/column patterns do not
provide adversarial numerical coverage. The existing original 68-case test and
its accumulator cancellation/range/bounds cases remain a separate required gate;
the controller pins its v2 PASS receipt and does not change or replace it.
Model quality, physical 262144 context, RESTORE and full pipeline checks remain
outside this proposal.

`--smoke fp16` and `--smoke bf16` use T=17,N=48,K=48 through all three routes.
They use the same parameterized native producer kernel and production GEMM
entrypoints as the larger cases, including the native partial tile path. oneMKL
may select different internal kernels by geometry. They emit the actual Intel vendor, device name,
driver, platform and complete matrix capability tuples. Main's smoke controller
uses the qualified UR tracing/parameter-validation environment and requires a
kernel-launch trace, full sanity PASS and normal owned exit. It runs these two
small processes separately and fully reaps each before any following process.
The measurement controller requires the completed two-smoke receipt and matches
its exact device/driver/tuple record. UR kernel-launch evidence and a GPU queue
identify submitted work; they do not independently prove that every arithmetic
step inside the vendor oneMKL library runs on the GPU. Do not infer a CPU or GPU
bottleneck from this fixture without the corresponding operator evidence.

The manifest pins the v2 build receipt, original 68-case receipt, baseline binary,
all eight linked project/ggml archives, four oneMKL libraries, qualified runtime
libraries, compiler/tools/selected SYCL headers, actual CMake recipe inputs and
direct_owner. It records the original v2 compile entry and final link stanza.
The v2 build was created at `9c1a2020897a9451ec616a6bb036a661de9d86b0`;
the manifest separately hashes the relevant APIs/implementations read at
`31ad125b...`, with their match status against that qualified v2 source. Before
launch, Main's controller requires all v2 source pins still match the live tree,
the immutable source dependencies match, and archives/toolchain/runtime/binary
pins remain unchanged. This is a strict compatibility gate, not a claim that
reading a newer commit rebuilt an archive. Pin failures require Main to review
and requalify a fresh proposal; the controller never silently refreshes pins.

The compiler lives below a directory whose name contains XeStrata. Only the
previously qualified official Intel LLVM installation there is used and hashed;
no XeStrata implementation/source is read, copied, adapted or linked. New package
code is authored for Strata under the repository LICENSE. The existing Strata
source and controller/owner interfaces are the implementation references; their
precise hashes are in the manifest. No source headers/templates with a new
license are introduced.

Main's controller acquires the established measurement lock, checks the explicit
latest prior receipt's boot and every recorded PID/start identity, requires
normal owned closure, and rejects detected same-UID model/fixture owners. The
additional bounded read-only cmdline scan covers engine/test names, known model
path arguments and repository server scripts; it cannot prove the absence of
uncooperative or hidden foreign GPU use. Main must serialize all GPU, build,
test and cleanup work through the shared lock. The controller does not adopt or
signal any process found by that scan. All child creation, PID pinning, limits,
signals to owned children, session observation and reaping use the pinned
existing direct_owner, without copying its source.

The controller reads the full current-boot kernel journal for preexisting fault
entries with the recorded filter, checks journal exit zero and cursor, checks
the expected driver and absence of device/core dumps, and takes fresh read-only
cursor boundaries before and after every GPU child and at completion. Any fresh
fault, unreadable/incomplete journal, unexpected dump, owner closure error,
boot change or pin drift fails the job and preserves its status. There are no
foreign signals, resets, services, dump removal, automatic retries or cleanup.

All complete build/smoke/measurement environments are assembled before exec;
there is no parent or initialized-runtime environment mutation. Clean measurement
processes have neither tracing/validation layers nor profiler settings. Safety
values come from the pinned receipts: copy offload disabled, direct submission
disabled, same queue enabled, persistent cache disabled and native capture checks
disabled. Smoke adds the exact qualified comma-separated layers and flushed UR
logger. Every child's effective environment is recorded by direct_owner.

Private build outputs and run receipts go below the existing external state root,
never into the source or qualified build tree. The controller reads v2
`compile_commands.json`, substitutes only this fixture's source/object paths, and
preserves the strict SYCL flags. It asks `ninja -t commands sycl_gemm16_test` for
the actual final link argv, then replaces only the single fixture object and
private executable output. It invokes that compiler argv directly using the
unchanged v2 archives; it does not configure, clone sources, build another target
or rebuild archives. Actual original and substituted compile/link argv are saved.
The aggregate build deadline is 600 seconds. Each GPU/check child has a 180-second
deadline; build/check RSS caps are 32/8 GiB, combined text cap 64 MiB, individual
file cap 64 MiB and total job cap 512 MiB. Budget or validation failure is an
incomplete/failed diagnostic, never a performance pass.

After Main reviews this source proposal, the intended command sequence is below.
The first `--prior` names the dispatched BF16 run's receipt; Main must wait for
that run's actual normal closure and use the actual latest owner if work has
advanced. A live, failed or non-normal prior receipt is refused. The example run
label is fresh only while those external roots do not already exist.

```sh
python3 -B bench/results/2026-10-15-pr6-recovery-audit/own-native16-paired-gemm-bench-v1/main_controller.py build --run-id r803-a --prior bench/results/2026-10-15-pr6-recovery-audit/own-native16-clean-bf16-r0-v2.json
python3 -B bench/results/2026-10-15-pr6-recovery-audit/own-native16-paired-gemm-bench-v1/main_controller.py smoke --run-id r803-a --prior /home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-r803-paired-gemm-r803-a-build/record.json --build-record /home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-r803-paired-gemm-r803-a-build/record.json
python3 -B bench/results/2026-10-15-pr6-recovery-audit/own-native16-paired-gemm-bench-v1/main_controller.py measure --run-id r803-a --prior /home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-r803-paired-gemm-r803-a-smoke/record.json --build-record /home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-r803-paired-gemm-r803-a-build/record.json --smoke-record /home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-r803-paired-gemm-r803-a-smoke/record.json
```

Every invocation emits an ordinal begin record, a flushed timing record marked
pending validation, then a separate full-validation status. All individual
samples, warmups, failures and command statuses are retained, including raw
bounded output if a process fails before JSON extraction. Main should commit
compact original samples, pins, recipes, environments, complete fault findings
and decisions after review. Raw first-smoke UR trace is external, with the
current consumer being Main's first-backend/tuple/normal-exit review, owner R803,
64 MiB combined job text budget and next review at smoke acceptance/rejection.
Once verified evidence is extracted, Main applies ARTIFACT_RETENTION.md in a
separate serialized cleanup step with a path/hash/reason manifest. This proposal
performs no cleanup and retains no tensors or full model traces.
