# Main-only serial qualification recipe (not executed)

Do not apply to the production tree while its clean measurement remains active. Pin source-hashes.json and source.diff first. Use an isolated source/build directory after Main closes that run; preserve the configured CMake include resolver, compiler and all real target flags. Never replace active archives/binaries. Rebuild every native_dense ABI caller, Prefill, SessionGraphs, Verifier, MTP and the actual configured generate entrypoint. Check the link map against proposed source objects. CUDA/HIP qualification is outside scope.

Host fixture examples, with task-specific absolute output paths chosen by Main:

```
c++ -std=c++20 -I<V2>/proposed/include <V2>/host_fault_policy_fixture.cpp -o <OWNED>/host-policy
c++ -std=c++20 -I<V2>/proposed/include -I<PRODUCTION>/include <V2>/host_composite_resolver_fixture.cpp -o <OWNED>/host-resolver
<OWNED>/host-policy
<OWNED>/host-resolver
```

First compile the normal component fixture against proposed native_dense.cpp and actual production GGUF/weights/native_mmvq/SYCL dependencies; do not link old and proposed NativeDense definitions together. Run `fixture PACK GGUF...` in one owned process with the current target 300-matrix configuration; all packed bytes and metadata must compare. Run a separately extended fixture for native PLE and canonical-skipped forms; the current hardcoded census fixture does not cover extra PLE row capacity.

For actual NativeDense allocation faults, compile ONLY its fixture object with `-include <V2>/allocator_fault_shim.hpp`, using the exact SYCL runtime headers/flags. Compile component_fixture.cpp with `-DSTRATA_DENSE_BANK_ALLOC_FAULT_FIXTURE` and without that force-include; link the fault NativeDense object and normal production dependencies. No linker C-vs-C++ symbol wrapping is assumed. Each arm is a separate owned process:

```
fault-fixture oom-null PACK GGUF...
fault-fixture unknown-null PACK GGUF...
fault-fixture unknown-nonnull PACK GGUF...
fault-fixture oom-nonnull PACK GGUF...
fault-fixture success-null PACK GGUF...
```

OOM-null must complete full payload/restore checks. Each other arm must stop at the expected production NativeDense operation with SIGABRT and no ordinary-return marker; exit 91 is a failure. Keep one compact mechanism-relevant excerpt per arm, owner/device/build/environment identity, expected/actual status and fresh fault receipts. Do not group by signal name alone.

Then use actual generator model runs, separate immutable child environments: default switch unset, explicit0, opt-in1+short (old path), opt-in1+long+layer-major1+MMQ0. Use existing Main supervised bounded model/RESTORE controller; record command/input hashes, all environment, binary/source identity, sample order and fault boundaries. Do not infer safe entry from a component fixture. Require captured and uncaptured SessionGraphs and both Verifier presence arms; seed actual Verifier segments/work and pending commit first. Check all arrays/PLE publications are withdrawn before native free and recaptured/rebound after full upload. Check unchanged short/decode configuration, every layer state/output, restored decode and conversation continuation. Exercise native-only PLE ready(), synchronous MTP draft_kv and fallback, callback error/cancel/throw, entry OOM and restoration OOM. Inject checked sync/destroy/upload failures in owned subprocesses: uncertain completion or restoration failure must stop with owners live and no decode/RESTORE continuation.

Explicit long negative arms: layer-major absent/0, token graph handle/captured, split/stage/peer/remote, window prompt, head-release1, missing snapshots and preexisting hook. Each must reject before dense release. Head-release-only must retain prior behavior. Unsupported compact/MMQ geometry must reject safely and restore any transaction already acquired.

Use <=64 MiB raw text per diagnostic job, finite deadline and Main's exclusive measurement lock. Commit compact individual outcomes; no successful full API trace is required unless a named unresolved question consumes it. No cleanup concurrent with builds/tests/GPU. Numerical/RESTORE/physical262144 gates precede clean repeated throughput; do not claim actual 2,066,032,640 B recovery or >600 tok/s from theoretical sizes.
