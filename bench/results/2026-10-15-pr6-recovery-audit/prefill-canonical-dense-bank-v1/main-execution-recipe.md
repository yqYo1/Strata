# Main-only serial canonical extension qualification (not executed)

Do not modify active fixed-source builds/measurements. Review source-review.md and immutable-base.json first. The only patch base is commit1025e0ed04816891c1df84c26ebf1bbc062784eb, which contains NativeDense bank v2. Main should assemble a fresh source/build from that immutable commit and overlay proposed files, or verify a reachable commit's exact pinned relevant hashes. Never compute/rewrite this diff against mutable production. Do not apply physical-consume-v2 wholesale; physical producer v3 is an independent later merge/review for Main.

This is header-only CanonicalDenseBank plus changes to WeightTable, NativeDense, Verifier, Prefill and generator; no new CMake cpp source entry is needed. Rebuild all actual source-resolved WeightTable::load and NativeDense ABI callers and the configured Arc generator, using the original compiler/flags/SYCL compatibility header path. Verify link-map/source/binary identity and absence of old duplicate symbols. All compilation/tests/GPU work belongs to Main and must be serialized with diagnostics/cleanup.

Host fixture example, with a private Main-owned output directory and immutable base includes:

```
c++ -std=c++20 -I<PROPOSAL>/proposed/include -I<IMMUTABLE_SOURCE>/include <PROPOSAL>/host_fixture.cpp -o <OWNED>/canonical-host
<OWNED>/canonical-host
```

HOST_ONLY is only policy/classification evidence. No physical model pass is inferred. Compile component_fixture.cpp against actual proposed weights.cpp/native_dense.cpp and configured production GGUF/MMVQ/SYCL dependencies from the isolated candidate (not both old and new ABI objects). Normal fixture invocation: `component-fixture PACK GGUF...`. It independently reopens pack tensor sources as a fixture reference only, compares every conversion/payload/metadata, all48 active overlays, persistent HC/PLE/non-layer rows, exact owned root restoration and the target-specific ledger. It does not run real Prefill/Verifier or task-quality inference. Current target census is canonical1538035200, bank32284672, persistent82821120; a different model's failure is not silently accepted.

For canonical allocation fault arms, compile only the fixture TU with STRATA_CANONICAL_ALLOC_FAULT. Its inline CanonicalDenseBank cudaMalloc calls use the local fake; actual WeightTable/native objects remain normal. Unarmed calls delegate to the real runtime. Each arm is a separate owned process:

```
fault-component prepare-oom-null PACK GGUF...
fault-component prepare-second-oom-null PACK GGUF...
fault-component prepare-unknown-null PACK GGUF...
fault-component prepare-unknown-nonnull PACK GGUF...
fault-component prepare-oom-nonnull PACK GGUF...
fault-component prepare-success-null PACK GGUF...
fault-component restore-oom-null PACK GGUF...
fault-component restore-unknown-null PACK GGUF...
fault-component restore-unknown-nonnull PACK GGUF...
fault-component restore-oom-nonnull PACK GGUF...
fault-component restore-success-null PACK GGUF...
```

Confirmed OOM-null must preserve the original Full state on prepare, or Bank+null published owner on restore, then permit the fixture's explicit no-reader retry and complete payload verification. Actual generator restore OOM does not retry or resume decode: it fail-stops. All uncertain arms must emit the production canonical allocation stop operation and abort while owners remain live. Exit91 means an unexpected ordinary return and is failure. Signal name alone is insufficient. Also inject NativeDense reservation failure after successful canonical prepare and assert checked reservation rollback leaves all original owners/publications unchanged. Separate subprocess injection is still required for staging allocation/upload/drain/free uncertainty, partial uploads and graph retirement/recapture errors; the provided allocator shim does not cover those APIs.

Actual generator environment must be fixed before exec. Required opt-ins are STRATA_PREFILL_NATIVE_DENSE_BANK=1 and STRATA_PREFILL_CANONICAL_DENSE_BANK=1. Use the existing admitted single-GPU native compact full layer-major configuration, explicit MMQ0, no token graph/split/peer/remote/vision and registered NativeEmbed. Keep STRATA_PREFILL_RELEASE_HEADS unset/0: simultaneous head release is still rejected. Do not use keep-canonical or model configurations with any native PackedSource.original.data/resident canonical overlap.

Qualify independent processes for: both flags unset/0, native-only1, both1 with short prompt (old full layout), both1 with multi-chunk long prompt; unsupported missing-native1/keep-canonical/overlap/graph-only/peer/split requests; captured SessionGraphs and Verifier graph+pending-commit routes; direct/fallback MTP callbacks; canonical BF16 PLE key and separately native quantized PLE key; cancel/callback-false/throw and entry/restore OOM. Confirm both destinations were allocated before first withdrawal, HC/lookahead refs and every canonical PLE pointer move to the charged pool, then both full owners/PLE bindings restore before pool free, SessionGraphs recapture and Verifier resume. Critically seed an existing Verifier commit_exec_ and verify it is retired/rebuilt: it contains canonical GDN/indexer norm addresses, unlike NativeDense-only retirement. Preserve all KV/GDN/QSA/PLE state buffers and expert cache/session ownership.

Use the standard Main-owned supervisor with finite deadline (e.g300s component/small model, a separately declared finite larger deadline for long/physical), exclusive measurement lock, <=64MiB total raw stdout/stderr, PID/start/boot identity, exact command/input/model/source/binary/toolchain/runtime/driver/effective environment and fresh kernel-journal boundaries. Never switch environment variables after provider initialization. Budget stops are incomplete. Retain compact operation-specific failure windows and all individual outcomes; no successful full API trace or full tensor dump is required.

After lifecycle/component gates, compare every-layer residual/GDN/QSA/PLE/indexer/KV/logit outputs under the identical FP32 reference and current residual/QSA/alias settings, restored short/decode behavior and real legal-context conversation RESTORE-and-continue. Then merge/review the fixed-base physical consumer separately and qualify actual262144/kv-resident32768; do not substitute full resident KV or context262145. A CPU/component result is not a physical model pass. Task quality and clean repeated long-prefill >600 toward1000 tok/s require separate measurements including capture/preparation/upload/restoration/dependency costs.

Planning payloads are in payload-ledger.json: persistent pool82821120, canonical bank32284672, native bank50918400, separate expert max1363148800. Candidate expert+dense+pool1529172992 is not the adaptive1440161792 hypothesis and excludes scratch/head/MTP/session/KV/staging/residual/workspace/runtime owners. Entry target-weight peak3821010432 and sequential restore peak3770092032 must fit alongside those unchanged owners. RAM snapshots total3654986240 payload plus bounded metadata; query real RAM/free GPU headroom and preserve restoration headroom. Neither allocated logical bytes nor cudaFree success alone establishes physical recovery.
