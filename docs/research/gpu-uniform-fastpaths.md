# Uniform GPU fast paths: profitability and code generation

**Use:** retrieval-only methods for kernels guarded by a subgroup- or workgroup-uniform condition. This note contains no Strata speed result, adoption recommendation, or device-specific cost threshold.

## Profitability

A uniform branch can skip a costly body for groups taking the fast arm without splitting the participating lanes into divergent paths. It still costs predicate formation and control flow; a vote adds its own work and dependency. The benefit depends on the actual fraction and placement of groups taking each arm, the work removed, and changes to code footprint and register allocation. Uniformity is scoped: all members of one group can agree while different groups choose different arms. A vote is not automatically a barrier or a single native instruction.

| Form | Can remove | Common counterexample | Check |
|---|---|---|---|
| Direct uniform branch | The body on groups taking the skip arm | A short body is cheaper to execute than to form/test the condition and branch | Confirm the condition is uniform for the intended group and inspect target code for the skipped body and branch. |
| Predication or select | Divergent control flow for small per-lane choices | Controlled operations remain scheduled; selecting after both values are computed does not avoid either computation | Inspect whether expensive operations disappear or remain issued under masks. NVIDIA’s predication rules are architecture/compiler-specific. |
| Vote then branch | A lane-varying test can become one decision for the vote’s group | The vote and its dependency cost more than the body saved, or the group mostly takes the slow arm | Check the exact vote scope and lowering, then compare against an equivalent direct-uniform or predicated form. |
| Hoist / loop-unswitch | Repeated invariant testing or a branch on every loop iteration | Hoisting only the comparison leaves the branch in the loop; unswitching clones the loop and can grow code | Inspect the CFG and final code. Try unswitching only when it changes generated control flow. |
| Compile-time specialization | Runtime condition and unused arm when the choice is known before compilation | More variants, compilation/dispatch cost, or a condition unavailable until device execution | Include real dispatch and cold/warm compilation costs where they apply. |

At SPIR-V level, subgroup vote operations reduce values over the subgroup; LLVM separately maps `GroupAll` and `GroupNonUniformAll` to `OpGroupAll` and `OpGroupNonUniformAll`. Check the actual group object and emitted scope instead of inferring scope or cost from a source helper’s name. ([Khronos subgroup-vote extension](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_subgroup_vote.html), [LLVM SPIR-V builtin mappings](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/SPIRV/SPIRVBuiltins.td#L3785-L3810))

## Code growth and register effects

Loop unswitching can remove a repeated decision and specialize each loop body, but LLVM’s implementation explicitly limits cloning by a code-size threshold. That is evidence of a compiler tradeoff, not proof that a particular GPU compiler enables the transform or uses the same threshold. ([LLVM `LoopUnswitch.cpp`, pinned revision](https://llvm.googlesource.com/llvm/+/4bd19b66187df029dd209b370e89e7d24b135033/lib/Transforms/Scalar/LoopUnswitch.cpp))

More unrolling or cloned paths can enlarge the hot instruction working set. NVIDIA’s H100 Smith–Waterman study is a useful counterexample to “more unrolling is always faster”; its result motivates checking instruction-fetch behavior, not transferring an NVIDIA threshold to Intel. The indexed `gpu_instruction_cache_unroll_dispatch_tradeoffs_20261023` reference contains the broader code-footprint evidence. ([NVIDIA instruction-cache case study](https://developer.nvidia.com/blog/?p=86868))

Register allocation can also depend on the largest live set across paths. AMD’s occupancy guide describes a shader whose peak register use in one branch can constrain allocation even when that branch is rarely taken. This is AMD evidence; measure Intel GRFs, spills, and relevant occupancy limits in the actual target build rather than assuming the same allocation rule. ([AMD GPUOpen occupancy guide](https://gpuopen.com/learn/occupancy-explained/))

## Evidence for a real kernel

1. Freeze equivalent source, compile route, launch geometry, and fast/slow input patterns. Include the production dispatch costs.
2. Inspect optimized IR, SPIR-V, and final target ISA. Verify vote scope, branch placement, predication, whether the costly operation is absent on the skip arm, and whether an invariant branch remains on a loop backedge.
3. Compare code size, live ranges, GRFs, spills, and available occupancy metadata as diagnostics. A shorter IR, fewer branches, or smaller register count alone is not a speed result.
4. Measure clean end-to-end kernel latency on representative distributions, keeping first-use/JIT separate from warmed execution. Repeat when spread could change the decision; report unavailable hardware metrics as unknown.

An Intel Triton backend document describes separate scalar/SIMT and explicit-vector paths that pass through LLVM IR and SPIR-V to IGC. That is a useful compiler-path example, not proof that another SYCL entrypoint uses the same lowering route. Intel’s ESIMD guide likewise shows that scalar control flow around per-element vector tests may emit branches, while fully unrolling can still leave multiple SIMD1 updates; this is a caution about code shape, not a cost model for a uniform branch in a different kernel. ([Intel Triton backend architecture](https://github.com/intel/intel-xpu-backend-for-triton/blob/main/docs/ARCHITECTURE.md), [Intel ESIMD conditional-operation example](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/optimizing-explicit-simd-kernels.html))

The CUDA guide explains that divergent warp paths serialize and that predicated instructions may remain scheduled even when their predicates are false. Treat this as NVIDIA-specific semantics, not an IGC threshold or latency estimate. Intel’s 2019 OpenCL guide recommends compile-time preprocessing when a variant is known ahead of time; it is historical specialization guidance, not evidence for the runtime cost of current Intel GPU code. ([CUDA control-flow guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/), [Intel OpenCL specialization guidance, 2019](https://www.intel.com/content/www/us/en/docs/opencl-sdk/developer-guide-processor-graphics/2019-4/using-specialization-in-branching.html))

**Limits:** Exact Intel vote instructions and latency, dynamic group-arm frequencies, register allocation, instruction-fetch effects, and end-to-end gain remain target- and workload-specific. A profiling attribution or codegen change does not establish throughput improvement. If a branch guards a group collective, verify its convergence and participation contract independently of its profitability.

**Retrieval tags:** `uniform-branch`, `subgroup-vote`, `predicate-vs-branch`, `if-conversion`, `loop-unswitch`, `code-duplication`, `instruction-fetch`, `register-live-range`, `IGC`, `Xe`, `sparse-kernel`.
