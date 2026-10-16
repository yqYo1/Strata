# Bounded FP16 MoE candidate, 2026-10-14

Status: prepared patch only, not applied, compiled, tested, measured or adopted. Main exclusively owns subsequent builds, tests and GPU work. Product source, CMake, controllers, Git references, historical isolated tree and active historical build were not changed. No throughput claim.

Files: `compact-fp16.patch` is a unified patch targeting only `src/prefill/prefill.cpp`; `prefill.cpp.candidate` is its derived full file; `source-hashes.txt` pins original and candidate bytes. Proposal baseline is supplied feature/sycl 2bcbcc6f; applicability was not tested with git apply. The patch was visually inspected. Only proposal artifacts were written. An initial generation invocation could not find `python`; generation subsequently used python3. This was file generation, not a test.

## Origin and scope

Project strata-upstream-sync skill and artifact policy were read. R495 route correction was read directly from research-20261009. Donor own-fork 1eb89482a4afd20277ae0405780ed4f8eb98eb20 `sycl/src/prefill/prefill.cpp` contains bounded compact buffers and per-expert gather at historical lines 1018–1035 and 4792–4804. Its measured source selected the FP16-dequant/oneMKL route, not compiled MMQ/fused kernels. Historical composite settings/results do not qualify this narrow candidate.

Current root `src/prefill/prefill.cpp` stays the entrypoint; current `src/prefill/sycl/gemm.cpp` implements the existing FP16 GEMMs. The candidate imports only bounded MoE Xs/GU/Hh scratch and offset logic. QSA query/merge layout, HC aliases, layer-major scheduling, core/server/decode, quantization and precision policies remain outside this patch.

## Activation and changed active call sites

Set `STRATA_PREFILL_MMQ=0 STRATA_PREFILL_MOE_COMPACT_FP16=1` before process initialization. The new option requires exact string `1`. It admits only a plan with fallback=true and any=false, no STRATA_DBG_NAN presence, and native formats whose embedding dimension is 2560 and FF dimension is 640. Empty native format lists decline the option. Geometry validation already requires N=2560 and router K=10. Unsupported native shapes, diagnostics or any quantized layer retain original allocations and execution. There is no attention/XMX restriction because attention buffers are untouched.

Original-source locations:

- Impl MoE fields around line 448 gain compact_moe_fp16.
- Following mmq_plan around line 625, compact_moe_fp16() provides admission; fp16_moe_rows(T) supplies the same row count to counting and carving.
- moe_bufs around line 637: admitted GU holds T*1280 float elements, rather than T*K*1280. H shape metadata is T*640, but float H is not allocated with mp.any=false.
- moe_set_bytes around line 645 and take_buffers around line 832: Xs and Hh change to T*N and T*640 uint16 elements. All other takes preserve their sequence, type and count; both use the same helper and GU plan. Region max calculation, capacity estimator, init and relayout inherit the revised count through existing callers.
- Host routing count loop around line 2085 rejects an increment beyond T rows for any expert before offset construction and compact scratch use.
- Gather around line 2169 skips the full T*K input gather only when compact is admitted.
- FP16 compute around line 2510 validates positive ne<=actual T<=allocated m.T and [o0,o0+ne) within T*K. It gathers only m.src_dev+o0 into scratch row zero, then keeps original GU(ne,1280,N), SwiGLU and Down(ne,N,640) shapes. Down still writes m.Dm+o0*N. Expert ordering, DQ modulo selection, resident/staged dispatch, MMQ grouping and final moe_combine remain unchanged.

## Sizing and liveness proof

For actual chunk t<=allocated T, each expert's counted ne<=t is checked, rather than merely assuming unique top-K. Global routing ids retain their existing range check. The host-produced offset ranges partition t*K rows; each gather source row is generated as i/K and therefore lies in [0,t). The compute check independently bounds its source subrange. Compact Xs needs at most ne*N<=T*N uint16 elements; GU at most ne*1280<=T*1280 floats; Hh at most ne*640<=T*640 uint16 elements. Dm remains T*K*N floats because all experts' results are simultaneously needed by combine. Router ids, weights, source/slot maps and shared-expert output stay full sized.

Both count-only and carving use fp16_moe_rows(T) for the same two uint16 takes and moe_bufs for the same float GU take. Everything preceding/following those takes has identical order and alignment behavior. Thus sizes match when process environment and loaded expert layout remain fixed across count/carve, as current static mmq_plan already requires. Do not mutate environment/model layout concurrently with allocator planning. No independent stale compact sizing cache is introduced.

Unaligned payload reduction is (K-1)*T*(2*N+4*1280+2*640) = 103680*T bytes: 405 MiB at T=4096 and 810 MiB at T=8192. This is MoE payload arithmetic, not measured allocation reduction: alignment and the maximum of attention/GDN/MoE determine actual region savings. The full context capacity of 262144 remains governed by unchanged KV/QSA allocation and range logic.

For each expert, the same m.cs queues dequant -> gather -> GU -> SwiGLU -> Down -> next expert. Current SYCL queue_for in include/strata/sycl/launch.hpp explicitly requires in-order queues; current GEMM submits oneMKL onto that queue. Therefore the next expert's overwrite of Xs/GU/Hh follows the prior Down read. Input m.mixed_h remains live throughout the expert walk; m.src_dev remains full sized. Existing copy-stream waits, used-slot record after dequant, stage lease handling, queue draining/freeing and graph retirement are untouched. Per-expert gather has no source blob access, so it does not extend the blob lease beyond dequant. Numerical operations, weight dequantization and scatter positions are preserved; only intermediate locations and gather granularity change.

Diagnostics keep original full-layout arrays by declining compact when STRATA_DBG_NAN exists; current full T*K GU diagnostic thus cannot overread compact scratch. This means existing full-intermediate NaN diagnostics cannot be used to validate the compact arm directly; output/tensor comparisons or dedicated per-expert diagnostics are required before adoption.

## Unresolved checks for Main

No executable check is represented as passed. Main should apply-check against its chosen revision, inspect the actual configured root/SYCL compile resolver, compile and exercise option off/on at matched 4096 first. Validate per-layer output/logprob or tensor differences, both resident and streamed/non-streamed expert paths, final short chunks, tails and relayout, plus the pathological concentrated-routing guard. Confirm quantized mixed plans and STRATA_DBG_NAN retain original sizing/dispatch. Record admitted actual chunk before trying 8192. A declined compact option silently preserves stock behavior; Main must capture the effective mode rather than infer admission from environment alone.

Check large-capacity allocation, QSA physical boundary, actual 262144 context, KV paging/residency, cancellation/early error queue drain, staging reuse and API cache/RESTORE lifecycle. Follow the same current range guards and full-capacity correctness requirements. Shape admission is conservative but compilation and live-model layout remain unverified. Evaluate throughput only in quiet serialized matched processes with current math, attention and controls held fixed; historical composite performance cannot be attributed to this patch.

Retention: these compact source/patch/report/hash artifacts are the entire output of this proposal. No traces/captures were created or removed. Next consumer is Main's serial integration/review; next review boundary is candidate applicability, numerical and capacity validation. Forbidden private record was not read; no broad environment/run-record scan occurred.
