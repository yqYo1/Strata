Merge upstream Strata v0.1.39 (`6f32ec0`), including its full experimental Intel Arc SYCL engine, and make the updated port run on a B570 with CPU expert misses. Previously the model stopped at layer 1 because the GPU flag handshake did not complete. The verifier now captures the original mixer and MoE phases separately, waits on events for CPU work, and explicitly transfers plans, outputs and staged expert weights. The Docker launcher keeps this path enabled unless fully resident operation is explicitly requested.

The follow-up fixes align shared core APIs, preserve the original S2 bit comparison and PLE rejection guards, pair snapshot USM allocations with SYCL frees, generate the missing IQ/PLE reference fixtures, support a one-token prompt at position zero, and replace CUDA compatibility metadata and registration placeholders with actual SYCL checks and truthful pageable-memory reporting.

Validation on Intel Arc B570 10 GB / Ryzen 5 5600X / 128 GB RAM, Linux 7.0.0-38, oneAPI 2026.1.1 and compute runtime 26.35.39758.10:

- Release JIT and BMG-G21 AOT builds pass; all 25 kernel/snapshot tests pass on both.
- All 48 layers' native expert checks pass on real IQ3_S shard weights; setup tests pass 6/6.
- Captured/eager split verifier and JIT/AOT first logits are bit identical across 248,320 finite entries. Genuine host USM staging preserves the preceding DMA path's logits and generated IDs.
- Persistent generation, prompt checkpoint reuse and parked conversation restoration preserve generated IDs and logprobs, with 30 tokens actually reused.

Commands, hashes, logs, scope and fixture instructions: `bench/results/2026-10-05-sycl-upstream-arc/README.md`. These are correctness checks, not a throughput result; further B570 optimization continues separately.
