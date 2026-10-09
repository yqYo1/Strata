Long prompts upload the same native quantized experts again for every chunk. This adds optional compact scratch and layer-major traversal to reuse one layer's 512 experts across all prompt chunks, with FP32 residual rows stored in RAM or a complete chunk prefix kept in VRAM. Defaults retain the original traversal and workspace.

The FP16 expert path keeps the original dequantization, GEMM shapes, conversions and combine order. The implementation requires an in-order queue, a full single-GPU model and all K/V pages resident in VRAM. It restores decode's cache and callbacks afterward and saves a whole-model checkpoint only at the final coherent chunk. Aggregate diagnostics separate native expert transfers, host residual transfers and copies within VRAM.

Validation on the Arc B570, native IQ3_S model, oneAPI 2026.1 and the recorded selected driver:

- All 30 JIT tests pass. Nine compact/host cases and eight GPU-residual cases preserve every first-logit bit, every residual row and all persistent state bytes, including K/V and recurrent state. Normal and long MTP checks preserve IDs, logprobs and checkpoint restoration.
- At a fixed 65,538-position K8/V8 context with 8K chunks, single transfer-timed 32K observations change from 79.509 to 74.477 seconds (412.13 to 439.98 tok/s). The 64K observations remain about 157.1 seconds (417 tok/s): lower transfer volume does not establish an overall speed gain. Complete heads and IDs match each original control with the same chunk size.
- Real 262,146-position K8/V8 allocation, dense weights, decode cache, compact scratch and the full layer cache fit with a 1K chunk; a two-token run loads all layers and produces a finite full head. This establishes actual allocation, not full 256K prefill throughput. Larger chunk failures and sampled memory are preserved.
- The current-source B570 AOT build succeeds; repeated normal-wall comparisons and AOT GPU validation are pending GPU recovery. A stopped diagnostic exposed an xe kernel fault; incomplete runs are excluded. The requested standalone PCIe bandwidth test is compiled and will run after recovery.

Raw commands, hashes, complete-output/state comparison records, transfer reports and process memory samples are in `bench/results/2026-10-05-sycl-prefill-scaling/layer-major/`. The grouped oneMKL candidate changes output bits and is recorded as rejected.
