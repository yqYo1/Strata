# AVX2 sign expansion: rejected candidate

On Ryzen 5 5600X, replacing two source-level 128-bit shuffles and a lane insert
with one 256-bit shuffle gives no useful gate/up improvement. The precise
icpx 2026.1 optimizer already emits the same sign-helper instruction sequence
for both forms (`sign-helpers.asm`). The complete inlined row code differs;
the four-token candidate is slower. No production kernel is changed.

The CPU-only harness uses real Qwen3.8 Flash Next GSQ-RCO IQ3_S GGUF slices and
the engine's pinned ggml activation quantizer. It compares all FP32 bits,
including untouched output guards, for one through eight tokens. It covers all
90 expert samples across the pack's 30 IQ3_S/IQ2_S layers, six synthetic i-quant
formats, three block-aligned widths, five byte offsets, partial output ranges,
and 165,602 sign patterns independently checked against a scalar definition.
All checks pass; the total float comparisons are 7,591,680 including guards.

One thread is pinned to logical CPU 2. Nine paired gate/up timings alternate
order. One expert measures reuse; 96 different real layer-17 IQ3_S experts
occupy 135,168,000 bytes, above the 32 MiB L3. Dimensions are 2560 / 640.
Compiler options are `-O3 -fp-model=precise -mavx2 -mfma -mf16c`; no fast math.

For the 96-expert dataset, median paired original/candidate time ratios for
one, two, three and four tokens are 0.988, 0.990, 1.013 and 0.922. Whole-engine
TG is not measured. A separately stalled GPU control process is still alive;
these CPU measurements cannot establish a decode speed improvement.

`summary.json` and `run.jsonl` retain every pair. The initial harness compilation
used the wrong TensorInfo member name; its error log is preserved. The first
six-layer selection lacked IQ3_S timing data, so its incomplete record is also
preserved separately. The final harness checks all 48 layers and completes.

The original implementation and the local pinned llama.cpp reference
`3cf03257f219afbe7334045ff7c6a06ac68c627d` both use bit-selector sign expansion.
The pinned CPU source was inspected directly; no newer upstream behavior is
claimed. Build and source-generation scripts record the local compiler, model
and checkout paths. Real model tensor bytes and the probe executable are not
included in Git.
