# Exact selection methods for Strata QSA

Retrieve this only after a representative stage profile makes QSA selection a plausible cost. It is algorithm knowledge, not evidence that selection is slow or that a model route should change.

## Pinned QSA contract

The cell selector sees up to 32,768 float scores and emits `min(n_kv, 2051)` cell IDs in ascending ID order. Its ordered key canonicalizes signed zero, ranks NaNs below negative infinity, and breaks equal scores by the lowest cell ID. Block selection scores each four-cell block once; complete blocks weigh four selected cells and a partial tail weighs its valid-cell count. It uses the same ordered-key boundary rule and expands selected blocks into ascending cell IDs. Keep this weighted-tail behavior in any exact alternative.

Current SYCL selector families in the pinned source are:

- Per-cell and default block selection: one 256-item workgroup per row, 32 threshold-bit rounds, followed by strict-greater/equal scans and ordered emission.
- Optional block `Mapping == 1`: eight 4-bit histogram rounds over the 32-bit ordered key, then weighted boundary repair. `STRATA_SYCL_SELECT_RADIX4` means four-bit digits, not four rounds.
- Optional cyclic `Mapping == 2`: 32 threshold-bit rounds with cyclic lane assignment.

Every extra key pass rereads scores and performs group-wide reductions. Histogram candidates add reduction work. Hierarchical top-K avoids those serial key passes only by retaining enough candidates per partition; scratch grows as partitions × retained K, which can exceed the final ~2,051 IDs. Bitonic/merge concepts are plausible for small K, but comparator shuffles, local state, SLM/register pressure, and subgroup layout need measurement on Xe2. Persistent/distributed rows can help skewed batches but add coordination, scratch and queue boundaries. None is a directly portable SYCL kernel by virtue of the cited CUDA results.

## External design references

- [Shanbhag, Pirk, and Madden, Efficient Top-K Query Processing on Massively Parallel Hardware (SIGMOD 2018)](https://anilshanbhag.com/static/papers/gputopk_sigmod18.pdf): GPU top-K families and bitonic top-K up to K=256 on the tested NVIDIA setup; not a B570 result.
- [RadiK, Scalable and Optimized GPU-Parallel Radix Top-K Selection (ICS 2024)](https://doi.org/10.1145/3650200.3656596): scalable radix design and input-distribution counterevidence; not an implementation to assume portable.
- [FlashInfer MSA top-k API](https://docs.flashinfer.ai/generated/flashinfer.msa_ops.msa_topk_select.html): fixed K=16, ascending IDs and `-1` padding, with near-tied boundary identity intentionally unspecified. This does not preserve Strata’s tie rule unchanged.
- [FlashInfer DeepSeek fused router API](https://docs.flashinfer.ai/generated/flashinfer.fused_moe.fused_topk_deepseek.html): an example of grouped router semantics and documented caps, not Strata’s canonical 512-expert/top-10 route. DeepSeek/35B execution remains out of scope. Use documentation only; check repository license and pinned source before code reuse.

## Hardware limits and retrieval gate

Pinned QSA scoring kernels request subgroup size 32, while the 256-item selection workgroup does not request a subgroup size. SYCL leaves subgroup layout and scheduling implementation-defined. Treat 16-vs-32 subgroup mapping, register/SLM footprint, score traffic, scratch traffic, occupancy, and queue cost as unmeasured until checked on the actual device. QSA scores and IDs are device-local in these paths, so PCIe is not a selector cost absent a host crossing.

If profile evidence makes QSA selection material, retrieve per-kernel row lengths/K, actual route, subgroup/register/SLM evidence, score bytes and queue boundaries. Then compare one radix candidate and one local/hierarchical candidate against the current implementation, preserving NaN, signed-zero, tie and weighted-tail behavior. The existing source safety and prefill-entry audits are separate evidence and should not be repeated as algorithm research.

Tags: `qsa-cell-radix-threshold`, `qsa-block-weighted-rank`, `float-total-order-nan-zero-ties`, `b570-sycl-subgroup16-32`, `selector-score-reload-vs-candidate-scratch`, `selection-queue-boundaries`.
