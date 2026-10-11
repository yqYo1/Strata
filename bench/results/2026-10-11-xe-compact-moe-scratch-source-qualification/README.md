# MoE temporary-storage source and CPU qualification

The opt-in compact MoE scratch implementation was committed and pushed to the own fork.
Default remains OFF. FP16 gathered inputs/activation output and FP32 GU reuse at most T rows;
the route-sized FP32 Down output and all arithmetic precision remain unchanged.

Both free and contrib-llvm full builds passed with WERROR. Each mode's release and
address/undefined host-sanitizer CPU selection passed13 tests and skipped the AVX512-only
expert_multi_test. Standalone compact routing/count/release fixtures passed release and
ASan/UBSan. These checks did not execute a GPU or model. Actual both-mode mandatory lint
has no new findings against the unchanged parent; existing warnings remain.

Original failed C++17 host-command and original failed lint-path comparison receipts stay
FAILED. Their separate corrections change only Main's qualification commands/comparison.
The actual source content equals all42 pins of the uncommitted builds; publication did
not rebuild their banners. GPU output/lifetime parity, physical VRAM/8192 fit, performance,
RESTORE and full262144 occupancy are still unqualified. The separately corrected source-v2
diagnostic controller passed actual CPU admission/prerequisite/output/QUIT fixtures and
CLI refusal checks. The failed outer-file-limit CPU-root-v1 is preserved. Main next resolves
the actual admission/publication bridge, then compares N64,2049,32768 OFF/ON at actual6144.

[Individual actual receipts](receipts/) preserve commands, explicit environments,
CPU outputs, toolchain/binary/source identities and owners. [Separate Main corrections](main-research-corrections-v1.json)
fix research arithmetic and workload scopes without rewriting reports or measurements.
The source-only GDN/QSA lifetime aliases and layer-major proposals are not implemented.

本家の測定は箱出し設定だけではないが、本家単体の最適設定探索は未完了。
prefillとdecodeを個別に選択し、32K以上の入力で選択後の独立検証を行うTODOを維持する。

[Source diff](source.patch.lossless.json) preserves the exact original patch as base64 JSON.
The original publication failed only on the patch archive context-line whitespace and stays FAILED.
No source, command result, or research report was rewritten to correct that representation.
