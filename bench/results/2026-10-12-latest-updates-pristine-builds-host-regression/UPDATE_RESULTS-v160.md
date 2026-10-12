<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Latest upstream and XeStrata updates on Arc B570

Official v0.1.43 `7695859` and XeStrata xe0.1.42 `da3d426` were built and measured without
local engine patches. Required upstream functionality was imported and pushed on
`sync/xe-v0143-nonfinite-20261012` at `0691bcdf`. This is selective integration into the
existing fork; its engine version remains xe0.1.40.2.2. The original workspace branch remains unchanged.

All rows use native Qwen3.8-Flash-Next IQ3_S, 32,768 input IDs and16 generated IDs on
Ryzen5600X / Arc B57010GiB. Compiler, runtime and prompt routes differ: this is not a
controlled implementation speed ranking. Exact commands, environment, individual samples,
fingerprints, source/runtime/binary and normal-close/fault receipts are in the JSON results.

| Source / setting | Prompt route | Prefill t/s | Decode t/s | Scope |
| --- | --- | ---: | ---: | --- |
| Pristine XeStrata xe0.1.42 | Batched8192 | 217.61 | 10.41 | One diagnostic; LLVM7.1.1 free;29/144 slots filled |
| Pristine official cache128 | Serial windows8 | 19.07 | 6.29 | One diagnostic; icpx2026.1;144 slots |
| Pristine official cache512 | Serial windows8 | 17.77 | 8.48 | One diagnostic; icpx2026.1;624 slots |
| Selectively integrated update | Batched8192 | 371.80 | 7.86 | Three clean samples; LLVM7.1.1 free; nonfinite detection on |

Required imports include server memory admission/session SAVE, context overflow, reasoning/
launcher/setup fixes, cache opt-out and logit-bias validation. Unsupported native logit bias
returns400. The server suite has516 pass/11 skip; focused helpers, free/contrib-LLVM CPU,
host sanitizer and small real GPU integer/nonfinite checks also passed. Actual32K runs
closed normally with fresh health and no observed GPU fault. Nonfinite detection stays:
three decode pairs did not establish a reproducible benefit from removing it. The whole
37,462-ID held-out input matched both detection arms'16 IDs and reported LP5.

Cache512 was slower in prompt and faster in decode in its single diagnostic sample.
Its output IDs and reported LP5 differ from cache128, consistent with changed CPU/GPU
placement but not a task-quality parity proof. No cache default or inference implementation
was selected from that screen. Best-tuned upstream parameters, full262,144-token physical
context, RESTORE, full-vocabulary logits and task-quality parity remain unqualified.
No CUDA/HIP change was selected. Read-only research continues; histogram/raw-sample
proposals are unapplied source packets. See registryv160 for reviewed research and corrections.
