<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# Full decode-cache physical lease and GPU residual qualification

The candidate releases all physical decode-cache extents during prefill, then refills from RAM and publishes them for decode.
A two-layer/two-cycle actual-device fixture and fresh GEN1024/32768 in one process completed with normal QUIT.
All-RAM and GPU16384-prefix/RAM-tail configurations passed separately. The32K hybrid actually allocated671088640 GPU bytes and671047680 RAM bytes.
Configured context262144, int8 KV resident32768,8K chunks,FP32, explicit700 MiB reserve,max_new1 and no reuse.
All512 experts were uploaded once per each of48 layers, processing four chunks.

The retained32K row containing all248320 logits matched the previous processing order byte for byte for both candidates.
This covers only fixed-history position32767, not whole-model capability,256K or RESTORE.
Individual receipts record GPU pre/post checks, same boot, full SHA checks of12 model inputs, providers, fault-free checks and normal owner closure.
Free/contrib-llvm builds, four CPU/ASan/UBSan configurations, server lifetime and actual-parent static checks passed.
The initial four static findings retain failed status. Multiplication is widened first; two reachable postlease failures now failstop before unwind.

Native GEN diagnostic times do not decide speed. Next:4K workspace qualification and unchanged upstream bench.sh --long --runs3.
Full physical262144 input,RESTORE, capability quality and adoption remain pending. New4K and decode ledger candidates have separate worktrees.
R412 inspected only the parent; R413 audits all18 actual source pins and is the relevant4K audit.
R413 says five but enumerates six files. R415's missing-registry claim is a lookup error; the actual L registry hash is verified here.
R416's eight-expert traffic bound applies to NT1 decode, not a32K prefill route union. Source+copy and nested phase spans are not pure PCIe time or additive critical-path fractions.
A path/hash/reason manifest replaces closed duplicate logs with structured evidence. Immutable receipts stay for current controller consumers.
Skill discovery exists only in this project's.agents/skills, not user-wide paths. Adoption must also carry the corrected relocation from this branch's old root layout.
