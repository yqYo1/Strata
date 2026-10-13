<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# Actual4K workspace and full32K GPU residual qualification

Fresh GEN1024/32768 completed in one process followed by normal QUIT.
Fixed4K chunks, full physical cache/head lease,FP32,int8 KV resident32768,context262144 and explicit700 MiB reserve.
For32K,32767 residual rows occupying1342136320 bytes were actually allocated on GPU; RAM tail0. Each of48 layers processed8 chunks, with a4095-row final tail.
All248320 logits at position32767 matched the old processing order byte for byte; the same fixed-history target NLL delta is0.
This is one-position numerical evidence, not whole-model capability,full256K,RESTORE or speed qualification.

The internal prefill subtotal was55,568.811 ms, including an exposed source-fetch/checked synchronous-copy interval of14,961.143 ms.
Expert upload made24576 copies of50,292,326,400 logical bytes. The span is not purePCIe time or official TPS.
Compute starts after these checked waits complete; the measured interval motivates a separate bulk-upload prototype.

Free/contrib-llvm builds,four CPU/ASan/UBSan configurations,server lifetime,static checks,real tiny full lease and pre/post GPU/model-hash/fault checks passed.
The initial two doc blank-line findings retain failed status. Main removed only one redundant blank in each translation.
All16 code pins/binaries remained unchanged; completed code checks were reused and changed docs rechecked with all tools.
Ordinary GEN cannot consume full256K with zero output; a distinct explicit diagnostic arm is source-only.
That arm and the new bulk-upload candidate remain unbuilt/unexecuted. Callback coverage is not a physical endpoint-store proof.
Next: unchanged upstream bench.sh --long --runs3 debug closure and quiet comparison. Adoption remains pending.

Registryv197 references987 previous and7 new reports. R422 corrects Main's active8 assumption to actualK10.
982272000 bytes is logical NT1 payload, not physical DRAM/PCIe traffic or a speed ceiling.
Current skill discovery is project-only. Adoption must also carry the corrected relocation from the old root layout.
A manifest and structured results replace closed duplicate logs; original hash-pinned receipts stay for current consumers.
