<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Optional copy queue: tiny GPU qualification

On Arc B570 (10 GiB), source-built intel/llvm 7.1.1 with UR Level Zero v2,
Main passed six separately owned tiny GPU cases: free generic/native,
free native with forced tag policy, free service-profiling generic fallback,
and contrib-llvm generic/native. Each completed normal drain and process exit,
had exact copied payloads/canaries and queue dependencies checked, and passed
fresh integer GPU health checks and kernel-fault windows before and after.
The source and binary identities are those of the preceding host qualification.
No GPU reset, service changes, model input or speed measurement was performed.

The three native cases actually selected COPY-only group 1, ownership::keep.
Successful API handles associate both 1056-byte and 1024-byte copies with the
same immediate list, followed by host synchronization and one destruction.
No native-list kernel launch was recorded. The selected calls are preserved;
this normal-close observation does not prove raw UR import-exception cleanup.
The service-profiling case correctly chose the generic queue and is not counted
as a native treatment. Native event markers stayed barriers under forced tag.

The initial controller failed after a healthy probe because its expected
provider set omitted the validation-layer library. No queue fixture ran in
that attempt. Corrected controllers verified all eight actual provider paths
and hashes, including the requested validation layer. This failed outcome is
preserved. Tiny compute API traces have no unexpected errors; NOT_READY query
results are counted separately. xpu-smi discovery's Sysman permission and ECC
capability diagnostics are retained and are not compute-health results.
Two compact-evidence preparation failures are also preserved: a pre-existing
empty output directory and incorrect classification of Sysman diagnostics.

Full successful API/polling histories were extracted and verified before
duplicate raw command files were retired. Original receipts temporarily remain
under a 4-MiB exception for the active SourceOnly model-controller consumer;
review at its FINAL. The compact qualification receipt is the future admission
input. The model's Prefill/Stager lifetime, >=32K speed, task quality, RESTORE and
physical 262144-token input remain unqualified. Default copy treatment stays OFF.

R336 confirms generated-text equality is a narrow diagnostic, not a universal
model-quality gate. Fixed-history likelihood and held-out task evaluation remain
open. R337 identifies an independent phase-cache VMM context-granularity guard
gap to fix before its GPU execution. The phase-cache caller packet is source
only and untested. Next: bounded >=32K copy-queue model diagnostic, then balanced
quiet repeats with separate prefill/decode decisions. Objective remains active.
