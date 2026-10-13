<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Actual serve first-row logit capture

Full free/contrib-llvm builds, CPU Release and ASan/UBSan, and server phase lifetime
checks passed. All five changed files were compared with actual parent6035c4f;
there were no new lint findings.

On B57010GiB / Ryzen5600X /128GiB RAM, fresh1024/32768-token requests each produced
one token with8192 chunks and phase/head lease/scratch/gated enabled. Both captured
finite FP32 logits for all248320 vocabulary entries at P-1, actual T=1 and reuse=0.
Argmax matched the emitted ID. The engine closed normally. GPU checks before/after execution,
model/provider hashes and fault checks passed. This qualifies the producer and
lifecycle only, not fixed-history distribution equality,NLL,full physical262144,
RESTORE or speed. Default OFF remains.

Registry v194 archives R396–R401 and restoration/route-ledger prototypes v1/v2, totaling958 reports.
R398 distinguishes retaining existing perchunk KV stage seed from optional persistent
stage reuse. Main rejects R401's GEN128 speed-comparison proposal: only unchanged
upstream bench.sh supplies new speed comparisons. Research proposals are not measurements.

Retain only the32K logit outside Git for actual producer registration and numerical
comparison. Retire completed short tensors, partial aliases and duplicate command
logs after manifest hash checks; preserve individual outcomes and original statuses.
Next qualify layer-major support for ordinary serve and streamed KV.
