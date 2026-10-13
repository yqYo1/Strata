<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# Actual layer-major bulk expert upload qualification

Default-OFF bulk upload passed free/contrib-llvm builds,four CPU/ASan/UBSan configurations,server lifetime and static checks.
Real small GPU tests in both modes verified all512 experts,seven widths,physical accounting and declined-layout per-expert fallback.
Actual fresh GEN1024/32768 completed one owned Host-USM copy per each of48 layers followed by normal QUIT.
Each GEN copied50292326400 logical bytes in48 calls,zero fallback; largest single copy1363148800 bytes.
All248320 FP32 logits matched the old4K path byte for byte at two fixed-history positions.
This is two-position evidence,not whole-model capability,full256K,RESTORE or speed qualification.

The initial controller newline-comparison failure and subsequent11 static findings retain failed status.
Main corrected zero/overflow-safe cache sizing,active-array extent,fixture widening/side-effect sequencing and doc blank lines.
Full rebuild/CPU tests and fresh all-tool checks on five corrected files passed; completed checks on17 unchanged files were reused without suppressions.
Pre/post model hashes,same boot/providers,GPU health and fault checks passed. Copy wall spans are not purePCIe time.
Next: unchanged official bench.sh debug closure and matched quiet3runs. Adoption remains pending.
