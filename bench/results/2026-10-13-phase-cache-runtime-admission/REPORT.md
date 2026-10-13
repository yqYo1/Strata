<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Phase cache admission checks

The default-OFF phase expert cache retires prefill workspace before mapping,
filling, reading back and comparing extra expert residency for decode. The next
fresh GEN retires graph readers before removing those mappings. KV, GDN and PLE
owners and numerical formats remain the same. Uncertain map/unmap state stops
the process and retains ownership until exit.

On Arc B570 / Ryzen 5600X, source-built DPC++ 7.1.1 builds in free and
contrib-llvm passed, together with CPU ownership tests, a separate host
ASan/UBSan build and two server phase lifetime tests. The preceding full server
suite passed249 tests. All changed-source lints were compared with an actual
pristine baseline, with zero new findings. Existing mypy, CMake and analyzer
findings keep the raw lint exit at1. Earlier failures, including the v4 build's
signed comparison error, retain their original failed outcomes.

The tiny actual VMM result is in summary.json and receipts. It does not qualify
the model, verifier graphs, speed, quality or full physical262144 input. No real
provider map/unmap error injection was performed. Next, verify two fresh GENs
and normal QUIT before measuring with unchanged upstream bench.sh. Exact prose
identity is not a universal quality gate.

Parallel research is retained in registry v186, including original reports and
corrections. Main's errata are in summary.json. Head lease and full-context probe
are archived as source deltas only; their builds, runtime and physical memory
effects are unverified. Closed-run duplicate logs retire only after evidence
and hash verification, under the deletion manifest. Original pinned receipts
remain unchanged.
