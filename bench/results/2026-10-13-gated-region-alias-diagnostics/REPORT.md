<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Actual gated region alias diagnostics

The existing8K head release/reload and scratch alias completed two fresh GENs.
The separate12K diagnostic completed its first1,024-token request, then refused
32,768 tokens before workspace allocation:4,886,504,448 requested bytes against
4,422,336,512 free bytes. The controller also mismatched the FATAL message. Preserve
that failed status and incomplete observation; a separate closure checked process
exit, GPU health, model/provider hashes and fault records. The897,024,000-byte drop
from initial fresh free remains unattributed. There is no recorded GPU hang.

The candidate overlays FP32 gated output on the shared region of the same in-order
queue. GDN,QSA and MoE write after the last HC mix reader. Precision is unchanged.
Stock12K requested storage drops by503,316,736 bytes including padding. The exact
STRATA_PREFILL_GATED_ALIAS=1 opt-in remains default OFF.

The candidate completed fresh GEN requests of1,024 and32,768 tokens, normal QUIT and the following GPU check.
summary.json retains actual chunks,pointers,capacity ledgers,requests and exit results.
These diagnostics prove no speed,NLL,state equivalence,full physical262144 or restore.

Full free and contrib-llvm builds and CPU Release/ASan/UBSan checks passed. The first
lint found a retained unknown-owner fixture leak. Main made child exit explicit
after all assertions and rechecked the fixture in four configurations. Preserve the
original failure. All five changed files passed the no-new-findings comparison with
the actual parent. CPU helpers verify layout arithmetic/ranges,not device ownership.

Registry v192 holds939 reports. R389 v1 inspected HIP/CUDA,out of scope here; retain
it and the v2 correction for actual SYCL chooser behavior. R390 does not attribute
the897 MB to a particular cache. R391 independently audits candidate source. R393 finds that a getter over the old
unkeyed,unsynchronized chooser cannot safely narrow the bound; retain the cold bound.

Speed comparisons use unchanged upstream bench.sh --long --runs 3. Diagnostic times
are excluded. Qualify fixed-history distributions/state,restore and full physical
context,and compare8K/12K with the same upstream client before adoption. Preserve
individual original outcomes and retire only closed duplicates through the manifest.
