<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# Official bulk transfer and nonbulk repeat results

Unchanged upstream bench.sh --long --runs3; actual32049 input tokens,24output,noReuse.
Bulk prefill693.5/691.1/701.1 token/s,median693.5; decode10.1/7.9/8.1,median8.1.
Nonbulk repeat prefill666.5/666.2/668.0,median666.5; decode7.4/9.5/8.5,median8.5.
Initial nonbulk decode8.7/10.4/11.1. Control decode also declined; a bulk-specific regression is not established.
All individual samples remain. Prefill improvement alone does not justify decode adoption; adoption is held.
Both arms completed normal QUIT,pre/post GPU health,same boot,source/model/provider identity and fresh fault checks.
Full initial builds and dependency rebuilds compiled all ABI-dependent translation units; measured binary hash matches.
Next Main qualifies CPU-service instrumentation to separate GU,quantization and Down.
Physical final256K stores,RESTORE and whole-model quality remain unqualified; configured262144 alone is not a proof.
Branch-specific history is recoverable by original commit from the prior preservation merge; history preservation is not inference adoption.
