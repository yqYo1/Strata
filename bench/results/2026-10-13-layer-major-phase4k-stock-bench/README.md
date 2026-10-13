<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# Official bench.sh measurement of4K layer-major prefill

Unchanged upstream bench.sh --long --runs3; actual32049 input tokens,24output,noReuse.
Quiet prefill654.7/667.6/669.8 token/s,median667.6; decode8.7/10.4/11.1,median10.4.
Debug medians675.4/9.6 are diagnostic only and are excluded from performance comparisons.
Both arms completed normal QUIT,pre/post GPU checks,same-boot/source/model/provider identity and fault checks with no new faults.
A faster observation alone does not isolate the contribution of4K or settle decode adoption.
Full physical256K,RESTORE and whole-model quality remain unqualified; adoption is pending.
Quiet measured source:db088ff24e67e6a4566566dfc4762c63c55bf343.
Next: separately qualify24576-to48 checked copies and measure decode service times.
R428 confirms cumulative resident counts are not an expert union; selective upload needs an exact union census.
A manifest replaces closed successful raw logs with individual results,commands,environment,progress and fault evidence.
