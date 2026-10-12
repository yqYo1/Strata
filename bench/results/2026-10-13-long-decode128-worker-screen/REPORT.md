<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Closed long-output worker screening samples

Arc B570 10 GiB, Ryzen 5600X, free build, source-built intel/llvm7.1.1,
ordinary persistent serve, no MTP, forced segmented verification, greedy,
32768 input positions and128 generated tokens. The same binary, input, fixed
18 tasks, 144 expert-cache slots and other settings are pinned in each receipt.
The predeclared worker order is5,3,4,4,3,5,3,5,4. This boundary is incomplete:
three independent samples per worker are required before a worker decision.

| Sample | Workers | Prefill ms | Decode ms | Prefill tok/s | Decode tok/s |
| --- | --- | --- | --- | --- | --- |
| 1 | 5 | 83695.4 | 32005.4 | 391.515 | 3.9993 |
| 2 | 3 | 84247.8 | 16507.8 | 388.948 | 7.7539 |
| 3 | 4 | 87278.0 | 15103.8 | 375.444 | 8.4747 |
| 4 | 4 | 87800.9 | 13542.0 | 373.208 | 9.4521 |

Every listed request is complete, normally closed by QUIT, finite and fault-free,
with observed/reaped/empty owned sessions and fresh pre/post health checks.
They are operational speed samples. Quality stays unevaluated and unqualified;
no worker setting or new inference implementation is adopted. Full128 reported
tokens and LP5 happen to match, but generated output identity is not the general
quality criterion. Full-vocabulary fixed-history likelihood/distribution and
held-out task comparison remain separate work. Different generated histories
could also change the timing workload; that limitation is retained.

The first v4 attempt failed before model launch because Main's health parser
incorrectly required one LIBRARY row per path. The actual probe was successful:
29 mapping-segment rows described seven exact pinned libraries. v5 accepts a
bounded repeated segment representation while still rejecting missing/extra
library identities. The original failure receipt is unchanged. v5 passes eight
CPU fixtures twice plus owned normal-QUIT and STOP/cancellation mock checks;
Python lint comparison has no new findings, while original baseline findings
and their nonzero lint exits remain recorded.

R330 confirms that this actual no-MTP, no-suffix, no-lookup serve route invokes
Verifier::run with T=1 per token. R329's standalone remaining-window explanation
does not apply here. --adapt-swaps0 also excludes the adaptive-swap theory.
R328 identifies caller participation as a possible later existing-knob arm,
after the worker matrix. No scheduling or maximum-output causal claim is made.
R331 finds a strict fixed-history CPU comparator gap; its standalone scoring
route is not a qualified serve quality test. R322 corrects persistent Prefill
lifecycle; growing decode cache needs a real VMM owner and safe next-GEN return.
The two implementation packets remain immutable source-only proposals.

Original source, admission and model receipts stay byte-identical for the
ongoing nine-sample controller. Committed copies identify a public typed-hash
transformation. Duplicate successful command logs are retired only after their
required evidence is verified; each deletion has a path/hash/reason manifest.
Model/context fixtures, current source packets and provider/binary pins remain.
Full262144-input, RESTORE, whole-model quality, optimized official parameters,
contrib-llvm model qualification and the1000/70 tok/s goal remain open.
