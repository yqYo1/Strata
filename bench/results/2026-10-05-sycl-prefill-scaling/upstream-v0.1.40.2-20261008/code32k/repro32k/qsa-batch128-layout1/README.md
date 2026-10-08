# Attention batch128/layout1 on32K input

This environment-only experiment uses the same private e82fc5 executable,
Arc B57010GiB / Ryzen5 5600X /128GiB host and32K input/configuration as the
[host-only profile](../host-api-profile/README.md). The two changed settings
are STRATA_PREFILL_ATTN_BATCH=128 and STRATA_PREFILL_ATTN_LAYOUT=1. Subgroup32,
workgroup256, KV format, ordered dot/reductions/accumulation/merge, kernel
source, queue properties, graphs, drains and all other settings are unchanged.
Layout1 transposes local query storage then reconstructs the same eight
operands; it avoids that branch's float-array to float4 pointer casts. No
SG16, tensor/XMX attention, approximate selection or retirement is enabled.
Attention scratch has four times as many per-query slots. The engine shares
a region sized to the largest phase workspace, so that is not by itself a
fourfold increase in total VRAM allocation. Both startup INFO records report
830MiB free before the request; those snapshots do not measure peak usage.
The [source review](analysis/qsa-batch128-layout1-v01402-source-review.json)
records disjoint per-query scratch, uniform barriers and the127-query tail of
the8191-token final chunk. Source review is not a general runtime UB proof.

The first logged/validated32K run and two fresh32K captured repeats match
all66prefill state parts, all248,320first-head floats,64 output IDs and every
logprob with the completed default32/layout0 control. These durations are not
speed evidence. Only after those gates, four fresh processes run a clean ABBA
comparison: default/128-layout1/128-layout1/default. All read32,768 tokens,
use8192-token chunks, context33024, int8 KV, normal MTP4 and64 greedy outputs.
Debug logs, validation, state/head dumps, transfer timing, profilers and extra
waits are absent. Both implicit-conversion override and MKL CNR are absent.
An uninterrupted owned GDB/PTY observer is common to both settings. Every
clean job matches all64IDs and all logprobs; all seven jobs exit normally,
complete owned cleanup and record no new xe fault.

| Setting | Prefill token/s | Decode token/s |
| --- | ---: | ---: |
| qsa-default, run1 | 408.180 | 16.810 |
| qsa-default, run2 | 408.421 | 15.888 |
| qsa-default, mean | 408.300 | 16.349 |
| qsa-batch128-layout1, run1 | 408.140 | 16.771 |
| qsa-batch128-layout1, run2 | 408.573 | 16.672 |
| qsa-batch128-layout1, mean | 408.357 | 16.722 |

The two-run mean changes by+0.014% for prefill and
+2.279% for decode. Two runs per setting cannot establish
a small improvement outside observed variation, and a prefill-only setting
does not by itself explain decode variation. See the
[clean sequence](analysis/qsa-batch128-layout1-v01402-clean-sequence.json)
for every duration, environment, gate and ordering. These are measurements
of private settings, not an upstream-alone equivalence claim.

No production executable or default setting changes. Full262,144-cell
occupancy/repeat/restore/clipped-tail/refusal/later-valid gates and PP1000/TG70
remain open. No general hang-prevention claim is made. Large API/state/head
payloads remain private with hashes; exact controllers, configuration,
fixture, protocol, process ownership and journal records are public.
