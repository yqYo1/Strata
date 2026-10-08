# Private host accounting: first diagnostic

On 2026-10-09 JST, Arc B57010 GiB / Ryzen5 5600X /128 GiB RAM ran four fresh32768-token inputs with64 outputs, context262144, actual8192 chunks and128 expert-cache slots. Private binary `494cf4be288595f7abf4d56412fb519155cfccf602a2a060093a1f8fcd988123` adds CPU counters and steady-clock measurements to the qualified0.1.41 engine. It adds no GPU event queries, markers or waits. This is first-use correctness evidence; API logging and state dumps exclude its durations from speed comparisons.

All four first-head logits and66 live-state parts match the qualified updated baseline under the existing live-state comparator; IDs, logprobs, MTP counts, finish reasons and repeats match. The comparator excludes documented rounded-page padding, not active state or indexer spare state. The run finishes normally with exit0, no new GPU fault, no forced cleanup and no owned survivor. The instrumented engine remains private and unadopted; its full262144 lifecycle is pending.

| Fixture | Expert copies | Expert bytes | Hypothetical seconds at6 GB/s |
| --- | ---: | ---: | ---: |
| A and A repeat | 92998 | 190240998400 | 31.707 |
| B and B repeat | 92979 | 190199219200 | 31.700 |

These are expert H2D counts for32767 prefilled tokens. They do not cover every transfer. The6 GB/s estimate is the prior approximate application bandwidth, not a new measurement. It gives a transfer-only lower bound near32 s and a transfer-only ceiling near1033 tokens/s at this byte count; actual DMA duration, contention and overlap are not measured here.

A bounded32 MiB prefix of the already completed API log pairs880 UR expert copies with880 native copies,1653299200 bytes across eight ring destinations. The UR queue and native immediate command-list creation are paired; ordinal0 reports compute/copy/cooperative flags7 and one physical queue. The runtime also reports a main blitter available. This supports investigating transfer/compute overlap using a copy-only engine. It does not establish the safety or speed of a new queue. The whole API log was neither recopied nor rescanned.

Code inspection separately confirms that `NativeHead::load` puts `output.weight` in device memory. The host-mapped `NativeEmbed` table is `token_embd.weight`, used for embedding gathers; it is not the output classifier. Moving that table is therefore not assumed to remove a full-table read from each decoded token.

The separate quiet baseline/off/on comparison is still live and is not included in this immutable archive. No speed or full-capacity adoption decision is made from this diagnostic. JSON receipts bind source, runtime, binary, fixtures, comparators and cleanup; the manifest hashes the archived files. Large logs, binaries and state/session tensors remain private.
