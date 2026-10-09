Prompt measurements mixed input-length-dependent GPU work, expert transfers and PLE reads. Add actual transfer-event accounting, phase timing and fixed-memory prompt fixtures so those costs can be inspected without subtracting overlapping timers. Add uncached resident measurements that validate complete first heads against the original CLI, including an explicit diagnostic mode without MTP.

Add opt-in prompt attention launch batching, a 16-lane layout that preserves the original 32-lane reduction order, and a bounded 256-thread integer radix selector. Larger or unsupported selection geometry keeps the original kernel. Defaults remain unchanged. On the Arc B570, Ryzen 5 5600X and 128 GB RAM with the IQ3_S model, three paired normal CLI measurements show attention settings improving 8,087-token prefill from 17.560 to 17.283 seconds. A separate three-pair comparison adds the selector: 17.278 to 17.147 seconds (468.07 to 471.64 tok/s). These runs include PLE and have no phase markers, transfer markers or preload. The records distinguish overall throughput from the local input-length slope.

Validation:

- All 28 registered tests pass in JIT and B570 AOT builds, with no skips. The selector's 28 internal cases match the original GPU implementation and an independent stable CPU sort.
- All 16 paired resident requests and six paired CLI requests match the accepted complete 248,320-value first heads and output IDs bit for bit, with finite values and one full prompt chunk. AOT 4k/8k model checks also match.
- JIT and AOT normal MTP server checks pass four requests each; checkpoint reuse and restoration preserve output IDs and finite logprobs exactly.

Retain the negative compact-XMX and grouped-FP16 studies with sources and raw results. Compact XMX fails the existing hybrid accuracy bound; grouped FP16 is slower in every tested shape and changes some bits. Neither changes production behavior. Measurements and limitations are in `bench/results/2026-10-05-sycl-prefill-scaling/README.md`.
