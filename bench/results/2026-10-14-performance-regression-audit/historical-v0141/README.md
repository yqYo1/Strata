# Upstream 0.1.41 snapshot integration

The upstream main snapshot `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` reports 0.1.41. It was built and measured without local patches, then integrated at engine commit `1eb89482a4afd20277ae0405780ed4f8eb98eb20`, retaining the previously qualified SYCL completion, ownership, typed-access, state and output corrections. This is a pinned main snapshot; no release tag is claimed. Only SYCL and the common CPU/server paths were evaluated. HIP and CUDA were outside this comparison.

Measurements were made on 2026-10-09 JST with Arc B570 10 GiB, Ryzen 5 5600X, 128 GiB installed RAM and Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S. Every speed input contains 32768 tokens and every output 64 visible tokens. Each build/configuration has two processes and eight fresh inputs; first reads are separate from the six later full reads. Logged correctness runs are excluded from speed tables.

| Configuration/build | Later prefill tokens/s | Later decode tokens/s |
| --- | ---: | ---: |
| Default actual 6144: qualified old control | 341.035 | 16.691 |
| Default actual 6144: unmodified upstream | 348.887 | 17.135 |
| Default actual 6144: integrated update | 361.693 | 16.456 |
| Tuned actual 8192: qualified old control | 427.452 | 16.586 |
| Tuned actual 8192: integrated update | 449.698 | 16.635 |

The [default comparison](default32k/README.md) reports the raw upstream's different actual cache admission,144 slots versus128, and output/MTP differences. The [matched 8192 comparison](tuned32k/README.md) enforces 128 actual slots and identical settings; all cross-version output IDs, logprobs and MTP counts match. It measures +5.204% later prefill and +0.298% later decode. The decode variation does not establish a speed change. The default comparison's small negative decode difference does not reproduce as a regression in the matched previously tuned configuration.

[Full physical 262144 qualification](physical256k/README.md) passes two fresh full inputs through cell262143, all saved tensor bytes against DD5 including inactive MTP, repeated saved-byte equality, actual full disk restoration, clipped tails, capacity refusals and a later fresh 32K read. It completes normally without a new GPU fault or surviving owned process. Engine-version fingerprint differences are checked separately; no tensor bytes are waived.

All twelve changed common server files match upstream byte for byte, including request hardening, Responses support, parallel/slot allocation, tool-name handling, vision-busy handling, telemetry and the frontend. The 593-case suite passes 585 with eight skips. [The five real-tokenizer cases](server-tokenizer-cpu/README.md) subsequently pass with no skips; only three Windows-specific cases remain unexecuted. The SYCL launcher also passes ten CPU cases covering explicit runtime zero values, retired-path rejection and literal argument forwarding.

These results support keeping the updated CPU/SYCL engine and every server feature change, with the qualified SYCL safety corrections and the launcher fix that preserves explicit `EnableDirectSubmission=0`. The measured binaries contain no QSA-reduction, new copy-queue or host-accounting experiment. Integration and evidence are kept on the review branches; main and production binaries/services were not changed. Each subfolder supplies terminal receipts, source/runtime bindings, fixtures where applicable, per-read statistics and hashes. Large API logs, binaries and model/state/session tensors remain private.
