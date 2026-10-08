# Upstream 0.1.41 snapshot: raw and integrated 32K comparison

Measured on 2026-10-09 JST on Intel Arc B570 10 GiB, Ryzen 5 5600X and 128 GiB installed RAM. This compares the unmodified upstream snapshot `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`, the integrated update at `1eb89482a4afd20277ae0405780ed4f8eb98eb20`, and the previously qualified private DD5 binary. The upstream version string is 0.1.41; no release tag is claimed. The raw upstream tree has no local patches.

The control is binary `dd5efb9002167481d180f370b86fb6978d724f37745f7e4a923432296725ad34`, reporting 0.1.40-sycl. Its observed worktree commit `452044ab2546185bebad51c2cb16b9375b6683e3` records the earlier review/archive boundary; it is not a source-identical rebuild of the private DD5 corrections. The [control build record](checks/control-build.json) and earlier qualification archives identify those compiled inputs. The raw upstream binary is `a4f6b30f3765140a0d03855586f27a57081b8c76b57705661328e96354344468`; the integrated binary is `86972697ecb1903750d202f8be29a7a1da351eb3415678c3e3b6af790804d1c8`.

## Hardware, model and settings

Ubuntu 24.04.5 LTS, kernel 7.0.0-38-generic, NEO 26.31.39395.14, IGC 2.41.5, oneAPI 2026.1.1 and Level Zero v2. The builds use the same project Release options and ggml commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`; no device-specific compiler experiment is included. Only SYCL was built and measured. HIP and CUDA were not tested. The unrelated embedding service stayed inactive.

The later read-only [PCIe topology capture](checks/upstream-v0141-hardware-and-pcie-topology-readonly-20261009-v1.json) observed 16 GT/s x4 on the external GPU link at `0000:03:00.0` and its upstream AMD bridge path. The GPU leaf and internal Intel bridge report different link metadata. The capture was made during the separate logged physical test; it is not a new application bandwidth test or telemetry sampled inside these timed reads. No PCIe setting was changed.

The model is Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S, loaded from the two-part native GGUF and existing expert pack. Exact model paths, pack path, profile, argv and environment are in each run receipt. Context is 262144, int8 KV with 32768 resident cells, requested prefill 8192, requested expert cache 128 with the per-layer profile, five CPU pool workers, PCIe fraction 0, no prefill borrowing, MTP 4, minimum draft probability 0, greedy output, no suffix draft, no adaptive swaps and prompt cache 0. Each input contains 32768 tokens; each output contains 64 visible tokens.

The actual default chunk is 6144 in all three modes. The recorded progress positions are 6144, 12288, 18432, 24576, 30720 and 32767. The actual cache differs despite the same requested setting:

| Build | Actual expert slots | Cache MiB |
| --- | ---: | ---: |
| Qualified old control | 128 | 325 |
| Raw upstream | 144 | 281 |
| Integrated update | 128 | 325 |

The integrated tree retains the previously qualified host completion, uniform DPCT profiling definitions, typed KV access, canonical indexer spare and visible-output commit corrections. It includes every changed common server file byte for byte from the pinned upstream, with new SYCL server support retained. The [server parity check](checks/server-feature-parity.json) covers all twelve changed common server files. The [server test record](checks/server-tests.json) reports 593 tests, 585 successes, eight skips and exit zero. The separate launcher correction preserves explicit runtime/driver values of zero and numeric first-chunk/stager controls; [ten CPU cases](checks/server-wrapper-cpu10cases.json) exercise the actual script with a Docker stub. No production container was started by these checks.

## Method

First-use logged checks precede quiet timings. Their flushed UR, Level Zero and Strata logs are excluded from speed results. Quiet processes have no API logging, validation layers, profiler, added phase waits, head/state dumps or interposer. Runtime safety settings are the same: `NEOReadDebugKeys=1`, `EnableDirectSubmission=0`, persistent SYCL cache disabled, Level Zero v2 explicitly selected and copy offload disabled.

The quiet process order is control, raw, integrated, integrated, raw, control. Each process performs four fresh A/B/A/B reads. Both RESUME and REUSED must be zero, and the full prefill prefix must be processed. The first read of each process is separate from its later three full reads. There are two processes and eight quiet reads per mode: two first reads and six later reads. Loading is outside engine prompt/decode durations. The pooled rate divides the sum of tokens by the sum of the corresponding engine durations.

The [completed sequence](sequence/record.json) binds each terminal receipt to its SHA-256. Included run receipts retain every timing, progress line, output ID, logprob, MTP count, actual executable identity, PID/start ticks, runtime settings, normal exit, cleanup and kernel fault check. The A/B token inputs accompany the quiet runs. The controllers retain their original local pack/reference paths and prerequisites; those paths must be supplied on another machine. Large diagnostic stderr, executables and model/state/session images remain private; completed stderr sizes and hashes are in the receipts.

## Results

Pooled rates in tokens/s:

| Build | First prefill, n=2 | First decode, n=2 | Later prefill, n=6 | Later decode, n=6 |
| --- | ---: | ---: | ---: | ---: |
| Qualified old control | 324.564 | 15.622 | 341.035 | 16.691 |
| Raw upstream | 331.587 | 16.705 | 348.887 | 17.135 |
| Integrated update | 343.166 | 15.274 | 361.693 | 16.456 |

The integrated update's later prefill is +6.057% and decode is -1.407% against the control in this configuration. Two processes per mode do not establish the cause or repeatability of the small decode difference. Raw upstream's later prefill is +2.302% and decode +2.659%, with its different actual cache admission and output/MTP behavior included in that measurement.

Later per-read medians and ranges in tokens/s:

| Build | Prefill median [min, max], n=6 | Decode median [min, max], n=6 |
| --- | --- | --- |
| Qualified old control | 341.103 [340.764, 341.172] | 16.661 [15.759, 17.683] |
| Raw upstream | 348.887 [348.748, 348.999] | 17.179 [16.672, 17.482] |
| Integrated update | 361.712 [361.548, 361.765] | 16.414 [16.013, 17.079] |

[summary.json](summary.json), [summary.csv](summary.csv) and [per-read-statistics.json](per-read-statistics.json) preserve pooled rates and all first/later per-read statistics. TTFT, answer quality and peak memory were not measured here.

## Correctness and scope

Every successful process completed four fresh inputs, all visible output/logprob checks, own repeated-output checks and normal QUIT with exit zero. No forced cleanup, surviving owned process or new kernel GPU fault was recorded. The integrated update matches the control's output IDs, logprobs and MTP counts on all four reads. Raw upstream's A output differs; its B output IDs match but logprobs and MTP counts differ. These differences are reported in the summary and are not claimed to be numerical equivalence.

The first pure diagnostic controller stopped after a completed request because its Python logprob parser raised an IndexError. Its [excluded failure receipt](failures/pure-diagnostic-parser-r1/record.json) is retained. It did not record a GPU fault; owned cleanup completed. The corrected v2 controller and its completed pure diagnostic r2 are included. Failed/controller or logged runs do not contribute to the throughput table.

This folder supplies the completed default-6144-chunk comparison. The previously tuned actual8192 configuration and full262144-cell lifecycle require their own numerical, saved-state, repeated-read, disk-restoration and capacity evidence. This report alone is not an adoption decision or proof of the updated binary's full-context lifecycle. No main branch, production binary, driver, package or global setting was changed by the measurements.
