# Integrated 0.1.41 snapshot: actual 8192 comparison

Measured on 2026-10-09 JST on Arc B570 10 GiB, Ryzen 5 5600X and 128 GiB installed RAM. This compares the previously qualified private DD5 binary with the integrated upstream update at engine commit `1eb89482a4afd20277ae0405780ed4f8eb98eb20`. The [unmodified upstream](../default32k/README.md) has a separate same-day default 6144 comparison. [Full physical 262144 qualification](../physical256k/README.md) passed before this quiet sequence started.

The control binary is `dd5efb9002167481d180f370b86fb6978d724f37745f7e4a923432296725ad34`; the integrated binary is `86972697ecb1903750d202f8be29a7a1da351eb3415678c3e3b6af790804d1c8`. The old observed worktree commit `452044ab2546185bebad51c2cb16b9375b6683e3` marks its review/archive boundary, not a source-identical rebuild of its private corrections. Prior build receipts identify the compiled inputs.

## Matched method

The model is Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S. Both builds use context 262144, int8 KV with 32768 resident cells, actual 8192 prefill chunks, 128 actual expert slots/325 MiB, five CPU pool workers, PCIe fraction zero, no prefill borrowing, MTP four and greedy 64-token outputs. Compact mode is two, attention layout one, owned KV staging enabled and KV prefetch disabled. Prompt caching uses interval 262139, root zero and turn token -1; every measured request must nevertheless be fresh, with RESUME 0 and REUSED 0.

Process order is control, integrated, integrated, control. Each process reads A/B/A/B with 32768 tokens each. First-use allocation/capture costs remain in the first-read group. Later reads execute the entire prefill again; they do not reuse a prompt prefix. There are two processes per build: two first reads and six later reads per build, sixteen quiet reads in total. API logging, parameter validation, profiler, interposer, phase timing, extra phase waits and head/state dumps are disabled for these timings. Runtime safety settings remain the same across builds.

The [terminal sequence](sequence/record.json), SHA-256 `606e8b8befda5e3720f4197bc4a88894087dcc0ded9b3732fcc7c1300fafe22c`, binds all four process receipts. Actual progress is 8192, 16384, 24576, 32767 on every request. Requested argv excluding the executable and actual target environment are equal. Exact executable identity, startup geometry, PID/start ticks, timings and numerical checks remain in each receipt. Both fixtures are included once and were byte-identical across all processes.

## Results

Pooled tokens/s, calculated as total tokens divided by total corresponding engine duration:

| Build | First prefill, n=2 | First decode, n=2 | Later prefill, n=6 | Later decode, n=6 |
| --- | ---: | ---: | ---: | ---: |
| Qualified old control | 393.229 | 12.316 | 427.452 | 16.586 |
| Integrated update | 408.648 | 16.088 | 449.698 | 16.635 |

The integrated update's later prefill is +5.204% and later decode +0.298%. Later prefill per-read ranges do not overlap in this sample. Decode has substantial per-read variation; these two processes per mode do not establish a decode improvement. The old control's first decode readings were 9.815 and 16.526 tokens/s, so its low pooled first-read rate is reported separately and does not support a repeatable 30.6% decode gain.

Later per-read medians and ranges:

| Build | Prefill median [min, max], n=6 | Decode median [min, max], n=6 |
| --- | --- | --- |
| Qualified old control | 427.442 [427.242,427.677] | 16.262 [15.018,18.342] |
| Integrated update | 449.674 [449.491,450.083] | 16.182 [15.581,19.060] |

[summary.json](summary.json) and [summary.csv](summary.csv) preserve every per-read timing, first/later pooled rate, median and range. Loading is outside engine prompt/decode durations. Six later reads are not six independently launched processes. TTFT, answer quality, peak memory and isolated CPU/SYCL kernel contributions were not measured.

## Correctness and decision

All sixteen reads passed their qualified per-build output, logprob and MTP-count references and repeated-output checks. All four cross-version A/B/A/B comparisons match output IDs, logprobs and MTP counts. All four processes completed normal QUIT with exit zero, no forced cleanup, no surviving owned process and no new kernel GPU fault.

This matched configuration supports retaining the updated CPU/SYCL engine together with the upstream server changes. It retains the qualified completion, pointer-lifetime, typed-access, canonical-spare and visible-output corrections; no profiler/QSA-reduction/copy-queue experiment is adopted. The combined measurement does not attribute the gain to one kernel. The completed physical lifecycle and server checks remain separate evidence. Main, production binaries, services, drivers, packages and global settings remain unchanged; the integration is kept on the review branches.

The archive includes small terminal receipts, protocols, progress messages, fixtures and original controllers. Their original local model/reference prerequisites remain explicit. Large stderr, binaries and state/session tensors remain private. Every included file is hashed in the manifest.
