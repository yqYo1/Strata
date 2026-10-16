# Separate decode comparison

On2026-10-09 JST, Arc B57010 GiB / Ryzen5 5600X /128 GiB RAM, six independent processes per condition completed twelve measured decode repeats each:72 measured requests per condition,216 total. All output IDs, logprobs and MTP counts match the qualified reference. All18 processes exit normally, with no GPU faults or forced cleanup. The first logged protocol check also exits normally; its timings are excluded.

Prefill and decode are assessed separately. Faster prefill is not a reason to accept slower decode. The earlier24-read comparison has only two independent processes per condition and mixes two prompts with different MTP counts. Its per-read ranges are not an estimate of timing noise or proof of decoder equivalence.

Each process first runs the same fresh32768-token prompt and64-output decode, creating its actual prefill workspace and queue. Before every subsequent decode it restores the same verified32831-token numerical state and sends32832 input tokens for64 outputs. All repeated decodes reuse32831 tokens, execute no PP chunks, and have46 accepted of51 offered MTP drafts. One warm decode is reported separately; twelve later repeats are measured. Restoring and priming times are excluded from these repeated-decode rates. The three conditions run in all six order permutations; every condition occupies each position twice.

| Condition | Repeated decode tokens/s | Mean64-token decode ms | CV of six process means | First decode after fresh32K tokens/s |
|---|---:|---:|---:|---:|
| Qualified baseline869 | 22.1645 | 2887.50 | 1.786% | 16.8259 |
| Candidate272, prefill queue OFF | 21.8212 | 2932.92 | 0.594% | 16.4438 |
| Candidate272, prefill queue ON | 21.8921 | 2923.43 | 0.787% | 15.4642 |

Uncertainty uses six paired independent process blocks, a log-latency Student t interval with five degrees of freedom. Twelve repeats improve each process estimate; they are not twelve independent process samples. Positive speed changes mean faster decoding.

- `nativeoff_vs_baseline`: -1.561% speed, paired process-block95% interval [-3.249%, +0.157%]. Direction unresolved; does not establish equivalence.
- `nativeon_vs_nativeoff`: +0.326% speed, paired process-block95% interval [-0.731%, +1.393%]. Direction unresolved; does not establish equivalence.
- `nativeon_vs_baseline`: -1.240% speed, paired process-block95% interval [-2.489%, +0.025%]. Direction unresolved; does not establish equivalence.

The first64 output tokens immediately after fresh32K prefill are a separate secondary comparison: six samples per condition,41 accepted of66 offered MTP drafts. These are not pooled with the restored continuation workload.

- `nativeoff_vs_baseline`: -2.251%, interval [-5.116%, +0.700%].
- `nativeon_vs_nativeoff`: -5.866%, interval [-10.167%, -1.360%].
- `nativeon_vs_baseline`: -7.985%, interval [-12.806%, -2.897%].

The fresh32K result is a measured regression in this workload: all six ON-versus-baseline paired speed changes are negative, and its95% interval is entirely below zero. ON versus OFF also has an entirely negative interval. The candidate as a whole is held back. The qualified decoder remains the reference; faster prefill is not used to offset this loss. The opt-in prefill work stays a candidate until the transition into decode is isolated and measured again.

Existing host counters put the first-decode CPU expert work near123.78 ms/window for baseline and125.22 for ON, and GPU-reach waits near2.76 and2.96. The input-stage category is3.23 versus10.10 ms/window. These are diagnostic counters, not physical GPU busy times. Source review finds that `Verifier::run` counts the PLE gather/copy interval both explicitly and inside the enclosing input-stage interval; therefore its stage number cannot be treated as a time fraction. Graph capture occurs before that stage clock starts, so the stage increase is not a direct graph-capture measurement. The data points to checking input staging and phase resources; it does not yet identify the cause. See [host-counter means](native-copy-decode-repeat-v0141-first-decode-host-counter-summary-v1.json).

An interval including zero does not establish equivalence or rule out a smaller regression. Repetition does not remove physical variance. These results cover one qualified32K coding continuation and the separate first decode after its fresh prefill, not every prompt or longer generation.

The measured engine's decoder source files are byte-identical to baseline869; only prefill.cpp and the common queue header change. The selector is called only during prefill initialization. That source scope does not prove zero runtime effect: the prefill queue/workspace survives into decode and the common header changes the build. Candidate OFF versus baseline checks the common build effect; ON versus OFF checks native queue selection and its retained resources. This experiment does not adopt a decoder implementation or change defaults.

Actual startup geometry matches in all18 processes: context262144, int8 KV resident32768, expert cache128 slots/325 MiB, workers5, MTP4, PCIe fraction0, free VRAM1565 MiB. LP5 scoring is included. Normal timings have no API tracing, GPU profiler, timestamps or new synchronization. Existing unconditional host counters are printed once per request in all conditions. Their CPU/GPU-wait categories are host wall times and can overlap other work.

The raw reused-request receipts inherit a generic `prefill_tok_s` calculation over all input tokens. That value is not a prefill rate because32831 tokens are reused; it is excluded from these summaries and decisions. Tensor data, model files, the619 MB checkpoint and the logged API trace remain private. The prior [full physical262144 lifecycle proof](../full256k/README.md) qualifies this unchanged binary; this timing experiment does not replace it.

[Summary and process-block intervals](summary.json), [all252 requests by category](measurements.csv), [source phase review](native-copy-v0141-phase-scope-source-review-v1.json), [sequence receipt](sequence/record.json).
