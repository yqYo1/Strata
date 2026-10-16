# Integrated 0.1.41 snapshot: full-context qualification

Completed on 2026-10-09 JST on Arc B570 10 GiB, Ryzen 5 5600X and 128 GiB installed RAM. This is a correctness and lifecycle check of integrated engine commit `1eb89482a4afd20277ae0405780ed4f8eb98eb20`, binary `86972697ecb1903750d202f8be29a7a1da351eb3415678c3e3b6af790804d1c8`. The exact unmodified upstream snapshot was measured separately in [default32k](../default32k/README.md). This physical test does not qualify the raw upstream binary.

The [terminal record](run/record.json) is SHA-256 `168d7896cbe3a60dfc604a08b6fe310f16d823a17d13d8a013260bfc7facb0a4`. All twelve requests and six SAVE/RESTORE operations passed. The process completed normal QUIT, exit zero, without forced cleanup, a surviving owned process or a new kernel GPU fault. Total diagnostic wall time was 4716.073 seconds. Logged durations are excluded from performance comparisons.

## Configuration and comparison

The model is Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S. Context is 262144; int8 KV keeps 32768 cells resident. Actual prefill chunks are 8192. The run uses 128 expert slots, five CPU pool workers, PCIe fraction zero, MTP four, greedy output, no prefill borrowing, compact mode two, attention layout one, owned KV staging and no KV prefetch. Prompt caching is enabled with checkpoint interval 262139, root zero and turn token -1. Exact argv, startup geometry, runtime environment, executable identity, boot ID and source hashes are in the record.

The flushed UR, Level Zero and Strata diagnostics were enabled. The completed controller recorded 67,452,126,763 bytes of engine stderr and its hash; the archive reuses that terminal hash without reading or copying the large log again. No profiler or interposer was used. Driver, package, PCIe and global settings were not changed.

The earlier qualified DD5 binary is `dd5efb9002167481d180f370b86fb6978d724f37745f7e4a923432296725ad34`. Its [full-context receipt](references/qualified-dd5-full256k.json) is included. Before the first full input, both versions perform the same fresh 32K read, SAVE and live continuation. The updated binary also has a separate [four-fresh-32K numerical check](checks/fourfresh32k-numerical-subset.json): first logits, 66 live state parts, output IDs, logprobs and MTP counts match the qualified references. That subset supplies only those four checks, not the earlier process's full-context result.

## Completed checks

| Check | Observed result |
| --- | --- |
| First full input | 262140 input tokens and four visible output tokens; last executed physical cell 262143; logits, live state, output, logprobs and MTP counts match DD5 |
| First full SAVE | 4,239,591,088 bytes; all saved tensors match DD5, including inactive MTP regions and main indexer spare rows |
| Fresh 32K between full reads | No prefix reuse; numerical reference matches |
| Second fresh full input and SAVE | Same numerical results and every saved tensor byte as the first full input |
| Full tail clipped to two outputs | 262142 input tokens, resume at 262139 and two visible outputs; last physical cell 262143 |
| Actual 32K disk RESTORE | Engine validates the saved checksum and compatibility, then reproduces the live continuation |
| Actual full disk RESTORE and SAVE | Engine restores the 4,239,591,088-byte image; every roundtrip tensor byte matches the first SAVE |
| Clipped tail after disk RESTORE | Same output, logprobs and first logits as the live clipped tail |
| Capacity refusal | 262144 input + one output and 262142 input + three outputs are rejected |
| Fresh 32K after refusals | Full prefill and numerical checks pass; normal shutdown follows |

Every full SAVE covers all 262144 physical cells of each of thirteen main KV layers. No tensor bytes are ignored. [summary.json](summary.json) preserves the checks and exact request fixture hashes; [protocol output](run/protocol.stdout.raw) and [project messages](run/project-messages.txt) retain progress and SAVE/RESTORE responses.

## Version identity and request history

The configuration fingerprint deliberately includes the engine version. The known old/new fingerprints therefore differ: `9380593023437872391` and `643052213586580166`. The included [version comparator](controllers/compare_saved_session_versions_v0141_v1.py) checks that exact pair and compares the remaining semantic image, including every tensor hash. It does not waive unknown configuration differences or changed tensor bytes. Existing storage-offset/LRU exclusions remain the same.

The [earlier rejected attempt](failures/different-request-history-r1.json) performed extra B/A/B reads before its first full input. It completed normally but failed saved-image equivalence: inactive MTP prefix cells still contained the different preceding history, and the engine-version fingerprint also differed. That receipt is excluded from full-context qualification and performance. The [eight CPU comparator cases](checks/version-and-history-cpu8.json) still reject that actual tensor mismatch, corruption of inactive MTP/main scale/GDN/spare data, an unknown version fingerprint and changed geometry. The corrected physical run uses matched history and passes all saved bytes. The [scope correction](checks/history-scope-correction.json) preserves the correction to an earlier explanatory note.

The archive contains small receipts, comparators, token fixtures and the original controller. Their original local model and prerequisite paths remain explicit. Executables, logits, live-state dumps, session images and API stderr remain private. The manifest hashes every included file. This folder establishes the integrated binary's full-context lifecycle; actual8192 quiet speed comparison and the integration decision are recorded separately after their sequence finishes.
