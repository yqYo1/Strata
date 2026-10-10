# GDN factor 32K component service comparison

Arc B570 10GiB root0000:05:00.0, oneAPI2026.1, source5ab8ab0a, binary8b7dc40a45e0b1044bcb0201a613cde6b1d509a4cfcc2405796aa2b2705ed6ee. CPU build completed119 steps in327.134599348s, normal0 with empty owned sessions. Source, binary, library and boot identities are pinned in original receipts.

The quiet paired synthetic prefix was32768 inputs in four8192 chunks, three reset samples per arm, complete untimed full-prefix warmup and balanced within-sample order. The clock spans gates+conv+recurrence+norm submissions through queue drain and async-error check. Uploads, readbacks, hashing, checks and progress flushes are excluded. Checks between pairs affect cache conditions. This is combined component service time, not model wall time or throughput.

| Sample | Legacy log gate (ms) | Producer factor (ms) |
| --- | ---: | ---: |
| 0 | 148.836091 | 149.337857 |
| 1 | 148.310772 | 149.213512 |
| 2 | 146.069502 | 148.959475 |
| Median | 148.310772 | 149.213512 |

Factor median is0.608681% longer and no sample sum improves. No speed benefit was demonstrated; do not adopt from this result. The option remains OFF by default. No model/full-physical262144 qualification or decode performance claim follows.

All12 measured chunk pairs and all warmup chunks passed full bitwise state, convolution history/output, beta, materialized-exp factor, FP32/FP16 output, guards and immutable-input comparisons. Repeated chunk hashes agree. The requested quiet quad configuration is not a quiet native launch trace. The separate default diagnostic identifies22 legacy and22 factor quad kernels and matches earlier default fingerprints. Host-only contract and six before-queue input/logging refusals passed.

Final benchmark normal0, empty owned session, no cleanup/errors/survivors, sampled peakRSS2463166464B, total process wall32.720775828s (includes excluded checks/setup). Complete new journal interval has no GPU entries and no coredump; runtime/binary/source/boot pins remained stable. No recovery or service change occurred.

Root controllerv5 failed while parsing the expected host rejection: it required 'FAIL ' although the pinned fixture correctly wrote 'FAIL,bench admission'. The host exited1 normally without queue construction; earlier diagnostic passed. Original failure receipt and21-byte rejection are retained, with the clean failure-time journal interval. No quiet benchmark launched in v5. New controllerv6 corrects exact expected CSV messages, rejects unknown/empty/reordered rows, checks exact stage order and interval scope (R159 suggestion), and completed qualification. Original v5 remains failed and unchanged.

Successful verbose diagnostics and full build chatter are represented by original bytes/hashes, command/status/flags, warnings and complete launch-name/count/shape summaries. They have no current full-history consumer and may be retired after this archive commit. Keep original rejection evidence and active fixtures. The next independent source experiment is private prefill-only IQ4_NL direct dispatch based on609a270d; its measurements are pending.
