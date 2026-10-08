# Additional real-tokenizer server checks

The earlier 593-case upstream server suite completed 585 cases successfully and skipped eight. Five skips required a pack tokenizer; three were Windows-specific. After the quiet GPU speed sequence ended, the five tokenizer cases were run with `STRATA_TOKENIZER` pointing at the actual Qwen3.8-Flash-Next IQ3_S pack. All five passed without a skip, in 4.888 seconds of unittest time/5.187 seconds of controller time. The three Windows-specific cases remain unexecuted.

The [terminal record](record.json), SHA-256 `93768e1651556290a5a3f60b954f5647be83305525e5b56eae5210f3921984e1`, preserves the exact five labels, argv, existing isolated Python environment, tokenizer file hashes and source hashes. The test, server and tokenizer sources match the unmodified pinned upstream byte for byte. [stderr](stderr) contains all five successful case names and the unittest result; [stdout](stdout) is retained. No model or GPU was launched. No dependency or package was installed.

The cases cover multilingual split-character streaming, random IDs/broken UTF-8 bytes, per-token detokenization cost after 16K tokens, real-vocabulary heap-BPE parity and an 8K-character CJK encoding speed check. The original 593-case receipt remains unchanged; this is a targeted addition, not a rerun of that whole suite. [Original suite and twelve-file server parity](../default32k/README.md) and [the CPU launcher checks](../default32k/checks/server-wrapper-cpu10cases.json) remain separate.

The tokenizer itself stays in the existing private model pack. Its three file hashes and path are recorded for replay. The manifest hashes the archived record, logs, runner and this report.
