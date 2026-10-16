# R961: native16 prefill arena scope review

- 既存記録はGEMM workspace削除、FP32 `emb/mixed/bo/ple_emb` alias、容量仮定を扱う。native16 scratchの型別plan/livenessは未評価。R685/R677/R712本文は未読で、訂正済みMain報告だけを使用。
- 現sourceはFP32 extentsをtyped viewで再利用。`R16`/`gated_b16`は旧`gated`の前後半で既に詰まっている。C8192 MoE `Dm`は旧予約838860800 B、BF16 span419430400 Bだが、shared regionはmax(GDN,QSA,MoE)なので総削減とは限らない。
- alias OFF時、`bytes_needed()`だけがnative16で使わない`xn` FP32 `T*D`を数える。unfused HCは拒否され、carveも省く。C8192の超過は335544576 B (320 MiB+guard)。
- long pathは`T`,`T_max`>8192を拒否。shortはFP32 `run_impl`、PLE/final callbacksはFP32 bridge、GDN/indexer/PLE stateはFP32。source-onlyにはprofile/census不要。capacity/quality/speed/RESTORE/262144は別のMain gate。
- fresh **decision-support** auditは`schedulable_now=true`、未dispatch。1,400–1,800語でdedicated arenaとtemporal remap、exact bytes/liveness、callback/exception lifetime、8192超のsource guardを比較する。16K fit/speedは主張しない。

必要source pinsと図書館評価はJSONに記録。indexは変更していない。
