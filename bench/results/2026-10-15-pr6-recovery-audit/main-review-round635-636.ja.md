# 入力照合と次の大きい改善に必要な情報

2026-10-15。MainがR635/R636と`bench-request-identity-v1`の全sourceを読んだ。
進行中のfull0は同じexec16956/native3892403(start38567673)で生存し、agent sessionを
処理している。新しいGPU/build/test/cleanupは開始していない。
この段階の新しい候補はすべてsource-onlyであり、実行結果ではない。

## 全入力の比較

`bench-request-identity-v1/compare_requests.py`は、両journalとstateを8 MiB/64 KiB、
2048行以内でsnapshotし、既存chain verifierで完全性を検証する。全startを順番通りに
phase/run/turn、max_tokens、thinking、実body SHA256で比較する。warm-up/calibration、
短文、長文、agent全turn、両long generationを省略しない。重複key、欠落、余分な件、
順序違い、body/parameter違いは不一致とする。元のファイルと結果statusは書き換えない。
identity行のsource/nonce metadataは独立に保存されるため、旧clientと新clientの差分を
隠すものでも、native closure・品質・性能を自動合格にするものでもない。

MainはSuiteの全request呼出しも照合した。現full0の設定ではphase/run/turnは識別可能で、
cohortを減らした疑似比較は不要。utility/host testsは未実行。job終了後、pinned旧clientで
nonce回復を先に試し、fixed-workloadとidentity utilityのhost checksをMainが直列実行する。
その後のpaired controllerは新しい依存pin・入力照合を使う。旧full1 recipeを無条件に
起動しないという前のMain reviewの変更を維持する。

## R636: waveを実装する前の情報

現在のcensusは`counts(l,T,cnt,512)`で各chunkの正しいexpert countsを受け取り、
GU/Downのissuer hookもexpert IDとneを持つ。新しいraw route captureを増やさず、
同じ任意診断ownerに512-bit unionと小さいne accumulatorを加える方法は妥当である。
通常実行のnull branch、queue/kernel/math、4 MiB owner/1 MiB output capを維持する。
ID別GU/Down seenを照合すれば、同じneの別expertを取り違えても検出できる。
実装・sizeof・overflow/partial-tail host tests・実モデル実行は未了である。

既存32088-prefixの48層histogramだけでも、82878 expert-chunk incidencesに対する
unionの範囲を導ける。R636のq2:41448–80604、q4:20736–48700はexpert ID重複の実測では
ない。q4では少なくとも34178回分のincidenceが同じexpert-waveに属するという算術上の
材料になるが、速度やメモリfit、oneMKL shape/丸めを証明しない。wave設計の優先度を
決める情報として使い、現在の最適化の採用根拠にはしない。

statusを区別する。元のmodel collector receiptは失敗のまま。別のassessment-v2は、
nativeを再実行せず、元の正常native実行と32088-prefixのcensusを検証した完了/PASSの
assessmentである。`GPU_executed=false`はassessmentで再実行しなかった意味で、実際の
censusが実行されなかった意味ではない。cleanな速度測定・全品質の合格でもない。

## R635: defaultへ反映する前に残る実装義務

Responses/runconfig/shared serviceの存在について、古いR447の欠落リストを現在へ
引き継がない。R635はこれらがorigin/mainと同じであることと、現root entrypointに
tail-role checkpointがあることを確認している。0.1.38表記だけで機能欠落を推定しない。

一方、Mainも現コードで、generic serverが`--batch`/BGENとelastic VRAMを扱うのに
root nativeはGEN/GENIのみで、`vram_elastic` booleanをunsupported flagへ変える箇所が
残ることを確認した。配置・graph pointer lifetimeが絡むnative機能と、server/configの
契約を分けて対応する。CUDA/HIPエンジンをコピーする理由にはしない。

明示的なunsupported応答/設定拒否は、未対応機能を誤って起動する問題への暫定対応に
なり得るが、依頼された機能の実装完了の代わりにはならない。機能ごとの義務と評価を
残し、単にdisableして「全機能取り込み済み」としない。default更新はこのsource/API
auditも含む具体的な境界で扱い、履歴が祖先であるだけを統合完了の証明にしない。
現在のpinned benchmark sourceを途中で書き換えない。
