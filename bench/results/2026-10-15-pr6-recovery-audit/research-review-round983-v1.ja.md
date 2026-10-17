# R983 レビュー

- R980/R981 は受理。R982 は本文保留の知識資料。
- 報告・仕様・入力のハッシュが一致。全タスクは完了、解放、配信確認済み。
- 解放前の Main 観測は `codex / gpt-6-luna`。解放後の provider=null 表示とは分けて記録。
- R980 の入力sourceと生成候補ハッシュは build receipt と一致。commit 表記差はあるが対象source bytesは同一。
- Main の QSA build、smoke、11ケースcomponentは PASS。最大誤差比0.96827は許容内だが、採用・実モデル評価ではない。
- 直近の阻害要因なし。用意済み untraced 6-process QSA component cohort を順次実行する。
- 腕間hash一致は要求せず、BF16独立oracle、guard、同一腕repeat、fault/owner closureを維持。
- R981 の4.75/4.85 CPU秒は instrumented diagnostic stamps。clean speedや転送量の証拠ではない。
- R982 は実測した資源上限の質問が出るまで本文保留。新規 research dispatch は推奨しない。
- host-USM と small-M は QSA cohort 後に Main が優先度を選ぶ。実装とGPU計測は Main 所有。
