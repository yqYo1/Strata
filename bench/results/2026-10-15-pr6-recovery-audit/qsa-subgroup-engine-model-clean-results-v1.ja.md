# QSA subgroup 実モデル clean比較

2026-10-17 Arc B570/driver1.17.39395+14/LLVM22.1。同一binaryのQSA選択だけを変え、
公式bench.shの実入力39262/read39262/reuse0、thinking有効、C8192/RGPU32768/window2。
optimized,subgroup,subgroup,optimized,optimized,subgroupの6独立process。詳細ログ・validationなし。

| 経路 | 各process PP(tok/s) | 中央値(tok/s) |
| --- | --- | --- |
| optimized | 490.6, 490.1, 491.4 | 490.6 |
| subgroup | 495.4, 478.7, 495.9 | 495.4 |

中央値の変化は0.9784%。候補全sampleが基準全sampleを上回る: False。
平均の変化は-0.1427%。中央値と平均を区別し、遅いsampleも残す。
実モデルの安定した改善はこのcohortでは未証明。現行経路を維持して次の検証を選ぶ。
全process正常終了・所有終了、fresh fault0、dumpなし。個別timing/環境/入力fingerprintをJSONへ保持。
この条件の観測であり、他条件のばらつき全体を保証しない。24生成は完了確認で一般品質やdecode速度ではない。
全physical262144モデルconsume/RESTORE、cancel/reuse、品質とデフォルト採用は未資格付け。
600/1000tok/sおよびdecode70tok/sの達成は認定しない。
