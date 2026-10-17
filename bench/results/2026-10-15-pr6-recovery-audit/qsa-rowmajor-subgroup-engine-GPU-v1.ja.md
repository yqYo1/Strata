# 実エンジンへのQSA subgroup統合

候補commit f685b16a (verify/moe-window)、Main実装。
選択肢bf16-prefix-subgroupを追加し、score33 row-majorとBF16保存/FP32 stateを維持。
実際の生成済みprivate候補とのnormalization body一致をsource manifestで確認。
旧経路・短入力・decodeの選択は維持し、現時点でデフォルト採用はしない。

2026-10-17 Arc B570/driver1.17.39395+14/LLVM22.1、fresh Release build127工程PASS。
実エンジンSHA256: 24ea1edf88570c456df2364594662c1f77e8b7bae0db8c17e981cfffe0bfbad9。
CMakeの新しいsubgroup targetをcombined UR tracing+parameter validationでsmoke2case、
component11case実行しPASS。各24/60kernel。全finite/guard/input/prefix/同経路repeat、
独立BF16 operand oracleとdomain/alias/extent拒否の既存基準を維持。
physical262144 cellsのfull prefixとlive13も含む。最大max_abs6.103515625e-05、
budget_ratio0.9682716738は事前基準内。全process正常終了・所有終了、fresh fault0、dumpなし。

実モデル39Kの初回診断は別receiptへ記録する。この文書は単体資格付けのみ。
一般品質、モデル全physical262144 consume/RESTORE、cancel/reuse、実モデル速度は未資格付け。
R984の監査は共通preflightのFP16/BF16 capability差を指摘した。現在のB570は同じBF16
実装の全componentを実行済みだが、他の機器の対応や早期拒否の網羅を証明しない。
元報告本文はMain未読、native reviewerによる対象/継承関係の整理後に対処を判断する。
