# R987 レビュー

- 3件のreport hash・byte数・budget、仕様hash、wave、settlementを確認。3 worker全員の完了・解放・配信確認済み。
- R984 は受理。7ファイルのsource hashは固定manifestとbuild receiptの両方に一致し、候補commitは `f685b16a`。
- FP16だけのshared preflightは既存仕様だが、新subgroup routeにBF16の早期確認がない点は統合で露出した他GPU向けの穴。B570での失敗ではない。
- 実際のB570 subgroup smoke 2件とcomponent 11件はPASS。BF16実行可否はこのGPUで確認済み。一般の事前判定、実モデル、262144-context/RESTORE、品質、採用は未確認。
- 他GPUを対象にする場合は、層処理前にBF16 matrix tupleとlocal-memory要件を確認し、非対応時にwork開始前に止める。
- R985 はlong-prefillのsmall-M MoE設計資料として保留。実測censusから対象M/formatを選んでから本文を参照する。
- R986 はquery間KV共有の知識資料として保留。Strata実selectorの重なりとQSAの優先度が確認されるまで使わない。
- 39K model diagnosticの結果はこのレビューで確認していない。成功・fault/owner closure後にのみ用意済みclean cohortへ進む。
- 現時点で追加researchは推奨しない。
