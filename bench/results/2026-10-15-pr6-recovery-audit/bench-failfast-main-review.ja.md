# Bench request failure: Main integration

2026-10-15。tools/strata_bench.pyにdefault OFFの`--fail-fast-engine-error`を適用した。
返ったresponseのdigestが失敗なら、単一result行をfsyncしてからsuiteを停止する。
次のchatや測定後のmetadata取得を行わず、途中までの全request sample、失敗理由、
比較identityをcomplete=false/terminal=failedのresults.jsonへ保存し、exit1とする。
通常serverの自動復帰は変更しない。正常responseのlong-generation適格性不足は
engine failureに分類しない。KeyboardInterrupt等の元例外も保持する。

Mainはdiff、7つの新host test、既存のmain/journal/long-generation testをレビューした。
同時requestを出す--clientsと本optionの併用は起動前に拒否する。既にin-flightのrequestを
止めると測定条件が変わるためで、今回必要なserial suiteに適用する。

Mainのcheck_bench_failfast_v1.pyが共有LOCK_NB下で適用し、GPUなしで実行した。
新test7件、既存bench test5件、journal/evidence test14件の計26件が通った。
第3回32Kの失敗を模したmain統合testでは、先行2sampleを含むpartial results、
valid hash chain、pending空、単一terminal row、以後chat呼出しなしを確認した。
calibration失敗、default継続、意味上の不適格、元例外も別caseで確認した。
各commandとsource hash、終了/回収結果はbench-failfast-host-v1.jsonに保存した。

今後の正式serial suiteは`bench.sh --long --think --long-generation --runs 3
--fail-fast-engine-error`を使う。元の失敗receiptと歴史的recipeは変更しない。
本変更は測定制御であり、推論停止原因の修正や速度改善の実績ではない。
