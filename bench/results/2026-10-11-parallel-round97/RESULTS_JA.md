# 直列serveの受付条件修正のレビュー

実装担当gpt-6.1-solの報告を全文確認し、Mainでcache binding、Verifierの初期化条件、完了を待つ各segment、実効slotを使うCPU pool、prefillのcache参照を確認した。concurrent handoffの許可とcache residencyの準備を切り離す小さい修正を適用した。通常の非serve条件とwhole-token graph・MTP chainの拒否は維持した。

[元の報告](implementation-xe-serialized-serve-admission-v1.txt)、[patch](serialized-serve-admission.patch)、[Mainのレビュー](xe-serialized-serve-admission-main-review-v1.json)、[索引](report-registry-v120.json)にsource identityと限界を残す。新しい修正はbuild/lint/ASan・READY・model request・品質・性能・262144セル全域の検証前であり、採用済みとは扱わない。受付失敗の元FAILEDは変更しない。

実装agentは完了した。Lunaはサービスusage limitで実行不可のまま、現時点のactive調査agentは0。他モデルへ代替せず、制限解除後に別スコープの並行調査を再開する。

[lint比較の準備済みcontroller](prepared-lint-v5/run_root.py)は、今回変更したgenerate.cppだけをcommit eaの修正前ソースと、両モードの実際のcompile databaseで比較する。新しいlint findingを既存の警告と混同しない。準備済みで、まだlint実行の結果ではない。

両モードのrelease v10・host ASan/UBSan v7は実行が正常に閉じ、各CPU確認は10件PASS・native ISAの1件skipだった。[個別receipt](qualification-progress.json)はbuildとCPUだけの結果で、新しいGPU/model資格ではない。現在のlint v5は解析中で、新しい修正のsource commit・model実行はまだ行っていない。
