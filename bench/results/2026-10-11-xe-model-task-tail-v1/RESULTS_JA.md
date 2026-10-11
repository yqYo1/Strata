# 受付修正後のモデル実行と入力末尾の確認

source51436a0の受付修正は両モードのbuild・host ASan/UBSan・lint比較を通過した。lintは既存の83件を各モードで維持し、新規findingは0。元run.shの非zero exitはreceiptに保存した。

[前回の実行](model-v3-failed.json)はREADYに到達し、4096と8191位置までの2チャンクを完了したが、停止トークン1個、MTP draft0で終了した。監督側のcoverage判定がQUIT前に失敗し、所有プロセスをSIGTERMで回収した。元FAILEDとexit-15は変更しない。[独立解析](analysis.json)は実行時library8個の一致、新規GPU障害0、144枠中29枠のfillを確認した。8K診断は長文性能・品質・262144全域の検証ではない。

[次のcontroller](controller-v4/run_root.py)は依頼文と会話の49-token末尾を残して8192入力を作り、完了済みのリクエストをQUITで正常終了させてからMTP coverageを判定する。[CPU検査](cpu-v5.json)はPASS。モデルv4はこの時点で未実行で、末尾欠落とEOSの因果関係は未確定。

[次のキャッシュ修正](../2026-10-11-parallel-round98/RESULTS_JA.md)は別実装agentから戻ったsource-only候補。現行sourceのv4が閉じてからMainが統合・検証する。元の失敗と今回の修正を混同しない。
