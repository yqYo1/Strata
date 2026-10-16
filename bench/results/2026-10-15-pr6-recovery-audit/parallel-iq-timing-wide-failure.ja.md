# 追加IQ展開/GEMM測定の停止

2026-10-15、`check_parallel_iq_timing_v2.py wide` は180プロセス中31件の
正常完了後、case-031で180秒の期限に達した。元の記録
`parallel-iq-timing-wide-v2.json` は passed=false、complete=false のまま保存する。
先に完了したne=64のscreen測定を、この失敗によって成功した全条件比較と解釈しない。

停止条件は既存経路OFF、IQ4_XS(type23)、GU、ne=16、展開+oneMKL GEMM。
PID3767352/start_ticks38377657は03:30:12.448935 UTCに開始し、stdout/stderrを
出さないまま期限に達した。監督処理は所有を固定したpidfdでSIGTERMを送り、
exit=-15、03:33:12.690701 UTCに回収した。session_empty=true、survivors=[]。
observation_complete=falseは異常終了を表し、正常終了の証明ではない。

直前の候補ON/type23/ne16は8.6255秒で正常終了し、全半精度要素、GEMM出力、
guard、入力不変性を確認した。これは既存OFF停止の原因を同定する証拠ではない。

最終journalにxe CCS Engine resetが1件あった。journal時刻は03:33:12.533911 UTC、
記録monotonic=383956655081us、kernel source monotonic=383953528979us。
resetと監督終了の時刻が近く、現在の記録には停止したAPI/カーネルや終了シグナルの
個別送信時刻がない。GPU停止が先か、未完了GPU仕事を持つプロセスの終了に伴うresetかは
未確定。devcoredumpなし、boot変更なし。恒久的GPU停止・自動復旧のどちらも未確認。

同じ測定は再実行しない。次のGPU検証の前に、queue生成、USM確保、upload、reference GEMM、
展開、候補GEMM、各wait、retirementの進捗を記録し、期限前に所有プロセスのスタックを
取得できる診断版を準備する。CPUのみの実装確認は継続できる。

保持対象はこの異なる未解決停止機構の元receipt、空ログのhash、全31件の個別sample、
該当kernel行、binary/source/env/owner/終了記録。成功側の重複全文ログは証拠抽出後に整理する。
