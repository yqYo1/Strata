# 既存FP32 GEMMの別キュー回帰テストの期限超過

v2全体ビルドは成功し、新規GEMM16の68ケース、runtime/compatの確認は正常終了した。一方、変更していない `tests/sycl/gemm.cpp` の既定実行は300秒で期限超過し、所有プロセスのみSIGTERMで終了した。元の `own-native16-regression-correctness-v2.json` はFAILのまま保存する。reaped=true、session_empty=true、survivorsなしを再確認した。正常終了を表す `module.closed` は期限停止を受理しないため、追試の事前チェックには終了・不在を明示して判定する。最初の追試コントローラーのこの判定失敗も別のFAILとして残す（GPU実行なし）。

所有PID3765874/start45015903へのGDB観測を `own-native16-separate-queue-timeout-v2.stack.txt` に保存。Mainはpipeline内のyield、残る3ワーカーはcudaEventSynchronize→UR/LevelZeroのevent wait。main戻り先0x408d2aの実バイナリ逆アセンブルでは直前のcallがesi=1、edx=0で、`pipeline(q,true)`（転送と計算が別キュー）に対応する。numerical loopsはそのcallより前にあるが、完了したケース数を途中で出力する実装ではない。v1の同じ既定テストは約7.49秒で通過しており、再現性・原因は未解決。registry寿命修正が原因とも、GPUが故障したとも断定しない。

今回のnative16経路は単一キューを必須としている。追試は元のfixtureを変更せず、その別キューcall1か所だけを除いた私有fixtureを別に構築し、全numerical casesと両方の単一キューpipeline、native16 postops、明示/静的解放の終了を検証する。これは元の全回帰テストのPASSにはならない。元FAILと範囲制限をモデル比較にも記録する。長い測定はこの実使用経路の正常終了が得られた場合のみ実行する。

同一boot、新しいkernel GPU fault 0、devcoredumpなし。GPUリセット、unbind/bind、他プロセスやサービスへの操作は行っていない。R797へコード/API契約の読み取り専用調査を依頼した。重複したGPU試行は行わない。
