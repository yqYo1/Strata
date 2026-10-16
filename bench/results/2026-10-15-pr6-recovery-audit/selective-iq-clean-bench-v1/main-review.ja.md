# Main review

2026-10-15。source handoff後にcontroller、wrapper、pinsと実際のserver/runnerを確認。
新nativeのpath/hashをwrapperとbuild receiptの両方で固定し、古い固定wrapperは使わない。
serve/server.py:454は`[exe, --serve, *args]`で起動するため終了argv検査と一致する。
runnerはplanのoptionsを実際のbench.shへ渡す。long-generationのstrong/weak判定keyも
現在のtools/long_generation_bench.pyと一致する。wrapperは実行可能mode755。

通常のbenchmark、model、metadata、全配列比較、所属process正常終了とfresh fault gateを確認。
元32089入力に対応する実prefill prefixは32088であり、元collector failureを保持した独立
assessment v2をgateに使う。raw arrays/API traceをretention policyで退役しても、hashで固定した
完全な比較結果と同じ48層censusが残る。端末末尾の抜粋を代用品にしていない。

full arm0から開始し、nominal32Kと6turn agent sessionを各3回、thinking有効の
strong/weak長生成を各3回取得する。clean比較は同じnative binaryのIQ flagだけを変える。
arm1はarm0の正常閉鎖・全suite資格・新規faultなし・recipe/source/env対応を確認後だけ実行。
失敗したarmは再実行せずそのreceiptと個別結果を保存し、原因に応じた次の実験を別名で行う。

GPU speed measurementは未実行時点のsource reviewであり、完走や性能採用を主張しない。
single-server内反復だけで独立process間のばらつきを解消したとは扱わない。
元の32K停止機構は未解決で、clientのfail-fastは原因修正ではない。
