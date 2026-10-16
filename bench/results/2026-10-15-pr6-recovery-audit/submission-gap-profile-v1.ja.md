# GPUコマンド空白と計装の影響

同じcanonical個別表を使い、長入力Attention envelope94.944752862秒のコマンド和集合と
空白を再計算した。380,993区間、合計12.892909056秒で元の集計と整数nsまで一致した。
各空白を次に開始したコマンドで分類すると、barrierが2,583区間・9.009487150秒、
vLLM DenseKernelが86,996区間・0.698151473秒、M2D copyが23,656区間・0.544613736秒。
これは**次に観測したコマンドの分類で、そのカーネルが空白を起こしたという意味ではない**。

全空白のうち、観測された何らかのappend→start区間と重なる部分は2.972612322秒、
重ならない部分は9.920296734秒だった。barrier前の空白は、そのbarrierのappend時刻より
前が8.957010753秒。ホスト/投入系を調べる手掛かりにはなるが、CPU律速やhardware idleを
断定しない。公式PTIのappend時刻はOnEnter callbackで元APIが戻る前に取得するため、
append済みを「driverが受理済みで実行可能」とは扱えない。時計は同じPTI相関trace内であり、
外部CPU/wall時計とは接続していない。全個別空白はcanonical tableから再現できる。

また、同じsource63326・engine633449・request hash・実入力39262・thinking ON・出力24の
既存clean1件はprompt97,315.9ms /403.4tok/s、計装1件は106,041.6ms /370.3tok/sだった。
複数の診断制御が同時に異なり、1件ずつなので8,725.7msの差を特定のprofilerへ帰属できない。
coarse/QSA markersはcudaEventRecord経由でbarrierを追加し、collector/CPU profileにも
観測コストがある。従って上記9.009秒をそのままproductionで回収可能とは主張しない。
採用判断には別の反復clean比較が必要。元のcurrent-source clean cohortは未完了のまま。

解析source、分位値・全kernel別集計・80µs以上の実個別空白は
`submission-gap-profile-v1.json`、元2件と環境差・制限は
`resource-instrumentation-context-v1.json` に記録した。新しいGPU実行はしていない。

実メトリック83列のoffline readerは、型上限・欠落・重複・未知列など13ケースで期待通り。
ただしReportError=trueも形式上decodeできること、catalog buildとquery packageのidentityを
誤って同一扱いするsource条件を記録し、v1を未実行/HOLDのままにした。元receiptを改ざんして
条件を通さない。実際のAPI statusとreport-validityの確認後に、まず小さい実収集を行う。
