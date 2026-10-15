# prompt Direct-outputの現native反復評価

2026-10-15。前goal turnはXeStrata由来診断撤去、actual再build、実78453入力、
前後768 FP32 selected rows一致、fault0の証拠を得てcommit/pushしたためprogress。
XeStrataは性能参考値のみで、以下はpre-Xe own-forkのprompt Direct-outputを評価する。

currentnative8fdc241d5227f7bde921a8bbdccf9535d6b263d35237751488338cdf438b89de。
Direct0/1旧同native比較はPP251.8/303.2、撤去後Direct1は293.8で、全て各一回の診断。
診断の差やばらつきという説明だけで採用せず、同currentnativeのclean各3processを実測する。

run_native_direct_clean_cohort_v1.pyの事前順は(Direct,repetition)=
(0,0),(1,0),(1,1),(0,1),(0,2),(1,2)。失敗したらそこで終了し自動再試行しない。
各native独立起動でwarmup後にofficialbench.shのprefill-only 65536-min requestを一回、
workloadIDはprefill_v1_20261015で固定、実入力78453/read78453/reuse0を結果で照合する。
thinkingON、24出力はprefill終了の確認用でdecode性能やtask品質には使わない。
C8192、FP32/R_GPU32768、nativebank1/head0/QSA1/alias1、no-token-graphを固定。
TRACE、数値capture、profiler、validation/log layersは外し、既存runtime安全設定は維持する。
Direct0/1既合格selected-row比較とcurrentkernel/runtime source不変をlaunch前に確認する。

Mainがexclusive measurement lock下で全build/test/GPU/cleanupを直列管理する。
測定中はactualsource/linkinputs/client/modelを固定し、canonical v3のMain reviewとLuna
read-only fixture/failure-state調査R723/R724を並行する。source改変やcompileはこのcohortの閉鎖後。
各sampleの正常exit、所有退去、freshfault、個別PP/TTFTと実requestを保存し、後に比較する。
この文書の初版は計測前のplanであり、clean合格・速度採用・600/1000達成を主張しない。
