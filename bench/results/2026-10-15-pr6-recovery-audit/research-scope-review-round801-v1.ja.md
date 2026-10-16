# R801 R798レポートレビュー

R798は1,480語、6資料で割当範囲内です。細い行列、ragged batch、大規模dense GEMMを扱い、regular tile・padding・grouping・K分割の利点と費用を対比しています。比較表と限定した実験案があり、Xe2/SYCLへの転用は仮説と明記しています。調査範囲と深さは適切です。

R796のoneDNN API/型/queue契約、R790の残差mapping/fusionとは調査軸が異なります。index上のR778 sparse GEMM、R779 gatherとも重複しません。R628はGEMM scheduling census、R701はlocal grouped GEMM/prefillとして記録されています。各本文は再読していないため、全ての旧GEMM資料との重複なしとは断定しません。これ以上の一般的なtile/padding/grouping調査は依頼しないでください。

6件のURL、版・取得日、文献中のhardwareとprecisionは報告内にありますが、本レビューでは外部ページやハッシュを独立確認していません。従って資料の記述、取得hash、Xe2への推論は報告者記載として扱います。Strataのbottleneck、速度、品質、最適tile、実機codegenを示す結果はありません。研究indexは変更していません。

Mainから渡された現状ではnative16のsynthetic 129-token差は品質根拠として不採用、matched BF16速度は未確定、tile schedulingの実測問題も特定されていません。このため原文の即時参照は不要です。indexに追加するretrieval metadataはJSON内に記載しました。具体的なdense GEMM schedule判断が実測で開いた場合のみ「Shape-dependent cases」「Intel Xe2/SYCL transfer assumptions」「Bounded discrimination experiment」を参照してください。

次調査候補は条件付きです。B570上のsource/binary固定true-16 fixture、dtype/accumulationを含むcorrectness contract、clean processでの再現可能なtile-tail/padding/underfill費用、及びMainが決めた具体的なschedule判断が揃うまでは `schedulable_now=false` です。現在の68 GEMM PASS、synthetic model差、FP32 n=1測定だけでは条件を満たしません。

入力と成果物のSHA-256、Orcaのlaunch/settlement/release状態、eligibility metadataは `research-scope-review-round801-v1.json` に記録しました。
