# 現行row-major QSA: subgroup softmax単体検証

2026-10-17、同じArc B570/driver1.17.39395+14/LLVM22.1。
Mainのprivate generatorからcorrectnessとmatched timingの2binaryを完全compile/linkした。
生成source hashは事前source-review JSONの予告値と一致した。
現行score33 row-majorにFP32 subgroup max/sumを適用し、BF16保存・FP32 online stateを保つ。
production source変更や速度採用はまだない。

correctness binaryのcombined UR tracing+parameter-validation付きsmoke2case PASS、
続くcomponent11caseもPASS。kernel submission数は24/60で予告通り。
独立BF16 operand oracle、全logical prefix、actual allocation guards、input不変、
mask/missing/empty/NaN診断、domain/extent/alias拒否、同経路repeat bitsを確認。
physical262144 cells/cap2051と、同じphysical容量でlive13のprefix caseを含む。
cap17/3queryのmax_abs6.103515625e-05、max_budget_ratio0.9682716738が最大で、
他の今回caseはmax_abs0だった。事前の1BF16 ULP+2e-6基準を変更していない。
これは検査範囲内の結果で、一般の実モデル品質を証明しない。

別timing binaryのbaseline/current-optimizedとcandidate/subgroup両方も、
小さいcombined trace smoke各1caseでPASS、各8kernelだった。
同一binaryでstorage/input/scratch/packコストを保つtiming経路の初回確認に留める。
全4process正常終了/所有終了、fresh fault0、dumpなし。

次は8192query/width2051/physical32768、2warmup+10samplesを各arm3独立process、
base,candidate,candidate,base,base,candidateの順で測る。
合成selected-setはuniform spreadであり、実モデルのlearned選択の局所性ではない。
両armの独立oracleと全guard/finite/input/repeatを要求するが、
FP32 sum associationが変わるためcross-arm FNV/bit一致を合否にしない。
arm内のprocess間repeat hash一致は引き続き要求する。
research3件が閉鎖しnative reviewが終了してからclean timingを実行する。
全実モデル262144 context、RESTORE、cancel/reuse、品質、速度は未資格付け。

build: qsa-rowmajor-subgroup-build-v1.json。
correctness: own-qsa-rowmajor-subgroup-{smoke,component}-v1.json。
timing smoke: own-qsa-rowmajor-subgroup-bench-smoke-{base,rowmajor-subgroup}-v1.json。
next: run_qsa_rowmajor_subgroup_cohort_v1.py。
