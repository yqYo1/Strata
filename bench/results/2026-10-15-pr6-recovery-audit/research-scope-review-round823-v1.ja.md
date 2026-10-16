# R820 スコープレビュー

R820はknowledge collectionとして指定予算内の1,731語です。Mambaの中間chunk状態を別dtypeで保存しながらFP32 carryとfinal stateを保つ経路と、FLAの学習backwardで中間値を再計算する経路を、関数・テンソル型・状態境界まで追っています。どちらも実装契約の調査で、B570の速度、精度、採用を主張していません。

ピン留めされたMamba/FLAのソースを照合し、記載された状態保存、FP32局所演算、FLAの再計算経路を確認しました。Intel SYCL matrix仕様は可変の`sycl` branchで、実験的なsubgroup APIと実行deviceごとの型・shape照会を示しますが、BMGには触れていません。「仕様のBMG-family combinations」という記載は削除するか、対象runtimeで要確認の仮説として直してください。scalar affine経路の優先やStrata形状差も、性能結果ではなく推論・ローカルsource主張として扱います。

R759のglobal scan schedulingやR782の一般的なdtype/state契約とは問いが異なります。R773はchunk size、tail、workspaceの判断を既に扱い、R820の将来実験案もchunk-size/tail sweepで一部重なります。実験する場合はR773の選定済みchunk条件を使い、dtype境界または再計算だけを切り分けてください。R786のgraph fan-inとは重なりません。

現在Mainが進めるvLLM Xe2 GEMM qualificationやR821/R822の作業には原文読込は不要です。後でlong-prefill recurrenceの境界保存と再計算を具体的に選ぶ段階で検索資料として取得してください。次の知識収集は、Mainの同一入力比較でstate境界の誤差が実測された場合に限ります。現時点のeligibilityは`schedulable_now=false`です。indexは変更せず、提案metadataをJSONに記録しました。

確認した一次資料: [Mamba pinned source](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/triton/ssd_state_passing.py), [FLA pinned chunk source](https://github.com/fla-org/flash-linear-attention/blob/f470469e3192112814df229a4221454a195666a6/fla/ops/gated_delta_rule/chunk.py), [Intel SYCL matrix specification](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc).

運用digest（400字以内）:

> R820は1,731語で範囲内。Mamba/FLAの型契約を確認し、現時点でMainの読込は不要。Intel仕様にBMG記載はなく修正対象。R773とchunk/tail実験が一部重なる。後続は実測でstate境界誤差が確認されるまでschedulable_now=false。
