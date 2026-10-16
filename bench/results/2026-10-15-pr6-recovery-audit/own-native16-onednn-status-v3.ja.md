# oneDNN BF16補助候補の結果

Mainの3回目build/smokeはPASS。実B570/LevelZero同一queueでBF16→BF16の小行列を実行し、independent oracle/guards/実際のmapped DSO確認まで完了。selected implは `jit:gemm:any`、strictFP32 attrs、scratch/primitive-owned queryとも0。実XMX・内部全FP32領域ゼロ・高速化を証明したものではない。

`own-native16-onednn-check-v3.json` はFAILのまま。negative suiteはPASS、numericsは9有限dyadic形状+midpoint+cancellationを完了後、12番目のidentity/subnormal fixtureで `exact BF16 value/zero/subnormal changed`。原文はexactnessの失敗までで、got/expected bitsや具体rowを出していないため、FTZまたはzero sign等と断定しない。native process exit1はCPU側oracleの例外をcatchしたもの。全所有processは通常の観測/reap/sessionemptyで閉じ、killなし、fresh GPU fault0、dumpなし。overall normally_closedは全exit0を要求するfieldのためfalseで、ownership_closed=trueと区別する。

chain/lifecycle/scratchはこのcheckでは未実行。採用/性能/モデル品質は未承認。主候補をvLLM Xe2とし、oneDNNのstrict exactness差の追加bits診断は必要になった時に限定する。旧Gemm exact-RNE/68ケースや失敗結果を緩めて通過扱いにしない。

oneDNN3.11.4公式data-types文書 https://uxlfoundation.github.io/oneDNN/v3.11/dev_guide_data_types.html の469–510をMainも確認。defaultのruntime FP環境とfaithful変換、低精度HWによって環境に従わずFTZとなる場合があるという一般的説明は、このB570の失敗値の特定ではない。

先行結果も保持: build-v1はconst SYCL eventのcompile FAIL、build-v2はPASS、smoke-v2は数値oracle/正常exit0だが未登録mapped DSOによりqualification FAIL。Mainはinstalled dependenciesを静的にpinし、実symbolのdladdrをPLT回避に修正してbuild/smoke-v3を取得した。元の各FAIL receiptは一切変更していない。追加ライブラリ環境は私有比較childだけで、標準FP32 engine設定は変えていない。

当面保持するconsumer: Mainが必要時に行うexact BF16差の再現/追加bits診断のため、私有v3binary/sourceとnumerics stdout/stderr（小さな1代表）を保管。その他成功trace/古いbuild duplicateは次のexclusive retentionレビューで整理する。
