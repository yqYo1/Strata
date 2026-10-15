# R789 レビュー（R790）

R789 は decision support として依頼範囲を満たし、1,756 語で指定の 1,400〜1,800 語に収まっています。フラット1D、row-feature tile、型付き16bit vector、producer/consumer fusionを比べ、coalescing、live range、丸め順序、検証手順と反証要因を示しています。ソース数も上限内です。R782の型境界、R783のGEMM/APIとは異なる残差combineの割当て・融合の問いで、重複調査は不要です。

本文が使うのはSYCL仕様、Intel oneAPI Optimization Guide 2025.2、ACM論文、B570製品仕様です。コード移植やライセンス混入はありません。このレビューでは外部URLとカーネル実装を再検証していません。報告書とspecはローカルコード確認元を `662e3a1e` と記載しますが、ここではGitやソースを読んで確認していません。Orcaの端末識別子にある `@@b555b954` の意味も未確認です。したがって、実装に関する記述はMainが完全な不変revision上で再確認するまで報告書由来の観察として扱ってください。

Mainが今回参照するのは次だけです。

- `## Geometry and current dataflow` — HC4、N=2560、D=10240、8192-row chunkと型境界。
- `## Fusion choices and constraints` の **Fuse the producer, combine→write** と **Fuse the consumer, write→next norm**。
- `## Recommended next experiment` — 後続比較のゲートと測定条件のみ。実測結果ではありません。

融合の数値境界では、combineのFP32 expert/shared計算をBF16へ丸めてから残差更新が `bo` を読む点と、writeのFP32残差FMAをBF16へ丸めてからnormが値を読む点を保つ必要があります。`write→norm` を融合する場合も、norm reductionの入力は未丸めFP32値ではなく、保存すべきBF16残差を丸め戻した値です。注入係数とreduction/scaleはFP32のままです。これは本文の演算契約を読んだ結論で、現在のソースコードや実行結果を独立確認したものではありません。

報告の160/40/160 MiBは、指定shapeとdtypeから求めた論理テンソル量です。cache再利用後のDRAM traffic、ピーク常駐量、性能を表しません。subgroup順序、vector lowering、register/spill、occupancy、B570上の性能・品質・全長context適合性も未確認です。producer fusionは `bo` に他consumerがない場合に限って中間materializationを除けます。

次の品質誤差局所化調査は条件付きにします。現状は孤立typed-postops fixtureがPASSし、whole-engine buildはlink中で、GEMM・モデル品質・速度は未評価です。全体buildとfault-free機能実行を終え、同一自然長入力でFP32/native16の差が実際に観測された場合のみ新しい調査を依頼してください。品質が通れば依頼しません。R782/R789や以前の一般的な精度調査を繰り返さないよう、現行producer/consumer境界の差の原因局所化に絞ります。

運用要約（400字以内）：

> R789は範囲内の実質的な比較です。HC4の形状・型境界と、producer/consumer fusionの2節だけ参照。融合後もboと残差のBF16丸めを維持し、normは丸め済み残差から計算してください。MiBは論理量で、DRAM量・性能・品質・実機codegenは未確認です。品質差の原因調査はmatched自然長比較で差が出た場合のみ。
