# 統合した4ヘッド並列QSAの検証

新しいfull engineと実API fixtureのビルドに成功した。
source `e3323ed07a95df3e0d42432579a1a8285722f80f`、engine SHA256 `cbb506a4b68658822e865190d7b5f8286c2aa290080c48436de948d5d22a1672`。
[build記録](qsa-score-headparallel-engine-build-v1.json)に実際のcompile flagsとsource pinsを保存した。
FP32のcontractionは無効。元のpublic scorer、query8の逐次ヘッド、decode/短いprefillの
呼出先とtop-k選択は保持している。

11プロセス、計51ケース、1528行の独立top-k照合が全て通過した。
全スコア・IDは元のscorerと全ビット一致し、guard、入力不変性、finite、repeatを確認した。
既定／明示0／逐次query8／4ヘッド並列を比較し、query数1〜9、選択境界2051/2052、
physical262144の256/255queriesを含む。追加でactive=-1の全宣言pool指定、
符号付きゼロと同点スコアを検証した。全プロセスは通常終了し、新しいGPU障害は無かった。
[各armの記録](qsa-score-headparallel-entry-qualification-v1.json)と
[検証範囲](qsa-score-headparallel-entry-results-v1.json)を保存した。

`STRATA_SYCL_SCORE_QUERY_TILE=8` と `STRATA_SYCL_SCORE_HEAD_PARALLEL=1` の組合せで有効にする。
両方とも実行前に固定する。既定値は従来経路のままで、性能採用はしていない。
[API契約](../../../docs/SYCL_QSA_SCORING.md)に全active poolの初期化・不変性・寿命を明記した。

この結果は実モデルの262144入力、RESTORE、一般品質、持続decode、1000PP/70TGの達成を示さない。
次に公式bench.shを使い、同じ新binaryでヘッド配置だけを変えた実モデル比較を行う。
正常な重複ログ329件、7,400,911バイトは必要な証拠を抽出・照合して退役した。
[削除記録](qsa-score-headparallel-retention-v1.json)には全パス・サイズ・ハッシュを残した。
