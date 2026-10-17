# 実際のrow-major attention APIの検証

2026-10-18 JST、Arc B570／Level Zero 1.17.39395+14。
現行full buildのarchiveをリンクした同一バイナリから、immutableなtest selectorで
実際の `qsa_bf16_prefix_optimized_batch` と `qsa_bf16_prefix_subgroup_batch` を呼んだ。
private packed版はビルド・選択していない。ビルドと両方式のsmoke 2ケース、
component 11ケースは全て正常終了し、新規GPU障害0件、所有プロセスの終了を確認した。

BF16 Q/K/V/P/output、FP32 accumulation／online stateという計算契約は共通。
subgroup版ではmax／sumの結合順序が変わる。独立oracle、既存のケース・許容値は
変更していない。入力不変、出力・scratch guard、logical prefix、masked page、
不正幅、NaN診断、alias／domain／extent拒否と、その後の出力回復を確認した。
私有fixtureのhost DMA backingは、checked drainより長く保持するようにしている。

各smokeは24、各componentは60のUR kernel submissionを実測した。
両componentには262144 cell／cap2051のケースが含まれるが、そのケースは1 query。
以前のstrict numerical failure `q2h13d31` を解決したとは扱わない。
大きなquery数での物理256K model／RESTOREとtask qualityは未検証。
この診断付き実行から性能差は主張しない。

個々のoracle結果・環境・ELF mapping・終了・fault windowは
`own-qsa-softmax-actual-entry-{serial,subgroup}-{smoke,component}-v1.json`、
ビルドは `own-qsa-softmax-actual-entry-build-v1.json` に保存した。
[元の実装・fixtureの由来](https://github.com/yqYo1/Strata/blob/8063aacc6749d18089f7a53db5d3e47b745f348c/bench/results/2026-10-15-pr6-recovery-audit/own-qsa-softmax-actual-entry-v1/manifest.json)と、
[次のclean model比較の全差分](https://github.com/yqYo1/Strata/blob/8063aacc6749d18089f7a53db5d3e47b745f348c/bench/results/2026-10-15-pr6-recovery-audit/qsa-softmax-actual-model-provenance-v1.json)も保存した。

次のコマンドは順に別のengine processを起動し、公式bench.shで実入力約78K、
thinking ON、24 token probe、最初1回と後続3回を測る。
score GEMM1／MoE window10／context131072／chunk8192／全GPU residualは固定する。
モデル全体の生成出力のbit一致は採用条件にしない。

```sh
python3 bench/results/2026-10-15-pr6-recovery-audit/run_qsa_softmax_actual_model_v1.py serial
python3 bench/results/2026-10-15-pr6-recovery-audit/run_qsa_softmax_actual_model_v1.py subgroup
```

[削除記録](https://github.com/yqYo1/Strata/blob/8063aacc6749d18089f7a53db5d3e47b745f348c/bench/results/2026-10-15-pr6-recovery-audit/qsa-resource-softmax-retention-v1.json)で、閉じた成功実行のraw／重複
79ファイル、5,043,425 logical bytesを退役した。canonical receiptは変更せず、
現行engine・archive・actual-entry fixture ELFはこの比較のconsumer用に保持している。
通常のprefill中央値725.2 tok/sは前の測定値のまま。候補の速度はまだ未測定。
