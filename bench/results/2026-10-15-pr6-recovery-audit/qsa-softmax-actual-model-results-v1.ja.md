# 実モデルでのrow-major subgroup softmax比較

2026-10-18 JST。Arc B570 10 GiB、Ryzen 5600X、RAM128 GiB。
実行ファイルのSHA256は
`1e7bf1cb5a7a8ec9b91e46bedbc2d7b7de3f0f7cbddbeeddecee80461a688b26`。
両方式ともcontext131072、chunk8192、BF16 hidden stateをGPU保持、
native16、score GEMM有効、MoE window10。変更したのは起動前の
`STRATA_SYCL_PREFILL_QSA_MATH`だけで、実行中の環境変数変更はない。
各方式で新規エンジンを1個起動し、入力の異なる4リクエストを順に実行した。
対応するリクエストのSHA256は方式間で一致する。

全入力78453 token、実際のbatched prefill78452 token、再利用0。
公式bench.sh系の入力・計測経路を使い、thinking有効、生成24 token。
初回と、同じプロセス内での後続3回を区別する。

|方式|初回PP|後続PPの全サンプル|後続PP中央値|初回decode|後続decodeの全サンプル|
|---|---:|---|---:|---:|---|
|serial optimized|682.7|725.8 / 721.5 / 725.7|725.7|14.0|14.1 / 14.1 / 14.1|
|subgroup|689.0|734.1 / 728.8 / 733.5|733.5|13.8|13.1 / 12.9 / 13.7|

単位はtoken/s。PPの後続標本標準偏差は2.454と2.902。
PP中央値の改善は1.0748%、対応する後続リクエストの短縮時間の中央値は
1.1473秒だった。後続3回は同じプロセスを共有しており、独立プロセス間の
ばらつきまで取り除いた結果ではない。

24 tokenのdecodeは候補側が遅かった。この観測をばらつきとして捨てない。
decodeの実装コードは変えていないが、この測定だけでは差の原因を特定できず、
長い生成の性能も証明できない。prefill改善だけを理由にdecodeを採用しない。
今回は既定値を変えず、両方式を実験選択肢として保持する。

8リクエストとも正常終了、thinking出力を観測、出力・reasoningの指紋は
対応する方式間で一致した。ただしこれは一般的な品質評価や同一出力の
採用条件ではない。プロセス・GPU所有権は終了し、新規GPU障害0。
物理262144 contextの全消費、RESTORE、長い生成の品質・速度は未検証。
既存の大きなクエリ集合での数値失敗を解決したという主張も含まない。

1000 token/sには、この78453 tokenの測定で約28.507秒の追加短縮が必要。
同じ実行経路のプロファイルでattention本体の120回・37.078秒を
長大prefillに結び付けたため、次はKのglobalメモリから共有メモリへの
読み込み順が候補になった。現在はGDN chunk64の独立実装を検証準備中。
カーネル時間はDRAM帯域や演算器の飽和率ではない。

全サンプル、入力・出力指紋、正確なバイナリ・ソース・起動環境、終了・
障害確認は[qsa-softmax-actual-model-comparison-v1.json](qsa-softmax-actual-model-comparison-v1.json)と
起動入力は[qsa-softmax-measurement-inputs-v1.json](qsa-softmax-measurement-inputs-v1.json)、
ビルド入力は[qsa-score-gemm-entry-engine-build-v2.json](qsa-score-gemm-entry-engine-build-v2.json)。
2個の元レシートと測定スクリプトは、削除前コミット
`8063aacc6749d18089f7a53db5d3e47b745f348c`から取得する。476件の
プロジェクト入力ハッシュがそのコミットの同じパスと一致することを確認した。
当時の実行コマンドは
`python3 bench/results/2026-10-15-pr6-recovery-audit/run_qsa_softmax_actual_model_v1.py serial --rep 0 --prefill-tokens 65536`
および`subgroup --rep 0 --prefill-tokens 65536`。保存先を再利用しないため、再測定時は新しい版の
レシピを作り、前回のレシートは変更しない。
