# QSA score GEMM 本体比較

B57010GiB、同じビルド・モデル・設定で、公式bench.shの64K以上指定を使った。実入力78453、再利用0、context131072、chunk8192、thinking有効、INT8 KV、BF16 native16、MoEwindow10、残差は全量GPU。実験flag0/1だけを切り替えた。

| 経路 | 初回tok/s | 後続3回tok/s | 後続中央値 |
|---|---:|---|---:|
| 従来FP32 query8/head-parallel | 643.8 | 681.1 / 676.4 / 679.4 | 679.4 |
| FP16入力/F32出力GEMM | 682.1 | 725.5 / 720.3 / 725.2 | 725.2 |

初回は5.95%、後続中央値は6.74%改善した。1方式1プロセス、各プロセス内で初回と後続3回を分けた測定であり、独立プロセス反復のばらつきまでは評価していない。両方式の入力hashは一致し、モデル・runtime・メモリ配置・windowは照合した。GPU要求追加は独立status256byte、約42MBのscore作業領域は既存attn_h内の再利用で、別途panelを確保していない。

全engine/server/runnerは正常終了し、新規GPU障害0・dumpなし。スコアの丸めで選択・MoE job数は変わるため、その差も記録した。出力24tokenはthinkingを含む動作/性能確認で、生成品質や長時間decode70の資格付けではない。既定のflagは0のまま。

目標prefill1000/decode70には未達。次は現行経路のdevice service/resourceを再取得してボトルネックを更新する。長い生成品質と本体256K使い切り/RESTOREは別途必要であり、古い失敗結果は変更していない。次候補の検証用に、実際のrow-major serial/subgroup APIを同一fixtureバイナリで呼ぶ準備をした。過去のpacked実装やC16/C64の再試行はしない。

[構造化結果](qsa-score-gemm-entry-model-comparison-v1.json)と各実行receiptに全サンプル・環境・所有終了・fault確認を残した。
