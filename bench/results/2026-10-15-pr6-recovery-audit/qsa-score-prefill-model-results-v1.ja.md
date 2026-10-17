# 実モデルでのQSA query8比較

同じ新エンジン（SHA256 `102f32f1960a78a7101d3eee80115182043a7fa1ef8e676ce379a82de84365b2`）で、
`STRATA_SYCL_SCORE_QUERY_TILE=0` と `8` を比較した。
Arc B570 10GiB / Ryzen 5600X、driver 1.17.39395+14。
公式 `bench.sh` のprefill-only拡張で最低65536を指定し、実入力は78453トークン
（エンジンのバッチ部分78452）、context/KV131072、chunk8192、INT8 KV、
GPU BF16残差、vLLM由来native16演算、MoE window10、thinking有効。

| scorer | 初回 PP | 同一プロセスの後続3回 PP | 後続中央値 |
|---|---:|---|---:|
| 従来のFP32（tile0） | 616.2 | 650.3, 646.6, 651.6 | 650.3 |
| キーを8クエリで再利用（tile8） | 573.0 | 664.3, 661.2, 663.8 | 663.8 |

単位はtoken/s。後続中央値は2.08%向上した一方、初回は7.01%低下した。
各armで別のエンジンを1回起動し、その同一PIDで4要求を処理した。
要求ごとの入力ハッシュが両armで一致し、毎回prompt reuseは0。
MoEのwindows/dequants/jobsとmaps転送量も一致した。
両armとも通常終了し、新しいGPU障害は無かった。

初回は各armにつき1サンプルであり、分散や悪化の原因はまだ確定していない。
初回と後続の差にはコンパイル、キャッシュ、確保、グラフ等が入り得る。
この測定からJITが原因とは断定しない。SYCL persistent cacheは無効で
XDGディレクトリも別だが、全ドライバーキャッシュのcold状態を証明していない。
初回悪化が残るため既定値は0のままとする。

24生成トークンは終了確認用であり、持続decode性能や品質の評価ではない。
この短い出力の一致は一般品質・262144モデル入力・RESTOREの合格根拠にしない。
`source_H2D_bytes=150627840` はmaps部分だけであり、全expert転送量ではない。

[個別測定と比較判定](qsa-score-prefill-model-comparison-v1.json)に
入力・出力fingerprint、各サンプル、起動条件、元receiptのハッシュを保存した。
次は4ヘッド並列の成否を確定し、初回スコア投入の時間を切り分ける。
目標1000 PP / 70 TGは継続中。
