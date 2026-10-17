# 約78K入力のMoE window2/8反復比較

同じ新engine binaryで各3回、別プロセスを起動して交互に測定した。
公式bench.shの64K最小入力を使い、実際の入力78453 token、batched78452行、
thinking有効、生成24token。Ryzen5600X、RAM128GiB、ArcB57010GiB、
xe1.17.39395+14。context/KV-resident131072、chunk/operator8192、全batched
BF16 residualをGPUに置き、FP16 matrix inputとBF16出力/FP32局所累積を使う。
QSAはbf16-prefix-optimized。TRACE、UR layer、coarse/QSA/CPU/command profileは無効。

|MoE経路|個別PP(tok/s)|中央値|標本標準偏差|
|---|---|---:|---:|
|window2|576.1 / 576.4 / 576.4|576.4|0.1732|
|expert-major window8|602.0 / 601.5 / 603.2|602.0|0.8737|

中央値は4.441%改善し、新経路の全3件が600を超え、かつ旧経路の最大値を上回った。
この入力・設定では、反復したprefill600超えを確認した。1000tok/sと持続decode70tok/sは未達。

新経路は65536行+12916行でFFNを実行し、expertの展開回数は65543から31665、
job数は66696から34224へ減った。各カーネルは8192行以下。
全件で同じ要求hash、cache再利用0、thinking応答、正常終了、所有プロセス消滅、
新規GPU fault/dumpなしを確認した。元の6receiptは変更せず、Mainが別にhashと
プロセスstart-timeを再検証した。

24tokenのthinking hashは一致したが、一般的な品質や長いdecode速度の証明ではない。
物理262144コンテキスト全消費とRESTOREも、この比較では未検証。
mode8はopt-inとして統合し、次は別opt-inのwindow10で78452行を一つにまとめる。
速度を理由に既存の安全性・状態・長contextの検証を省かない。

コマンド・環境・各測定値・fault/ownership証拠は各sample JSON、
集計と独立再検証はmoe-accum-model-clean-main-analysis-v1.jsonにある。
