# native BF16 自然文32Kの数値スクリーニング

同じ v2 バイナリ/ソース9c1a2020、同じ自然文32863入力token、C8192、GPU residual32768行、物理context262144、thinking有効で FP32-a→FP32-b→BF16 を直列実行。3armとも48layer/5chunkを完了し正常終了、fresh GPU fault 0、driver dumpなし。元の separate-copy-queue regression FAIL は変更せず、今回も qualified singlequeue の範囲。診断付き実行のため速度採用のサンプルにはしない。

255+512*i の64位置で全248320語彙の有限logitsを取り、Python FP64/math.fsum の独立 logsumexp で next-token NLL、分布差を算出した。全位置の個別結果・hashは `own-native16-quality-screen-v2-v1.json` に保存。Mainは全64位置の差、正常終了・fault証拠、2FP32 controlと両BF16ペアを確認した。

| 指標 | FP32-a/b | BF16またはFP32→BF16差 |
| --- | ---: | ---: |
| sampled mean NLL | 5.057467403 | 4.886277845 |
| mean signed NLL差 | control同士0 | -0.171189558 |
| mean absolute NLL差 | control同士0 | 0.555571641 |
| max absolute NLL差 | control同士0 | 2.647024878 |
| top1一致 | control同士64/64 | 59/64 |
| mean KL(FP32→BF16) | control同士0 | 0.053297457 |
| mean L1分布差 | control同士0 | 0.147631405 |
| max absolute logit差 | control同士0 | 10.500203609 |

FP32-a/bは今回の64全語彙行がbyte一致した。これは未抽出位置や過去の別ソース/条件で観測した2か所のFP32 NLL差の解消を証明しない。BF16の標本平均NLLは下がったが、単一のプロジェクト文書コーパスから抽出した64位置であり、連続した全文PPL、一般能力、長い生成やタスク成績の改善とは解釈しない。サンプリング生成の文字列一致を品質ゲートにはしていない。許容品質しきい値は設定・採用していない。

今回の custom native16 GEMM は既存oneMKLより5.57–30.66倍遅い単体測定、78Kの初回long比較でも346.5→123.8tok/sの速度低下が別途確認済み。default FP32 は維持する。次は実際のvLLM Xe2カーネルとoneDNN候補を、Mainが単体から順に検証する。QSA/GDN内部のtyped16置き換え、実入力262144境界とRESTORE、複数文書/タスクでの能力検証も未完了。

原始captureの保持消費者は、R810/R804の数値比較用の FP32-a と現custom-BF16の64全語彙行。FP32-bはbyte同一のため別コピー保持の必要性を整理する。削除は専用retention manifest/lockで行い、この原測定receiptのbyte/statusは変えない。
