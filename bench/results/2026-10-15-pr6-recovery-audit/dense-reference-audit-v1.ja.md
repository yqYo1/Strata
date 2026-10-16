# B570の常駐Denseモデル実測との比較

2026-10-16。ユーザー提供のQwen3.5 9B実測は **966.57 ±10.59 tok/s**。
`llama-bench -fa 1 -n 0 -p 131072 -ctk q8_0 -ctv q8_0`、
Qwen3.5-9B-UD-Q4_K_XL.gguf、SYCL、全層GPU指定、build abeada335 (11455)、
B570、driver1.17.39395+14。131072は確保contextだけでなく実際の入力token数。
この値はユーザーの測定結果として保持し、Mainが再測定したとは扱わない。
個別sampleは提供されていないため、平均とばらつきから生成しない。

## 77%の意味の訂正

既存profileの77.31%は、最初から最後の長入力QSAまでの94.945秒に対する、
compute command実行区間の和集合73.403秒の比率。**演算器の能力の77%でも、
全prefillの77%でもない**。kernel内の待ち、gather、SLM/barrier、padding、
行列の埋まり具合を区別していない。residentモデルの速度に0.77を掛ける根拠にはならない。
GPU内部の効率が良いことを示す値として説明した点を訂正する。

元profileの測定値は変更しない。hostで測った全prefillは105.007秒、
QSA60回のdevice service合計は33.963秒。両clockによる全prefill区間の対応は
未証明で、service合計をそのまま削除可能時間とも呼ばない。copy coverage9.11%も
同じattention envelopeでの値。この結果はPCIeだけが支配的という説明を支持せず、
GPU処理の効率を調べる必要を示す。

## 実物とsourceで確認した比較条件

両モデルのGGUFヘッダーと、提供build番号に一致するローカルllama.cpp revisionを照合。
weight payloadは読み込まず、header全体のSHA256・stat identityと関連sourceのhashを記録。
installed binaryもhashを記録したが、現在のsourceからこのbinaryをbuildしたという証明は別。
実行中のkernelを観測していないため、oneDNN/MKL等の有効flagから実dispatchを断定しない。

| 項目 | Qwen3.5 9B / llama-bench | Qwen3.8 Flash Next / Strata |
| --- | --- | --- |
| layers | 32 | 48 |
| GDN / full or selected attention layers | 24 / 8 | 36 / 12 |
| GDN value heads、各headのstate | 32、128×128 | 48、128×128 |
| FFN | Dense | 512 experts、tokenごと10選択、shared expertあり |
| hidden stream | embedding4096 | embedding2560、hyper-connection4本 |
| 今回の入力 | 131072 | 既存profile/clean referenceは39262 |
| compute batch | source既定microbatch512、batch2048 | chunk8192、uniform仮定でexpert平均160行 |

GDN state要素数は全層合計でStrataが2.25倍。これは形状の計算であり、GDNが
実際に2.25倍の時間を使うという測定ではない。active parameter数はこれらの
非weight処理や分割GEMMの効率を表さないため、A6BとDense9Bの速度比は直接決まらない。
ただし、architecture差だけで現在の性能差を説明できたとも扱わない。

同revisionのllama-benchでは既定5反復の前に**全131072 tokenをwarmup**し、
sampleごとにmemoryをclearする。model/context作成・warmupは計時外、prompt処理と
最後のGPU同期は計時内。入力はrandom token。Strata公式benchの自然文、短いwarmup、
layer-major転送込みの処理とは異なる。この差の寄与は未計測で、差の原因と断定しない。
また計装run105秒とclean reference約403tok/sを同じsampleとして混ぜない。

## チューニングへの反映

966.57tok/sを、このB570が有用な長入力処理能力を持つ実測基準として採用する。
「GPU computeが77%だからGPU演算は十分速い」という判断はしない。
優先対象は長いQSAと、細かく分かれたexpert行列処理。既存profileのGEMM投入scopeは
123192回で、CPU grouping自体は0.106CPU秒。一方QSAだけでdevice service約34秒ある。
groupingのCPU使用率やPCIe帯域だけを削って解決するという見通しにはしない。

QSAはFP32精度を近似維持するhi/lo FP16の2回のmatrix積、per-query選択/gather、
online softmaxと同期を含む。matrix命令を使うだけで高効率とは限らない。
既存のnative16候補を無条件に再採用せず、数値品質と反復した実速度で選ぶ。
現在の8192-query固定形状で、working-setとcommand分割を別々に変えるcontrolled測定を
完了し、既存の128-query診断から本番のbatch不足を推測しない。
physical VRAM traffic、実XMX activity、SLM/barrier stallの支配はなお未確定。

再計算: `python3 bench/results/2026-10-15-pr6-recovery-audit/analyze_dense_reference_v1.py`。
出力がある場合は上書きせず終了する。新たなGPU実行やモデルのコピーは行わない。
