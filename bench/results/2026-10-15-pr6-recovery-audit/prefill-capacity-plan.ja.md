# 長大prefillのVRAM容量計算

同日の後続SYCL workspace削減による差分表とnorm常駐の補足は
`prefill-no-workspace-review.ja.md`にある。以下は元v2 receiptのsource/仮定を固定した表である。

2026-10-15。GPUでdecodeしない間はdecode用cache/head/MTP full-draft重みを保持しない、という
ユーザーの仮定に基づく。さらにlayer-majorで**現在の一層の重みだけ**をGPUに置き、
他層のdense重みも必要な層になってから読む。次層用の二重weight bankは持たない。
これは現serveの所有配置から変更する設計上の試算であり、現実装の物理最大contextの実測ではない。

1 GiB=2^30 B、256K=262144 token。A16はここでは層境界に保存するhidden/residualをFP16とし、
既存のFP32作業領域・累積は維持する意味で使う。全operator bufferをFP16へ変えた数字ではない。
FP16/BF16の保存容量は同じだが、品質は別に評価する。

## 実ファイルと実行ログから確定した量

Mainが実packのindex/native_expertsとGGUF二shardのheaderを読み、48層を個別に集計した。
tensor payloadは読まず、header hash、file identity、各tensorのshape/type/bytesを結果に保存した。
native300 matricesの合計2116951040 B、canonical pool1538035200 Bは直前nativeのload表示と整合する。
canonical/nativeの二重計上はしていない。

最大の**一層全体**はzero-based layer47（第48層、QSA）である。

| その層の重み | bytes | GiB |
| --- | ---: | ---: |
| 全512 expert、元のpacked量子化形式 | 1363148800 | 1.2695 |
| canonical dense/norm/GR等 | 32366592 | 0.0301 |
| native dense projections | 44646400 | 0.0416 |
| 合計 | **1440161792** | **1.3413** |

この合計に収まる共通weight bankを層ごとに詰め直す仮定を主表に使う。
expert bankとdense bankを別々にそれぞれの最大量で固定確保する場合は1505173376 Bとなる。
dense最大はPLEを含むlayer1で、expert最大と同じ層ではないためである。
その保守的な別bank案もJSONに記録した。
token embeddingとPLE巨大lookup tableは従来どおりCPU RAM/mappingにあり、このVRAM重み合計に含めない。
PLE key/valueなどGPUで使うprojectionは該当層の重みに含めた。

既存カーネルの明示的なprefill作業領域は次のとおり。

| chunk | GDN/MoE/QSAの共用領域最大 | 全作業領域、full-resident KV条件 |
| ---: | ---: | ---: |
| 4096 | 607128320 B | **1386006784 B = 1.2908 GiB** |
| 8192 | 1180698624 B | **2596783616 B = 2.4184 GiB** |

三系統は同じ領域を順に使うため、和ではなく最大を取る。
全作業領域にはlive FP32 R、FP16 activation/dequant、dense-dequant64 MiB、reserved GEMM workspace32 MiB、
DQ2組のexpert scratch計19660800 B、ring8、PLE chunk buffers、各Allocのguard/alignmentを含む。
全512expertをFP16へ展開して保持する4.6875 GiB案ではない。

Mainの独立したAlloc再計算は、直前の実C4096/streamed-KV実行ログの
accounted1662831872 B、shared607128320 Bと**完全一致**した。
full-resident KVなら不要になる一層分の全context KV stage276824064 Bと四allocationのguard1024 Bを除くと、
1386006784 Bになる。C8192は同じsource sequenceからの計算であり、新たなGPU実測ではない。
`bytes_needed`は未使用xn等も数える保守的なloan sizeなので、実carveの値と混ぜない。

実nativeが報告したVRAM総量は10666115072 B = **9.9336 GiB**。
これから最大一層、上のwork、全36層GDN state117669888 B、PLE history368640 B、
token-ID bufferを控除する。小owner用16 MiB、driver/oneMKL等の余裕512 MiBも別途仮置きした。
従って**KVと保存hiddenに使う予算**はC4096で7168243456 B = **6.6759 GiB**、
C8192で5957450240 B = **5.5483 GiB**となる。
この16/512 MiBは設計上の余裕であって、vendor内部の最大確保量を実測した上限ではない。

## KVとhiddenの単価

このモデルは48層中12層がQSAで、各層KV heads2、head dimension256。
K8/V8は64値ごとのFP16 scaleも含め一token一QSA層1056 B。
K8/V4はK528 B + V288 B =816 Bである。
さらにFP32 pooled indexer、page table、共有RoPE、tail/count等を加える。

hiddenはhyperconnection4本×2560要素で、FP16なら一token **20480 B**。
layer-majorの一つの全prefix bankを層境界で更新する。47個の全prefix bankを保持する計算ではない。

| context | FP16 hidden | K8V8全12QSA state | K8V4全12QSA state |
| ---: | ---: | ---: | ---: |
| 32768 | 0.6250 GiB | 0.4418 GiB | 0.3539 GiB |
| 65536 | 1.2500 GiB | 0.8836 GiB | 0.7078 GiB |
| 262144 | **5.0000 GiB** | **3.5342 GiB** | **2.8311 GiB** |

256KでKV payloadだけならK8V8は3.09375 GiB、K8V4は2.390625 GiB。
indexer等を落とすと容量を過大評価する。
4の倍数のCについて全QSA stateはK8V8=`14476*C+43200`、K8V4=`11596*C+43200` B。
従ってhidden込みはそれぞれ`34956*C+43200`、`32076*C+43200` Bである。

## 求める二種類の上限

左の上限はKV容量も同じ長さCに設定して、全hidden C行もGPUに置く場合。
右はKVを最初から262144容量で全12層確保した上で、hiddenだけ何行置けるかである。
両者は異なる条件なので混同しない。
作業領域は両列とも262144-capacityに対応する最大サイズのまま予約した。

| chunk | KV形式 | KVと全hiddenを同じCで保持できる計算上限 | 256K KVを保持した場合のhidden行数 |
| ---: | --- | ---: | ---: |
| 4096 | K8V8 | **205060 token** | **164716 token** |
| 4096 | K8V4 | **223472 token** | **201580 token** |
| 8192 | K8V8 | **170424 token** | **105596 token** |
| 8192 | K8V4 | **185724 token** | **142460 token** |

chunk4096では256KのKVと全hiddenを両方保持するために、K8V8でさらに1.8583 GiB、
K8V4でも1.1551 GiBが不足する。C8192はworkspace増加分だけさらに厳しい。
別々の最大expert/dense bankを固定確保する保守案では、C4096/K8V4の右欄は198406行までになる。

ただし容量が足りないことと、全hiddenをRAMで47回往復させる必要があることは同義ではない。
**prefill中のQSA KVも現在の一層だけGPUへ置く**別案では、256K hidden全体5 GiBをGPUに残せる計算になる。
C4096ならK8V8でも余裕約1.3241 GiB、C8192なら約0.1965 GiB（後者は余裕が小さい）。
この数字はKV payloadだけでなくindexer/page stateも**現在の一層分だけ**GPUへ置く場合である。
他の11層のindexer/page stateをGPUに残すなら、追加372021936 Bを引く必要がある。
その場合C4096は1049759296 B（約0.9777 GiB）余るが、C8192は161033920 B不足する。
従って「一層分のKV payloadにすればC8192も全indexer常駐のまま収まる」とは言えない。
過去層KVは層完了時に一回RAMへ保存し、decode前に必要な配置へ戻す必要がある。
KV/indexerとhiddenの入替費用は別途実測するが、層をまたぐhiddenの反復転送を減らす設計として検討できる。
これは現layer-majorのall-state所有をそのまま使う案ではない。

## 検証範囲と保存物

K8V4はshared CLI/size計算にはあるが、現layer-majorはkv_hybridを拒否する。
今回のK8V4表は**仮定の容量**であり、動作・品質・速度の合格ではない。
FP16 boundary storageは既にsourceにあるが、実用品質は未合格。
MTPのprompt KV生成も後段に回し、長大target prefill中にMTP重みを保持しない仮定である。
decode準備・最大contextの実物理実行・RESTOREは別に検証する。
今回のために新しいGPU推論や大量確保は実行していない。

実行コマンドは`python3 -B .../measure_prefill_capacity_v1.py`、続いて`.../measure_prefill_capacity_v2.py`。
v1のPLE history仮定2行を実定数9行へ訂正し、v2を採用した。v1の元bytes/statusは保存する。
R677 v1にはGEMM64 MiBを32 MiBと読む誤りと、streamedの値をresidentと呼ぶ誤りがあり、
Mainの指摘後に別v2で訂正された。Main実再計算を優先する。
R676の「FP16 storage未実装」の表現も訂正する。未実装だったのはBF16であり、FP16は実装済み・品質未合格。

再現source、全層の計量、header/input hash、各caseのbyte予算と境界検査は
`measure_prefill_capacity_v1.py` / `measure_prefill_capacity_v2.py`、`prefill-capacity-metadata-v1.json` / `v2.json`にある。
Mainの再計算はC4096 observed allocationとの一致、両KVの全context式、上限の次page/次hidden行が予算を超すことを確認した。
新しいGPU allocation/実model最大context/quality/performanceの合格へ昇格しない。

直前の公式bench.sh prefill-onlyは実78453入力・reuse0で219.1/222.3/221.5 tok/s、中央値221.5。
三回とも完了、native/server0、全owned command退去、新規fault0、dumpなし、request journal valid/complete。
`prefill-only-clean-iq1-c4096-v1.json`に個別標本と閉鎖証拠を保存した。
