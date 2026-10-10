# ハードウェアの到達性能と処理方式の予算

2026-10-10、Ryzen 5 5600X・RAM 128 GB・Arc B570で測定した。以下は指定した操作での到達性能であり、あらゆる実装に対する絶対的な最大値ではない。基本の帯域・演算測定は３つの独立したプロセスで繰り返し、表には各プロセス内中央値の中央値を使った。個別サンプル、プロセス間の範囲、測定条件は[一覧](component-capacity-matrix-v2.json)に保存した。GB/sは10進数。

| 部分・操作 | 到達性能 | 解釈に必要な条件 |
| --- | ---: | --- |
| CPU、６物理コアのFP32 FMA | 0.818 TFLOP/s | レジスタ内の演算。量子化推論の速度には換算しない |
| RAM読み取り | 36.87 GB/s | ６物理コア |
| RAMの非一時的コピー | 37.58 GB/s | 読み取り＋書き込みを合計した帯域 |
| expertのRAMコピー、３worker | 13.74 GB/s | payloadの片道バイト数。通常RAMのfixture、GPU転送を含まない |
| host USMからGPU／GPUからhost USM | 6.447／5.643 GB/s | payloadの片道バイト数 |
| pageable RAMからGPU／GPUからRAM | 4.605／5.274 GB/s | host USMと混同しない |
| VRAM内コピー | 330.07 GB/s | 読み取り＋書き込みを合計した帯域 |
| GPUのGU／Down GEMM、大きな行列 | 52.69／41.61 TFLOP/s | 8192行、FP16入力・FP32出力、重みは事前配置 |
| GPUのGU／Down GEMM、小さな行列 | 13.16／10.08 TFLOP/s | 80行、８組の重みを循環。cold cacheの保証はない |
| GPUでの量子化重みの展開 | 75.88～207.70 µs/expert | 実モデルの７形式。GU＋Down、投入から完了待ちまで。転送は除外 |
| CPUでの量子化expert処理 | 90.98→38.74 µs/expert-token | 22/20形式、同じexpertへの１→４行、pool実装のfixture |
| SSDの連続／ランダム4 KiB読み取り | 2.266／0.05099 GB/s | 16worker、対象ファイルのZFS経路、O_DIRECT |

GPUの量子化展開は３つの独立したプロセスで７形式×７サンプル、計147サンプルを取得した。各形式から32個の実expertを使い、独立したCPU参照との全出力ビット比較とcanary検査を通した。終了・所有プロセスの消滅・測定区間の新規GPU障害も確認した。計時はhostの投入と最後の待ちを含み、GPU kernelだけの時間ではない。probeのO2とproductionのO3の違いも残る。[個別結果と適用範囲](../2026-10-10-gpu-iq-dequant-capacity/README.md)。

PCIeの外部経路はupstream bridgeまで確認するとGen4 x4であり、128/130符号化後の理想値は片道7.877 GB/s、packet等のoverheadは別に必要になる。GPU function自身が報告する内部リンクの2.5 GT/s x1を外部帯域として使わない。[実際のトポロジーと参照資料](physical-bandwidth-reference.json)。H2Dの6.447 GB/sはこの外部経路に対して現実的な到達値である。

## 順番で改善できる範囲と、転送量を変える必要がある範囲

32K（32768位置）を1000 tok/sで処理する時間予算は32.768秒。以前のモデル測定ではexpertのH2D payloadが190.241 GBだった。この量を今回の独立したH2D測定6.447 GB/sで運ぶと29.50秒になる。PCIeの理想値7.877 GB/sを使っても約24.15秒であり、実際にはprotocol overhead等が加わる。この比較は現在のモデル経路の再測定ではなく、同じバイト量を運ぶ場合の条件付き予算である。

このモデルのexpert GEMMだけの仕事量は32Kで154.619 TFLOP。すべてが今回測った80行の条件なら約12.95秒、160行なら約6.57秒、640行なら約3.85秒、大きな行列の条件なら約3.19秒になる。attention、共有expert、量子化展開、router、host作業等は含まない。実際のexpert別の行数と選択されたkernelが分かるまで、モデルの所要時間とは扱わない。

転送と演算を十分に重ねられるなら、合計時間ではなく重なりを除いたcritical pathで判断する。一方、安全設定を維持した独立した２queueの対照試験では、記録した144転送と3168 GEMMのGPU時間区間の重なりは０だった。この結果だけであらゆる重ね合わせを否定はできないが、現在の設定で投入順だけを変えれば転送を隠せるとも言えない。[対照試験](../2026-10-10-hardware-concurrency/README.md)。

現在のpackのmetadataでは、全48層について512 expertをそれぞれ１回だけ読むpacked重みは50.292 GBである。以前の190.241 GBに対して約3.78分の１、今回のH2D値なら7.80秒。ただし、packed重みの再利用とFP16に展開した重みの再利用は別である。generic経路で毎chunk展開すれば、その展開は残る。MMQ・fused・peer経路にgenericの測定値を適用しない。

layer-majorで残差を全てRAMに置く場合、残差幅はD=10240であり、32KのFP32残差は1.25 GiBになる。47層境界の往復を今回のpageable転送値で見積もると25.66秒。重み7.80秒との直列合計は33.46秒で、演算前に32.768秒の予算を超える。重なりが作れない場合は、順番に加えて残差のGPU配置、往復回数、重みの再利用範囲を変える必要がある。

例えば実効8K位置のGPU残差prefixは320 MiBで、その場合のRAM側残差転送見積もりは19.24秒になる。inplace化でGPU側の残差コピーを省く余地もある。これは配置候補の算術であり、必要な同時allocationが収まることや数値一致を確認した結果ではない。[配置別の条件付き計算](mixed-residual-placement-estimates.json)。

## 256Kで必ず確認する容量

262144位置のFP32残差全体は10,737,418,240 B（10 GiB）になる。今回のdevice capacity 10,666,115,072 Bを、他のallocationを加える前から超えている。残差全体をVRAMに置く方式はこの容量では成立しない。

現在のlayer-major経路はQSA KVの常駐も必要とし、full contextのQSA KV・indexer・RoPEだけで6,912,212,992 Bになる。元のdecode cache、MTP、session/prefill scratch、dense weights、temporary layer cache、graph/runtimeの生存期間が重なるため、単純にVRAM容量から１つのbufferを引いて「収まる」とは判定しない。cacheの解放とRAMからの復元はqueueの完了とgraphの退役を含めて検証する。

短い入力で通る配置でも、物理262144位置までの容量・state・数値の検証が必要である。今回のcomponent測定はfull-context推論の成功を意味しない。

## 次に照合する実行経路

prefillでは実際のchunk/layer/形式/kernel分岐、expertごとの行数、重みの常駐・コピー元、実転送バイト、GU/Down呼出しを集計し、対応するcapacityと比較する。decodeは別に、CPU miss数、同じexpertを処理する行数、GPU常駐処理、投機検証の採用行数を照合する。70 tok/sのdecode予算は１出力token当たり14.286 msであり、prefillのまとめ方からdecode性能を推定しない。

実行経路の集計はdefault offの診断で、clean timingとは分ける。現段階で新しいモデルthroughputの測定や最適化候補の採用は行っていない。比較に使った元のreceiptは変更せず、このreportを追加した。
