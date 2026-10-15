# Main review: checkpoint traffic and prefill dequant candidates

2026-10-15。Luna R609/R610のFINALを受領し、Mainがsource/既存receiptを照合した。
新しいmodel測定、build/test/GPU実行はしていない。既存結果のoffline算術だけを行った。

## R609: 1passのままcheckpointを集める

既存transfer receiptの48層blob_bytesと512 expertsから、whole layer-majorの論理expert
payloadは**50292326400 B**と再計算した。chunk8192、root/turnなし、n-1を処理する
32K/64K/256K入力を16Kごとに分けると、2/4/16回の長いLM passになる。
一回に対する追加payloadは約50.29/150.88/754.38 GB。この算術と、checkpointの
GDN112.21875 MiBが圧倒的に小さいという設計上の差は採用する。

ただしR609のbulk 2.2487/2.7293 GB/sを用いた除算は**時間のscenarioであり、厳密な
lower boundではない**。同receiptのordinary-host expert copyには5.7902 GB/sの観測値もある。
bulk中央値より速い同系統sampleがある以上、それを最大帯域と置いて「64Kの追加転送だけで
1000t/s目標を超過し、目標不可能」とは結論できない。hardware peak、実model service、
重なり、actual per-layer upload時間の証明もない。より速い観測値を使うscenarioでも64Kの
追加payloadは約26秒相当だが、これも予測や下限ではない。実装比較の優先度を決める材料とする。

R609のMTP restore秒数も一つの過去load条件の値であり、全extra passへ固定加算しない。
PLE owning layerは現行`prefill.cpp`の**l==1**であり、任意の最終層からそのprefixのPLEを
保存してはいけない。`run_impl`はchunk callback前にm.csを同期するが、新state copyの
queue順序・完了・owner退役は独立に保証する。既存callbackの意味を変えない新internal hookと、
全層piece/MTP/loan refill後のpublishを持つbounded one-pass提案を別Solへ依頼した。

## R610: scalar dequantが実際の次候補

Mainは`src/kernels/sycl/iq.cpp`を照合した。generic halfとGUの双方で、各work-itemが
`float values[256]`へscalar decodeしてからhalfへstoreする。GUのsource callerはnative
prefill expert fallbackにあり、Downのnative expert callのほか、generic half側は
`src/prefill/sycl/gemm.cpp`のquantized projectionでも使う。

従ってgeneric half profileの12.064秒をすべてDown専用としない。profile全体のGU22.401秒と
generic12.064秒からprefillの排他的wallや改善上限を直接は出さない。R610のcensus90297calls
からの約887.7 GBは別workloadの論理展開量で、今回profileと割った25.8 GB/sを
実帯域や最適化目標には使わない。codegenのprivate spillや占有率も未測定である。

候補は、現在のhalf operandを保つformat別lane並列unpackと、packed weightをXMX tileへ
直接decodeするGEMM。前者は演算順を変えず全half bitwiseを要求でき、後者は大きな展開を
省けるが、既存fused nativeはQ8 activationを用いる別の数値契約なので置換を自動採用しない。
type18/22/21/23のGUと20/42のDown、実call/row histogramでcoverageを分ける。

Mainの`analyze_dequant_shape_coverage_v1.py`で旧34560-batched-token / 9chunk censusを
再集計すると、ne<=64は65657/90297 calls = **72.7123%**、770086/16588800 rows =
**4.6422%**だった。半精度重みの論理出力量はこの部分だけで645434572800 B。
演算行数が少なくても毎callの全重み展開は生じるため、call-weightedとrow-weightedを
区別して候補を選ぶ。これはcurrent LMのcoverageではなく、per-format x neのjoint histogramも
既存receiptにない。global histogramとformat表からjoint分布を創作しない。

MainはR610のllama.cpp pinned公式docをwebで確認した。
<https://github.com/ggml-org/llama.cpp/blob/a0e6dc7384edb8f5d927155e1d26c10b4a310e6d/docs/backend/SYCL.md>
XMX直接decodeのtype/shape gateとne<=64のfallback条件は実在する。これはStrata/B570での
性能・品質の証拠ではない。docのoneAPI2026.1改善報告は現在のcompiler/libraryとの違いを
R611が調査中で、他モデルの数値を本forkへ転用しない。local ghqにはこのcommit objectが
未取得だったため、local show失敗を「upstream commitが存在しない」とも扱わない。

現在の実行優先度は同一binary Direct両arm完了、head-split/host時計の検証、checkpointの
one-pass修正とRESTORE、その後dequant候補。未検証の実装案を速度改善として報告しない。
