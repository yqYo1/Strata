# Research scope plan R875–R877

## Main向け運用要約

R875は、現在のQSA prompt-attention kernelについて、既存のunitraceと利用可能な場合のVTuneからメモリ、同期、matrix計算を見分ける方法を調べるdecision supportです。source revision、kernelのSHA-256、R861の診断receiptを固定しました。

旧profileはsource 6354のbinaryで、現行R861 buildとは別です。33.966秒のattention service合計は72回のwarmup/model launchに対する非排他的な値であり、critical pathやoccupancyを示しません。現行R861のclean cohort結果がそろうまでGPU計測は提案しません。

R876はGPU stencilのtemporal blocking、R877は粒子シミュレーションの幾何neighbor searchに関する知識収集です。どちらもshared indexのscope metadataで重複を確認し、本文はMain未読のままnative reviewに回します。

3件とも1300–1700 substantive words、primary URLは最大7件です。計画のみで、Orca workerはまだdispatchしていません。

## R875: decision support

対象はsource revision 63326f98c36656cf3a3e1adcc54c3c25a02aa6eeのSYCL prompt-attentionです。workerは固定した3つのSYCL source/headerだけを読み、R861 qualification、二つの長入力diagnostic、旧unitrace profile summary、K mapping比較、shared indexを照合します。CUDA/HIP実装・検証やXeStrata由来ファイルには触れません。

調査では32セル単位の処理を、selected ID/page table参照、K/V codeとscaleの読込、QK/PV joint-matrix演算、online softmax、local buffer、workgroup barrierへ分けます。各段階に対して、unitraceのduration/timeline、metric query/sampling、条件付きstall sampling、VTuneが既に導入済みの場合のGPU Compute/Media Hotspotsで何を区別できるかを整理します。profileの最小手順は、MainがR861 clean cohort終了後のGPU枠で実行できるように記述します。

旧profileはsource 6354 / binary 85643f39…で、R861 source 63326 / binary 6334491b…とは一致しません。旧profileの33.966秒 attention rowは72 launchのservice sumで、他のdevice intervalsと重なる可能性があります。4096 Bはkernel作成時のproperty snapshot、32336 Bは別capture contextのrequested local argumentsです。どちらもphysical SLM使用量やoccupancyとして加算・解釈しません。R748ではhardware counterへのunprivileged accessがblockedでした。counter取得ができない場合は、その証拠上限を示し、権限変更やツール導入を提案しません。

既存の関連scopeはR714/R725のK gather仮説、R736のtile geometry、R745のnext-K/double-buffer条件、R746の静的算術、R747のSLM byte会計、R748のcounter accessです。R875はこれらの再調査ではなく、現行sourceに紐づく計測の判断規則に限定します。unpadded K cell-majorはclean componentで11.28%遅く、KMapping 1/2の再試験は行いません。

source revisionと三つのsource hashはR861の両diagnostic receiptと一致します。diagnosticはclosed/pass/fault0で、現在のunitrace direct-model receiptもindex上のsummaryから特定できます。そのため、GPU実行なしでtool capabilityとsource mappingを調べる前提がそろっており、schedulable_now=trueです。

## R876: knowledge collection

**題目:** GPU stencilのtemporal blockingが有効な条件とhalo/resourceコストによる逆転。

streaming、spatial blocking、shared-memory/register temporal blockingを、stencil radius、tile寸法、time depthに沿って比較します。off-chip traffic削減、haloの再計算、local memory/register容量、同期、occupancyの間で、勝つ条件と負ける条件を示します。R855のconvolution mapping、R839のbank mapping、R870のtile rendering/BFSから範囲を分けます。primary sourcesはAN5DとRevisiting Temporal Blocking Stencil Optimizationsです。source制約は公開論文のみで、local code、GPU、build、profileを必要としないためschedulable_now=trueです。

## R877: knowledge collection

**題目:** GPUでの幾何neighbor searchと、cell list / stenciled cell / BVHの条件別比較。

粒子のinteraction radiusに対して、all-pairs、cell list、stenciled cell list、linearまたはquantized BVHを、build、candidate pair数、list出力、反復force stepで比較します。density、cutoff、粒子サイズの差、periodic boundary、Verlet list rebuild間隔とcorrectnessを含めます。R872のrelational join、R869のBFS、R867のquery fusionとは別の固定半径幾何検索です。元のGPU neighbor-search論文とneighbor-list artifact研究が公開されており、local artifactは不要なのでschedulable_now=trueです。

## 固定した運用条件

- R875–R877の出力先は各roundのresearch report 1ファイルです。workerは他ファイルを変更しません。
- Orca研究workerはgpt-6-lunaを使い、build/test/GPU/profiler/process inspection/Git command/product editを行いません。
- Knowledge collection本文はMainに渡さず、native Codex gpt-6-luna max review後にretrieval metadataと短いdigestだけを共有します。
- research-index.jsonはassignment中に変更しません。XeStrataのcode、test、helper、library、archive proposalは利用しません。
- この文書はplanning onlyであり、dispatchや性能採用の判断を行っていません。
