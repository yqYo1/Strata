# BF16-prefix の現行profile

2026-10-17 JST、Arc B570 10GiB / Ryzen 5600X / RAM128GiB。
source `d9f9a395`、binary `efd06473…00005332`。
公式bench.sh、実入力39262 / batched39261、context/resident65536、chunk8192、
residual GPU32768行＋残りRAM、thinking ON、出力24 token。
TRACE・parameter validation・CPU時計・coarse/QSA event marker付きの独立1process。
速度の採用値は別のclean9process比較であり、このprofileの434.3 tok/sではない。

wrapperは89.2009秒。queue区間は投入間隔や待ちを含み、相互に排他的ではない。
加算してGPU使用率やAmdahlの割合へ変換しない。

|区間|秒|
|---|---:|
|QSA全体|27.4816|
|うちattention（BF16 prefix packを含む）|21.5123|
|うちselection|3.88812|
|MoE|20.1915|
|GDN|9.24038|
|expert upload|8.40517|

発行threadのCPU時間81.2451秒のうち61.2584秒（75.40%）は明示的なGPU待ち。
待ちの外は19.9867 CPU秒。RAM source copy4.86134、upload投入4.92302、
123156 GEMM投入2.39120、host grouping0.08497、未分類5.99046 CPU秒。
route exchangeの40.0141 CPU秒には待ち40.0100秒が含まれ、待ちを除く処理は
0.00416 CPU秒だった。routing表示だけでCPU計算支配としない。
これらは発行threadの測定であり、他threadを含む総CPU serviceではない。

要求転送量はexpert upload50,292,326,400 bytes、residual入出力各方向で
host↔device6,249,902,080 bytes、device↔device31,541,166,080 bytes。
物理PCIe/DRAM trafficや使用率のcounterではなく、転送区間にもhost gapが含まれる。
今回の値からVRAM帯域、XMX、PCIeの飽和率を計算しない。

48層×5chunk、QSA12層×5chunk / 合計471132 queriesを照合し、Mode0 staged0、
全sample数と要求byte総和、event正常終了、所有終了、新規fault0を確認した。
通常の所有権返却は確認したが、保存sessionからのRESTOREやfull-model262144検証ではない。

次はQSAのtile/operand reuseを現行BF16配置で調べる。MoEも大きいが、bankやgrouped
GEMMを選ぶ前にdevice側のdequant・変換・GEMMと投入間隔を分ける必要がある。
CPU groupingだけのGPU化で大幅に速くなる根拠は今回の0.085秒にはない。
精度は同じモデルのSGLang/vLLM実行契約を主基準とし、llama.cppとの完全一致を求めない。
