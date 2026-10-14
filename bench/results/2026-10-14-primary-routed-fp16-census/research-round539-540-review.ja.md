# R539/R540のMain確認と次の比較

2026-10-14。原研究reportをbyte/hashで保持し、source・閉じたpilot・現build metadataを照合した。
live suiteと並行してfixture/build/GPU診断やsource適用は行っていない。

R539のZen3 kernel分岐をcurrent sourceで確認した。現在のnative build directoryのCMakeCacheは
STRATA_CPU_ARCH=znver3、compilerはClang。compile_commandsのnative_expert/iq_single_avx2/q2_avx2
は-O3/-march=znver3/-mtune=znver3で、後二者はAVX2/FMA/F16Cも指定されている。
既存GGML x86 quant objectのIQ3_S/IQ2_S/IQ3_XXS dotにもYMMのAVX2整数命令を確認した。
GGMLのAVX2 cache optionはOFFだが、GGML_NATIVE=ONで-march=nativeが使われるため、
optionの文字だけからscalar buildと判定しない。これは既存objectの読取で、opt-in exact NT1
branchの実行や全binary内のISAを証明しない。
pilot envはSTRATA_IQ_SINGLE_EXACTを要求していない。target用compiler flagが既にあることを
新しい改善として数えず、実job-ntごとのサービスを先に測る。

primary packは7つのGU/Down pairであり、R539の8pairという記述は4GU×2Downのsynthetic gridと
区別する。GU23/Down42は現在の48-layer metadataにはない。capacity fixtureでは追加controlに
できるが、current modelの測定populationに加えない。metadataはpilotのpinと同一hashを確認した。

[decode-pilot-source-budget](decode-pilot-source-budget.json)は既存counterの分母を整理した。
fixed spec4・probability floor0・suffix0のsourceではfirstT1、その後T4。
regular-codeのsource-derived verified rowsは3861、varied-proseは16325で、それぞれ観測済み
cache lookups = R×48×10と一致する。実emitted3346/8631に対して、verified rows/emitは
1.154/1.891。accepted率をそのままemitted割合やspeedupにしない。
source-derived window/commit数はdirect observerの値ではなく、次のMTP censusで照合する。

GPU cacheの実hitは38236/1853280=2.063%と67892/7836000=0.866%。logical CPU-group weightsの
source範囲をnt1..4とper-layer blob min/maxで計算すると、TG70でstrong約19.30..78.13 GB/s、
weak約32.15..129.22 GB/sと広い。これはlogical source extentsで、physical DRAM trafficや
RAMの下限時間ではない。実coalescing、cache/locality、format別serviceが未観測なため、
RAM/read peakと比較してCPU限界や70が不可能と結論しない。joined ledgerの必要性を示す計算。

R540はDmを半分にしてもregion=max(GDN,QSA,MoE)が現在4096ではQSA、8192ではGDNで決まり、
shared peakは下がらないと示した。m.yはcurrent prefillでGDNのwriterのみ、後のprojectionは
m.y_hを読むので、不要なFP32出力storeは候補になる。ただしcurrent kernelはyをvalidate/store
必須としており、nullptrをそのまま渡さない。API/span validationを含む変更とparityが必要。
persistent GDN stateをFP16へ下げる変更とは分ける。

count/carveにも独立した具体的差がある。xnはcarveでGR_UNFUSED時だけ確保されるが
bytes_neededで常に数え、grsは逆にcountがない。Mainは両方を合わせる
[未適用count案](prefill-admission-accounting-proposal-v1/report.ja.md)を作った。
fused8192のestimate減少は335413248 B（319.875 MiB）で、同じchunkの実確保は変わらない。
8Kが実際にfitする、speedが上がるという結果ではなく、post-PR#6 admission検証の候補。

R540の24-prompt/1pp non-inferiority例はそのまま品質認定に使えない。24問では1問が4.167pp、
6問categoryでは16.667pp。24問は大きな失敗を探すscreeningには使えるが、近い品質の証明には
測定単位と十分なsampleが必要。仮に独立なbinary harmful-pairを0件観測した場合でも、24件の
片側95%上限は約11.73%；1%未満にするには同じ仮定で299件以上必要。この計算は一般のLLM
品質を保証せず、固定test populationへの例。連続log-prob/perplexityはprompt単位のpaired差と
不確実性を評価し、相関tokenを独立sampleとして水増ししない。

benchmark→PR#6回復を優先し、live suite中はsource固定。後でobserver/source/fixtureをadaptし、
MainがCPU、healthy SYCL、32K+/64K、全256K境界を必要なscopeで直列管理する。
