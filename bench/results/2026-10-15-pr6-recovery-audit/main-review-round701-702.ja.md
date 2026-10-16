# R701–R702と長大MoEの次候補

2026-10-15。両Luna read-only FINALを受領。Mainは両report、現prefill MoE/HC、
SYCL `moe_combine`、installed oneMKL headerを読んだ。現在のQSA clean測定へsource変更は加えていない。

## Grouped GEMMの扱い

R701の「APIだけを替えて速度向上とみなさない」という判断は維持する。
Mainはinstalled `/opt/intel/oneapi/mkl/2026.1/include/oneapi/mkl/blas/usm_decls.hpp`
のgroup declaration（807行以降）とhalf/half/float/float展開（865行付近）を確認した。
R701のIntel参照は2023版なので、Mainは[2026版の公式仕様](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2026-0/gemm-batch.html)
も開き、同じprecision tableとgroup APIを確認した。これはcommand削減の実証ではない。

[vLLM XPU kernels](https://github.com/vllm-project/vllm-xpu-kernels)のMain確認時revisionは
`6b30c9b92cc525c32729df988b5f918e7c5661a5`。
treeの`csrc/xpu/grouped_gemm/xe_2`と当該hppのCUTLASS/Cute includeをGitHub APIで読んだ。
Xe2専用の実装参照があることは確認したが、Strataのpacked IQ2/3と同じowner/precision経路ではない。
[作者ブログ](https://ianbarber.blog/2025/02/11/grouped-gemms-and-moe/)もMainが読み、
variable-Mのgrouped kernelと通常のbatched APIの区別を確認した。移植・build・測定はしていない。
現scratchはexpert間で再利用するため、同時operand保持と実GPU launch削減を示すまでは新規採用しない。

## Superchunkと数値順序

R702の64K×top10×2560×4=6710886400 B（6.25 GiB）の`Dm`保持量は正しい。
ただしこれは**全slot出力を最後まで保存する現方式を拡張した案**の容量であり、
MoE一般にこのメモリ量が必要という下限ではない。
現`moe_combine`はtop-k slot順でFP32 fmaし、その後shared expertを加える。
expert-ID順のFP32 accumulatorへ変えると加算順が変わるが、これを理由だけに棄却しない。
ユーザーは実質的な品質差の評価を要求しており、byte一致が採用の唯一の基準ではない。
既存FP32/slot-orderを参照として残し、別flagの数値候補として誤差・teacher-forced logit/NLL・taskを測る。

Mainの追加候補は、expert出力の一つのFP32 accumulator、全行のFP16 `mixed_h`、
routes/weightsとinj/sgを保持し、shared expertの大きな出力planeは最後にoperator chunkごとに再計算して加える構成。
現shared projectionsは`m.mixed_h`、scalar gateは別のBF16 mixedから得るため、
scalar gateをprepassで保存すれば、FP32 mixed全行を保持せずに元のshared入力を使える。
Mainがprefill.cpp:2418–2424とpostops.cpp:248以降で確認した。
64Kで主要保持量はFP16 mixed320 MiB + FP32 accumulator640 MiB、
route等は別。per-expert jobを例えば4096行へ分割すれば、routing偏りでne=65536になっても
gather/GU/Hh/Down scratchをboundedにできる。これは未実装の案で、現shared regionへ必ず収まるとの結論ではない。

GDN/QSA/PLEの因果的なoperator chunkを昇順のまま保つこと、attention後のHC residualを保持すること、
最終FFN出力になる前にcallback/checkpointを公開しないことはR702どおり必要。
R_after_attentionを既存residual bankへin-place保存するなら、旧入力消費済み・中間・最終出力の
三phaseを明示し、host配置では追加往復も計量する。FP16中間化はlayer-boundary保存とは別の数値変更。

R702末尾の「262Kに8×64K」という表記は誤り。262144は**4×65536**であり、
通常batched262143なら65536三つと65535一つ、その後に最後のprompt tokenを通す。
物理境界のproducerはR700の別設計が必要で、今回のsource reviewを実行結果へ昇格しない。

現時点の優先は、QSA clean反復の完了、NativeDense phase実装のreader退役と実容量の確認。
GPUへ中間値を残す余地が増えた後、低メモリexpert-major案の全live bytesと追加転送を計算して実装候補を絞る。
巨大`Dm`のRAM spillを先に実装して転送を増やすことは避ける。
