# BF16 attention: local tiled operands (private experiment)

現在のscore33 production sourceを基準に、Q/Pは16列のdense tile、K/Vは16×16をBF16 pair-packed B形式でlocal memoryに配置する。元のglobal K/V producerのsource座標、BF16値、32cellずつのFP32 accumulation・max・normalizer更新順、barrier数、local payload28,224 bytesは保つ。変更はGPU内で完結する。CPU側の追加tensor計算・転送・待機は入れない。

API仕様のみを参照し、実装とindex式はMainが独立作成する。XeStrata sourceや他実装のsample/helperは参照・移植しない。LLVM binaryの既存パスはcompiler依存だけである。

- Intel matrix specification: commit a13d57a7f99adac86be159be55030e4cead7fd25, `sycl_ext_intel_matrix.asciidoc`: BF16を含む16bit VNNIは列ごとに隣接2行を格納し、packed load strideはpacked rowの要素数。B tileは8×32halves相当、stride32halves=64B。
- oneAPI matrix specification: commit cf12c3783cc6a7adaf76e54c6a4f11f81ec8599b, `sycl_ext_oneapi_matrix.asciidoc`: BMG base pointer4B alignment、byte stride8B倍数。A stride16halves=32B。
- 元のprovenance JSONと取得本文のSHA256一致を検証。仕様の本文はAPI資料でありsource実装ではない。
- 候補は全Q/KV/P配列のsizeof/offsetof/alignment/standard-layout/trivial-copyabilityをcompile-time確認し、Intel matrix aspectと既存matrix combination/WG/SLM guardを確認する。
- ビルド前に各全配列indexのbijectionと実際のA/B matrix tile loadからsemantic座標への対応を独立host loopで確認する。

判定順はtraced smoke→11 component cases（physical262144含む）→current score33 objectと同一binaryの独立process3回ずつ（各10samples）。数値oracle・guard・lifetime・faultのgateを保つ。componentのphysical262144はモデル256K消費/RESTOREの検証とは別である。componentが速い場合だけ生成ISAとモデルでの候補採用へ進む。静的命令数は実行回数・hardware utilization・bank mappingの証明にはしない。
