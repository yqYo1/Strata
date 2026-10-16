# 実JIT資源と次のattention調査

2026-10-16 JST。XeStrata sourceを使わず、このforkの既存qualified componentと公式Intel
compiler/runtimeだけを使用した。実carrier SHAは`48eeb3a760975096172390cfe9643a84eefe0331509486429af4d4a79dad5559`。
実loaded IGC/IGDFCLは`/usr/local/lib`の2.41.5。同じ7 libraryのpath/hashを各captureで確認した。
実行はMainがexclusive lockで直列監督し、全capture正常終了・出力/guard byte一致・owned退去・
同boot・新規fault0・dumpなし。nq256/repeat1の診断で、clean速度測定ではない。

v1のbare symbol filterではbinary RAとSIPだけを取得した。v2ではexact symbolを含むfile名の
filterを試し、ZEInfoをstdoutへ取得できたがASMはなかった。v3ではv1 RA file名に実在した
module hashをfilterに使い、同一moduleのASM/ISA等67 files / 56,640,870Bを取得した。
ASMのfile名はsymbolを含まずentry番号を使っていた。symbolとの対応は各ASM先頭で照合した。
元recipe/receiptは不変。RA binaryを推測decodeして資源の根拠にしていない。

| K mapping | 実ASM entry | numGRF | SIMD | instCount | flag store/load |
| --- | --- | ---: | ---: | ---: | --- |
| 0 従来 | 0008 | 128 | 16 | 6264 | 30 / 38 |
| 1 cell-major | 0005 | 128 | 16 | 6286 | 30 / 38 |
| 2 pitch40 | 0001 | 128 | 16 | 6285 | 30 / 38 |

これらは実compilerが出力したstatic metadata。ASMのreported git-hashは
`f05f218e6f0b888823c5678e1f64e91e597ea2dc`だが、別途source build全体の出自を検証した意味ではない。
flag spillの表示をメモリへのspill byteや実trafficと解釈しない。GRF spill declaration一致は0件だが、
これだけで全scratch trafficなしとは断定しない。動的occupancyや速度差の原因は未確定。

同一kernel handleの実UR launchはlocal128/global65536、13個のlocal引数の要求合計44,496B。
ZEInfoのslm_size4096と別の表示であり、4096だけを総SLMとしない。
deviceのUR local memory報告は131,072B。runtime側の配置・alignment・丸め・resident groupは
直接検証していない。詳細と実launch行は`prompt-jit-resource-analysis-v1.json`に保持する。

## R745/R746のMain補正

R745の「Kの4 scale/cellをFP32としてglobalから読む」は誤り。pool scaleはFP16で、
32 cellのKでは8,192BのINT8 codeと256Bのscaleがlogical payload。SLMにはscaleをFP32で置く。
double buffer案は追加16,384Bを必要とする。現在PVは全8 subgroupが参加するため、
同じwork-itemへ次Kのgatherを追加するだけでoverlapしたことにはならない。
unused subgroupを使うQK中の別stage等、実際のproducer/consumer割当と依存を先に検討する。
未実装・未採用。joint_matrixのgroup収束・address space・barrier/lifetimeの契約を維持する。

R746の10.694TFLOP/169.916GBのall-cap上限は1層分であり、69.692sの観測は12 QSA層の合計。
同じscopeで比べるには12倍する必要がある。元報告は書き換えない。
また幅は未知のhistogramを新たに測らなくても、現在のfresh input pos0=0ではsourceから
`n_kv=p0+t+1`、`width=min(n_kv,2051)`と決まる。prefill.cppのstep生成、qsa.hppとSYCL kv.cppを照合した。

actual batched78452についてsource定義の全12層×2KV headを数えると、
32-cell tileの合計120,787,680、joint_matrix_madの合計30,921,646,080、
source-issued DPAS算術126,655,062,343,680 FLOPとなる。選択IDのpageが全てvalidなら
code/scaleのlogical payloadは2,012,348,790,144B。invalid pageではreadをskipするため後者は上限。
cache/DRAM sector・重複読取りを測った実trafficではない。7 barrier/tileもsource回数であってstall時間ではない。
非排他的なqueue区間との除算からhardware利用率やcritical pathを確定しない。

次はbaselineのASM/ISAでload・barrierとsourceを対応させ、必要なら対応するcounter profileを取る。
否定されたmapping1/2を採用しない。opaque RA、SIP、各中間IR、重複outputと成功traceは
compact証拠確認後に退役し、現baselineのASM/ISAの2 fileだけを次候補比較向けに4MiB予算で保持した。
`closed-prompt-jit-and-nll-retention-v1.json`が全path/hash/reasonと元結果不変を記録する。
