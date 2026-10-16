# 実モデルの出力strideによるGEMM比較とGPU dispatch

2026-10-15 JST、Arc B570、現在のroot SYCL/oneMKL。
`gemm-orientation-validation-v2.json`の64 cases、CPU oracle192件が完了した。
実出力stride `ld=N`とpadding付きの双方で、row/column記述と現Gemm wrapperの
出力差は全件0。guardと入力不変も通った。モデル能力や全stateの一致ではない。

無計測系の別processでは8つのne×GU/Down×20 samples、計320件を採取した。
各sampleはABBA順の一方をsmall-ne16回/large-ne4回投入し、最後にqueue全体をwaitする。
各arm/cellは10 samples。実strideはGU1280、Down2560。
全個別値は`gemm-orientation-microbench-v2.json`、中央値/min/max/host submitは
`gemm-orientation-screen-summary-v2.json`に保存した。
前のpadding stride測定v1も元のPASSのまま保持し、今回と混ぜていない。

| ne | GU row / column、µs/call | Down row / column、µs/call |
|---:|---:|---:|
| 1 | 48.240 / 45.204 | 39.078 / 38.153 |
| 12 | 46.158 / 45.154 | 39.135 / 39.039 |
| 428 | 358.043 / 360.565 | 85.881 / 85.534 |
| 1490 | 436.570 / 434.346 | 150.880 / 150.805 |
| 1792 | 252.152 / 250.674 | 222.621 / 222.393 |
| 3245 | 486.489 / 449.230 | 379.265 / 372.227 |
| 3855 | 529.489 / 527.316 | 369.582 / 360.295 |
| 4096 | 546.687 / 547.645 | 378.112 / 370.282 |

この1 processのshape別差を全て「ばらつき」と断定しない。
ne1/12のrow記述のhost submit中央値はGU約39–41µs、Down約32–35µsで、
whole-call wallに近い。一方、ne4096はGU submit38.7µs/wall546.7µs、
Down24.6µs/378.1µs。submitにはhostとruntimeが含まれ、device serviceやcritical pathとは別。
hot FP16であり、7形式のcold quantized weightsや実モデルの重み付けではない。
以前の非排他的Down61.5s/GU6.0sをこの結果で直接分解することはできない。

## プロファイラーによる選択kernelの確認

既存のPTI buildを再利用した別のvalidation processで、GPU739操作を完全照合した。
内訳はGEMM192 kernels、初期化binary_test1 kernel、copy546件。
32 shape/stride cellsのX/W/Y初期copy、各beta/armのY reset→GEMM→Y read、
入力不変検査、初回だけの4-byte self-testまで全順序・byte数を検査した。
CPU10242 eventsのPID、B570 device lane、全739依存flowも一致した。
詳細は`unitrace-gemm-evidence-v1.json`。元のcollection receiptは変更していない。

64組全てでrow/column/current wrapperのdevice kernel名・実行geometryは同じだった。
従ってdescriptorだけの変更は優先度を下げ、製品GEMMは変更しない。
同じkernel名はhost費用の同一や全入力での同一dispatchを保証しない。
計測付きvalidationのdurationを無計測系の表やモデル速度に流用しない。

外部UTCとの対応は今回も不合格。内部の順序・durationだけを認定する。
PTI6c0d6b0bの`unitimer.h:29–38`はBOOT clockからepoch offsetを作る一方、
`:134–146`はRAW clockをtimestampに使い、`:69–77`はそのまま加算する。
clock domain差を検証するCPU確認は未実行で、任意の補正で過去traceを合格にしない。
VTune xpu-offloadのRyzen不対応と既存unitraceの経緯は
`../2026-10-14-primary-vtune/README.md`がcanonicalである。同じ非対応checkを繰り返さない。

build・validation・clean timing・profilerの全owned jobsは正常終了し、fresh fault0、dumpなし。
次は実モデルの自然なQSA経路の観測と、レイヤー単位の重み再利用を復旧する。
