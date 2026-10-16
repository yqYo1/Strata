# 同じ計算量で資源の使い方を変えた実測

2026-10-16、Arc B570 10GiB・Ryzen5600X・RAM128GiB。同じ独立診断binary
`6be5e0d3…452b017c` を使用した。現行kernel sourceは `5fa7927e…9bf0076f`。
新しい推論最適化を採用した結果ではない。

## 新しい比較

QSAのFP32入出力、K8V8、128 query、24 head、head_dim256、各queryの幅2051。
queryごとのK/V値、計算量、全allocation、初期化、キューを保った。
時刻はenqueueから全体drainまでのhost時計で、GPU-only時間ではない。

| 条件 | 独立3processそれぞれの中央値 ms | 3processの中央値 ms |
| --- | --- | ---: |
| A: 同じK/V行を再利用、128queryを一括投入 | 11.680966 / 11.692072 / 11.694208 | 11.692072 |
| B: query別にK/V行を複製、一括投入 | 11.909869 / 11.892607 / 11.915387 | 11.909869 |
| C: Aと同じ行を8queryずつ16回に分割投入 | 30.704889 / 30.700135 / 30.699188 | 30.700135 |

各processは別に2warmup後、10回を測定した。10回を独立process数へ数えていない。
順番はA0/B0/C0/C1/B1/A1/A2/C2/B2。個別90標本、各enqueue/waitのwallと発行thread CPU、
実環境、source/binary/library、PID/start/boot/終了は `own-qsa-resource-clean-v1.json` にある。
全processでDirect1、KMapping0、Nativefalse、Softmaxfalseをexec前に固定した。

A/Bで全device allocationは同じ294681024 bytes。実際に選択する異なる行のK/V/scalesは
Aが2165856、Bが277229568 bytesで128倍になる。両方に全複製を確保・転送し、timing内には
転送もCPU参照計算も入れていない。Bの時間増は**1.86%**。
これはこの入力でaddress working-set差への感度が小さいという結果で、物理VRAM trafficや
cache hit、帯域飽和率を測った値ではない。実モデル全体でVRAMが十分とも言っていない。

Cは同じ256 workgroupを、256group×1kernelから16group×16kernelへ分けた。
全体時間はAの**2.626倍**。enqueue wallの中央値は約0.0265→0.2565msで、増分0.2300ms。
全体の増分19.0081msに対しwaitの増分は約18.7813msだった。主な増分をCPUでの投入処理だけで
説明する結果ではない。各kernelに渡す並列仕事量、GPU側のgrid/command境界や実行効率が
重要な比較軸になる。ただし演算器、SLM/barrier/register/cacheのうち一つへの確定ではない。
発行thread CPUのほとんどがdrain待ちにあることも今回の全条件で再確認した。

現行の実モデルQSAは既に8192query等の大きなbatchで実行している。
このcomponentの結果を「さらにbatchを大きくすれば実モデルが速くなる」「モデルが2.6倍速くなる」
とは扱わない。短いquery入力の結果を長入力PP比較として提示するものでもない。
長入力を含む全prefillへの効果は公式bench.shと別の数値・state/品質gateで判定する。

## 正しさと実行の確認

先に小さい3ケースを実行し、共有/複製と3種類の投入粒度で全出力を比較した。
欠落page、負ID、重複ID、不正widthも含む。出力を各比較前にpoisonし、全small FP64参照、
全有限、全input不変、両端guardを確認した。UR traceでは39kernel提出を確認した。
次に128queryの初回をtrace付きで確認し、5kernel提出とsampled FP64参照を確認した。
その後のclean9processは全出力一致・有限・input/guard確認付き。大きいケースのFP64参照は
query0/1/2/末尾で、全queryのFP64参照ではない。timed repeatは最終frameを確認する。

初回とcleanの10個のframeを独立の `compare_frames.py` で全byte比較し、すべて一致した。
`own-qsa-resource-comparison-v1.json` に完全比較結果、各arm統計と解釈を保存した。
全11native processはexit0、33回の明示gateで実環境/mapsを照合し、所有終了、新規fault0、
dumpなし。通常のmodel/decode/RESTORE実行はない。physical262144や一般品質は未認定。

## 既存の資源測定との対応

| 資源 | 今の判断と根拠 |
| --- | --- |
| PCIe | 古い約39K入力診断のAttention区間でcopy占有9.11%、compute77.31%。PCIe単独支配の証拠ではない。新しい通常RAM往復proxyは4.35/3.89GB/sで、旧6.02/5.54との差の原因は未確定。 |
| CPU/RAM | 発行threadの高使用率の多くはGPU待ち。expert RAMcopyの3core化proxyは48層換算で約0.95秒の余地に留まる。物理RAM trafficの飽和は未測定。 |
| GPU内 | 今回の同値の資源比較では、選択working set増より、kernelごとの並列仕事量を減らす方が大きく遅くなった。GPU内の作業分割・同期・演算実行を優先し、VRAM帯域飽和やXMX飽和を断定しない。 |

古いAttention区間は94.945秒で、全prefill105.007秒とは別の分母である。
今回のcapacity/感度と古いprofileを無条件に混ぜず、同じモデルの判断時に条件を対応させる。
方法は `docs/RESOURCE_PROFILING.md`。実値・保持・未確定点は各receiptと削除manifestへ残す。
