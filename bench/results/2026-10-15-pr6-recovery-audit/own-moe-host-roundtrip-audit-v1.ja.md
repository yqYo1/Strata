# MoE行振り分けのホスト往復

2026-10-16 JST、Main source audit。対象は既存compact FP16 expert計算経路、FP32 residual、
C8192、48層、実78453 API入力/78452 batched、native/canonical full layer bank。
source base c8a682103b43b02e8cec458ffac29ec696e85023。
CPU観測v3のMain差分はown-prefill-host-profile-v3-source.diff、演算/待機順/配置の変更は無い。

1. GPU routerがtop-10 expert IDと重みを生成する。
2. group mappedの既定試行ではhost USMへGPU copy_i32 kernelがIDを書き、既存queue同期後にCPUが読む。
   compatのcudaHostGetDevicePointerはhost USM自身を返す。別のVRAM allocationではない。
   allocation失敗/STRATA_GROUP_COPY=1では通常host vectorと明示D2Hコピーになる。
3. CPUがIDを走査し、expert行数cnt、off、slot、srcとactive orderを作る。
4. mappedのslot/srcをGPU copy_i32が読み、VRAMのslot_dev/src_devへ置く。
   fallbackではH2Dコピー。GPUのgather/GEMM/combineが続く。

したがってホスト往復は実在する。CPUに戻すのはID/対応表で、ここでhidden stateやrouter重みを
CPU計算へ戻すわけではない。残差のhost保存/再入力は別の往復で、別途測る。

論理byte数は方向ごと48*78452*10*4=150627840B。
ID一回とslot/src各一回で451883520B、約0.452GB/430.8MiB。
480回のrouting boundaryがある。これはsource式と既存hostgroup480countの照合であり、実PCIe
traffic counterではない。mapped accessesのtransaction、cacheline、coherence overheadは未実測。
仮に全payloadを6GB/sで連続bulk転送するだけなら約0.0753秒相当だが、現行kernel accessや
480回のdependency/syncにこの速度/時間を適用できない。容量から待ち時間を断定しない。

v2の0.165362610 CPU秒/0.168038559 wall秒はwait後のgroupingとmap提出まで。
ID公開前のqueue wait、map提出後にGPUが取得し終わるまでを含まない。
この値だけで往復全体の重要性を低いとした優先判断は不十分だったので修正する。
逆にroute境界のcudaStreamSynchronizeは先行attention/GDN/共有expertなども全てretireするので、
そこに溜まったGPU計算の時間を追加転送latencyや削減可能時間として数えない。

GPU上だけでstable groupingしても、現行oneMKL compact callerはhostのne=cnt[e]とoff/orderで
2回のGEMMを提出する。countsだけを返す案はpayloadを減らせるが同期は残る。
全てのCPU待ちを除くにはdevice descriptorを消費するexpert計算まで設計する必要がある。
単純に全expertへT行paddingする方法は余分な演算/VRAMを増やすので無条件には選ばない。
順序/slot写像/重複/不正ID/empty expert、output combineと最大physical contextの正しさを保つ。

Mainはdefault OFF観測v3を実装し19host modes、runtime/compat、小GEMM16casesを検証した。
route_exchangeはID公開+既存queue waitをinclusiveで測り、重なるexplicitwaitも差分で分ける。
upload提出、host/device residual入出力の提出も分割。実78K診断は別プロセスで現在実行中。
CPU秒をcritical-path短縮見込みとは呼ばず、純粋なID転送時間が単独で測れたとも言わない。
Orca Luna R761–763は新しい並列調査で、Mainは途中steeringを送らず実測を管理している。

実78Kの最終結果はown-prefill-host-attribution-v3.ja.md。native正常/新規fault0。
route公開+queue drainは131.610017977wall秒、wait差引後0.010846707秒。
先行GPU仕事のretireを含むため、pure route転送latencyとはしない。
残差host転送提出のCPU25.793464402秒を別に確認。元postprocessorのversion判定FAILと独立再解析PASSを区別する。
