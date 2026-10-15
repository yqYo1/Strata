# R757–760 Main review and Orca transition

2026-10-16 JST。4報告をMainが全文読んだ。原文は変更せず、下記で補足する。
R757–760はCodex内サブエージェントの完了済み成果であり、Orcaによる成果とは呼ばない。
ユーザーの追加指定後、次の反復からOrcaへ切り替える。

R757: copy offload無効化はqueue/engine選択で、host pinning/stagingが無い証拠にはならない。
通常host対host USMはセットアップ、copy呼び出しCPU/wall、wait、全体を別々に測る候補。

R758: event区分の約0.00005秒はeventの明示waitだけであり、event生成/破棄/driver bookkeeping時間
ではない。これをevent最適化余地の上限とするのは誤り。冒頭のv1 total/waitとv2 host値も別runであり
一つのpartitionにはしない。現行long pathはno-token-graphで、command-buffer専用設定は、実際に
command bufferを使う経路が確認できるまで比較候補にしない。legacy L0設定のV2適用性も未確認。

R759: 原文冒頭No Git operationと最終メッセージが不一致。read-only git statusを一回実行したとの
完了時申告を保存する。R758も禁止したread-only git statusを一回実行したと申告。状態変更は申告無し。
GPU scan/persistent schedulingはGPU group間forward progressとmemory orderingの根拠が必要。
NVIDIA論文の数字をXe2速度へ流用しない。広い調査の目的は知識獲得であり即採用を要求しない。

R760: Mainも現行group mapped allocation、copy_i32、同期、CPU map、expert GEMM callerを追跡した。
CPU grouping約0.165秒はID公開前のqueue waitと、後続kernelによるmap取得完了を含まない。
したがってこれだけで往復全体の改善余地を小さいと結論しない。48*78452*10*4*3 =451883520Bは
論理payloadで、PCIe counter実測ではない。host-USM aliasもGPU内VRAMではない。
queue drainはrouter前/共有expertなど全先行workも待つので、そこで待った全時間を
「ID往復が追加した時間」「GPU groupingで全て削れる時間」とも呼ばない。

次のMain計測はdefault OFFのCPU観測v3。route公開+既存waitを独立regionにし、重複waitを差し引く。
expert uploadとhost/device residual入出力の提出区間も分割する。演算/同期順/配置は変えない。
新しい区間の観測結果、compile、実GPU結果は、この記載時点では未取得。

Orca runtime ready 1.4.219、CLI skills get orchestration/orca-cliとcoordinator-loopを読んだ。
run_c8f478972ec8を新規作成。既存feature/syclでレポートのみ書くworkerを起動し、新規branchは作らない。
Mainは実装/測定を継続し、workerへ途中steeringを送らない。完了fileを読み、worker_doneの
Task/Dispatchを照合し、settlement後にworker-releaseしてからmailをackする。
