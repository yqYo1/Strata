# SYCL実event時刻の初回校正

2026-10-17 Arc B570/driver1.17.39395+14/固定LLVM22.1。
Main作成の専用sourceをstrict O3/fsycl/Werrorでbuildし、既存Runtime compute queueで実行。
UR tracingとparameter validationをexec前から固定、Mainの3holdでruntime/ELF/環境を確認した。

短/長kernel各6、H2D/D2H各6、初期化copy1の25実eventを保持し、checked drain後に照会。
全status=complete、submit<=start<end、依存するin-order列のend<=next.start、
timer granularity条件、全6回の出力・guard・入力不変を確認。12kernel submission。
最短kernel service1250ns、照会1回の最大wall279607ns。
全eventは完了後に解放、全process正常終了・所有終了、fresh fault0、dumpなし。
実値25件とdevice timer resolution、binary/commands/source/環境はJSONに保持。

これは時刻取得APIとこの小さい列の校正で、GPU稼働率・PCIe帯域・モデル性能や
本番collectorの資格付けではない。ログ・validationを有効にした値を速度比較へ流用しない。

SYCL仕様では未完了eventのprofiling queryは待機しうるため、実装では先にchecked drainと
statusを確認する。timestampの同backend timebaseとaccuracyの限界も守り、CPU時計と
device値を差し引かない。参照: [Khronos event reference](https://github.khronos.org/SYCL_Reference/iface/event.html)。

次は既存MoEwindow2の正常retirementで、実command eventだけを固定容量で収集する。
元の数式・同期・tensor配置を変更せず、default offの診断として作る。
