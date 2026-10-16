# Stager error guard未適用案のMainレビュー

2026-10-14。Sol元artifactのhashを保存したまま、current sourceと公式referenceを照合した。
source適用、fixture compilation/execution、GPU確認は未実施。live v4 suiteはsource固定。

current adapterのattemptはRuntime::checkをbodyより先に呼ぶ。したがって失敗した
cudaEventSynchronizeが実waitへ到達したとは限らない。Runtime::wait(event)は
wait_and_throwの後にもcheckするが、adapter手前のcheckを回避する根拠にはならない。
Khronosの[queue reference](https://github.khronos.org/SYCL_Reference/iface/queue.html#wait-and-throw)
ではwait_and_throwは完了を待ち、未消費async errorをhandlerへ渡す。独自checkにより呼び出しを
省略した場合の完了は保証できない。

Khronosの[USM free reference](https://github.khronos.org/SYCL_Reference/iface/usm_allocations.html#sycl-free)
はfreeにcommand完了待ちがなく、使用中/queuedなmemoryをfreeする動作は未定義とする。
current cudaFree adapterはwait_allを行うが、ここにもattemptのprecheckがある。失敗statusを
lease完了や正常reclamationとして扱わない案はこの契約に沿う。prefill releaseに既存の
compute/copy drainを確認することと、全engine allocation/graph lifetimeを証明することは別。

Sol案の31wrapped callsは、一度だけ既存APIを評価し、SYCLでは失敗をflush付きnative abortへ
つなぐ。新queue/event/wait/copyは追加しない。current cudaSuccessはcudaError_t enum、
cudaGetErrorStringも同じenum引数でありhelperのsignatureはsource上整合する。
これはcompile proofではない。成功path、同一queue alias、copy_events条件、host/device各leaseの
出版順を保持することを実source adaptation後に再確認する。

CPU fixtureのfake wait failureはcompletionせずerrorを返し、shared probesでready/issued/bytes/freeを
子終了後に検査する。Stager・inline issuer・release bodyをsourceから抽出し、通常成功ケースも
検査する設計なので、helperだけの検査より範囲が広い。ただしone worker/inlineのfake adapterで
あり、actual DMA、threaded issuer、driver、graph、全256K、モデル品質の保証にはならない。
affinity案との合成はinit signatureを変えるためfixtureを明示的にadaptする必要がある。

abortは異常終了で、GPU drain/復帰やatomic rollbackではない。他threadの既存独立workを
停止時刻まで取り消したとは言わない。Mainはユーザー指定のbenchmark→PR#6安全性点検の
境界後にこのsourceと必要なold routeを照合し、適用する場合はCPU error fixture→healthy SYCL
off/on比較を直列に行う。実GPUに故意の障害は注入しない。

Intelの[graph usage guide](https://github.com/intel/llvm/blob/sycl/sycl/doc/syclgraph/SYCLGraphUsageGuide.md)
はrecording中のeventを使ったtemporary-memory再利用が通常実行のevent管理と互換ではないと
説明している。これはPR#6再点検の契約確認項目であり、この案でgraph lifetimeを検証したという
意味ではない。公式pagesは2026-10-14に参照した。歴史runtimeとの完全なversion一致は未認定。

元unified.patchに必要なblank context-prefix空白だけはartifact whitespace検査から除外し、
元patch/hashを変更していない。通常のstaged source/report検査は通したが、source適用合格ではない。
