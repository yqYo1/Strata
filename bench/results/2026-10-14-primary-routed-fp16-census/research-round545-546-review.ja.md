# R545/R546: 測定を先に行う候補

2026-10-15 JST。原研究reportとhashをregistryへ保存した。現在のthinking suiteは同じ
native/source/clientのまま稼働している。新しいGPU実験や他のtests/buildを並行実行していない。

R545の現在のunsplit verifierは48layerならwindow segment49 submissions、layer event wait48、
copy queue waitはlayer48とfinal1、verify queue final wait1、commit1、条件付きwork graphは
最大48 submissionsとなる。これはsource-derived call countで、各callに同じ固定時間を掛けた
latency見積りではない。CPU計算中にdeviceが進み、wait時点で既に完了している場合もある。

same in-order queueへのgraph replayで前のgraph eventをexplicit dependencyに重ねるのは
省略候補となる。[Intel graph仕様](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc)
と[Khronos queue仕様](https://github.khronos.org/SYCL_Reference/iface/queue.html)の順序保証は
確認したが、moving extension仕様が実runtime versionに一致することまでは証明していない。
省略できるのは依存記述であり、graph submission、completion event、layer waitは残る。
現在の実timerだけでsubmit overheadを排他的に分離した結果もない。

実装するならqueueの安定した同一性が必要。単なる生pointer addressの比較を選ばず、例えば
SYCL queueの参照意味論を保つvalueを保持し、同じunderlying queueかを比較する。
新queueへのreplay、destroy/recreate、submit失敗、graphの最後のevent/queue組の更新を含め、
元のmutexとcross-queue dependencyを維持する。最終的な正常completionとerror lifetimeの
チェックは省略しない。まだpatchを適用したりspeedを測ったりしていない。

R546ではNT2..4用のAVX2 kernelが既に一つのdecoded weight blockを複数activationに再利用する。
gate/upは別々のweight codesで、同じexpertという理由だけでdecoded valuesを共有できない。
codebook table自体と、各blockのindex/sign/scaleを展開したdataを区別する。
lossless cacheにはmodel/weight side/block/formatのidentityと、元の整数dot/scale/reduction orderが要る。

裸の1byte/weightだけでも全expertで120.796GB、FP16なら241.592GBというlogical sizeになる。
これはphysical DRAM trafficではない。128GB RAMにpacked arenaとruntimeを保持した上で
全modelのexpanded copyを常設する案は進めない。限定したnt1 cacheは仮説として残すが、
actual nt1 exposureと同一expertのcache lifetime内の再訪が未観測で、insert/write/read/evictionの
追加costとRAM帯域のtradeoffも未測定。

外部の[vLLM INT4/AMX PR](https://github.com/vllm-project/vllm/pull/51546)と
[pinned GGML x86 source](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/src/ggml-cpu/arch/x86/quants.c)
も確認した。前者のhardware/quant形式はZen3のGGUF IQと違い、後者はformat/reference mathの
根拠として使う。別hardwareの速度をこのPCへ移さない。

次の選択はjoined MTP観測でactual job nt/formatとCPU serviceを取ってから行う。
現在既にcompileされているoptional NT1 exact helperを有効にする比較も先に候補へ置く。
新しい巨大cacheを作る前に既存pathと実work distributionを比較する。いずれも未採用で、
今の測定結果を変える変更はしていない。benchmarkの検証後にはuser指定のPR#6高速prefillの
source/設定/安全性の監査と回復を優先する。
