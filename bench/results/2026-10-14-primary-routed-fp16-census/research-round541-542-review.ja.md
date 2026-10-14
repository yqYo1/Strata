# R541/R542: Mainの寿命管理確認

2026-10-15 JST。原reportはhash付きでresearch-index.jsonへ記録する。
R541のdraft uploadについて実際のSYCL呼び出し先も確認し、測定を安全のため中断した。
これは新しいGPU障害の再現ではない。閉じたsuiteのnative/serverはexit0、clientはMainが
SIGINTを送ったため-2、runnerは1。新しいkernel faultなし、dumpなし、boot不変、全owner回収。
原resultと未完了statusは[原receipt](../2026-10-14-long-generation-benchmark/receipts/main-primary-long-generation-suite-root-v4-evidence-v2/index.json)に保持した。

`draft_kv`の旧`tk`は同じvectorを書き換え、旧`bh`はループ内のstack arrayだった。
SYCL adapterはqueue.memcpyのeventを待たずに戻る。in-order queueはdevice commandの順序を
保証するが、CPUの次の書き換えやstack objectの寿命を延ばさない。最終stream waitだけでは
複数batch間の再利用を防げない。boundsのscopeは最後のbatchでも最終waitより先に終わる。

R541だけで判断せず、`src/prefill/sycl/gemm.cpp`、`moe_mmq.cpp`、
`src/kernels/sycl/native_mmvq.cpp`、`kv.cpp`、`include/strata/sycl/launch.hpp`を確認した。
非null streamのGEMM/quantize/MMQ/KV appendは正常時にhost waitを入れない。
`finish`はnull streamの場合だけ待つ。draftは`m.cs`を渡し、Gemmも同じstreamで初期化される。
したがって、この経路の中にhost sourceの再利用を保護する暗黙の同期があるという反証は得られなかった。

一次資料: [SYCL queue](https://github.khronos.org/SYCL_Reference/iface/queue.html)と
[USMのhostとの同期](https://github.khronos.org/SYCL_Reference/iface/usm_basic_concept.html)。
queueのin-order性をhostとの同期に読み替えない。CUDAのpageable-source stagingも、この
SYCL adapterの保証として使わない。

現geometry N=2560/hc=4/LR=320/KV=512、Q8経路ならper_row=263396 B。
観測されたregion=607128320 Bに対してBの式は2304 rows。32K promptの最初の4096 chunkで
remainingがそれを超える場合は2batchとなる。これはsource-derived sizingであり、全条件を
通過した実draft path/Bを計測した値ではない。MTPのfirst_neededやring limitも残す。

Mainの修正は全batch分のtoken/boundsをimmutableなhost storageに保持し、正常時は元の
最終wait後にleaseをretireする。SYCLでearly return/exceptionならlease guardが先にdrainし、
完了が保証できない場合はstorageが生きている間にnativeをabortする。これはfault recoveryでは
なく、不明なleaseを解放してserverが次のrequestへ進むことを防ぐ処理。正常経路の追加waitや
device arithmetic変更はない。compile/host fixture/GPU validationは別receiptで記録する。

R542の指摘は別のerror path。`Verifier::fetch_dma`のhost taskがinline `flag_sets_`を参照し、
sticky Runtime errorでadapterのpreflightがdestructorのwaitをskipすると、そのcallbackの寿命を
証明できない。正常なcopy queue waitは保護するが、watchdog registryのleaseはqueued host task
を保護しない。これには先行errorが必要で、最初のGPU障害の原因や過去のhangの原因と断定しない。
今回のdraft lease guardがVerifier全体のerror teardownも解決したとは主張しない。

modeの読み違いも避ける。`STRATA_SYCL_VERIFY_NATIVE_CAPTURE=0`は通常graph recordingを
選び、graph自体を無効にしない。`STRATA_SYCL_PREFILL_SAME_QUEUE=1`はprefillだけの設定で、
Verifierのcopy queueが同じqueueになるとは限らない。

元の正常pilot PASSは終了状態の歴史として残すが、それだけで実装の寿命管理や品質を認定しない。
userのthinkingON/256K context/16K output capは維持する。修正の検証を済ませた後にbenchmarkを
完了し、指定されたPR#6・Codex公開session log・旧高速実装の安全な回復調査へ進む。
