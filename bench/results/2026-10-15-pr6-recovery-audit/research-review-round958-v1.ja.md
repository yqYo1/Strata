# R958: vLLM Xe2 K-loop work-group split barrier review

既存のレビュー済み資料だけでは、現行Kループの`ScopeWorkgroup` split barrierがなぜ必要か、正しさの依存と性能上のスケジュールをどう分けるかは確定しない。R807とR810は操作順と全laneの一様なloopを記録する。R811はblock2D descriptorとprefetchの境界を扱う。R924は別のQSA kernelで明示的なSLM producer/consumerをまたぐbarrierと重ね合わせの条件を説明するが、このTLAのfragment/reorder/DPAS経路には適用を確定していない。

現行ソースのK=32 loopは、work-group一様に`barrier_arrive`、A/B block2D copy、次tileの条件付きprefetch、fragment reorder、`cute::gemm`/DPAS、`barrier_wait`を実行する。固定TLAの`xe_split_barrier.hpp`では一引数scopeをexecution/memory両scopeに渡し、既定のmemory semanticsは`SemanticsNone`。TLA docsはこのpairをsubgroupを「loosely synchronize」するものと説明する。これは意図の手掛かりだが、現行の全依存を説明せず、pairが不要または性能専用だとも証明しない。Strata kernelに明示的SLM bufferは見えないが、TLA helper内のstorageやcollective要件を否定する根拠にはならない。

したがって、一件の新しいdecision-support監査は**schedulable_now=true**。質問はこの固定loopとpinに限り、SYCL/Intel/SPIR-Vのexecution/memory意味、TLAのcopy/reorder/DPAS依存、barrierの収束要件とschedule意図、未解決のcompiler/runtime loweringを分ける。予算は1,200–1,500語、24 KiB、一次資料最大7件。実行やbarrier削除を勧めない。dispatchはこのreviewでは行っていない。

M=160のGU wide 0.174759 ms、Down wide 0.122230 ms、dense M=8192 GU 1.203226 msはsubmitからcompletionまでの時間で、device serviceやbarrier時間を分離していない。barrier bottleneckの証拠にはならない。このsource/API監査にGPU計測やMoE ne censusは不要。

R811取得範囲: lines 17–21, 33–43（block2D/prefetch契約のみ）。R924取得範囲: lines 15–21, 25–30, 35–39（別kernelでのbarrier、lifetime、schedule一般論）。詳細なhashと限定条件はJSONに記録。research-index.jsonは変更していない。
