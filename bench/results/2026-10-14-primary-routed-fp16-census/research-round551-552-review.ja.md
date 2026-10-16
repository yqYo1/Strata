# R551/R552とQSA投入回数の実装候補

2026-10-15 JST。Mainは原report、current root source、grid-stride proposalを読んだ。
現在のGPU suiteのsource/client/recipeは変えていない。候補は未適用・未compile・未実行・未採用。
CPU fixtureも測定との競合を避けるためMainの直列検証まで実行しない。

## Score grid-stride

原scoreのactive boundをA、起動group数をG=min(A,4096)とし、query毎にgroup rが
b=r,r+G,...<=device n_bidを処理する案は、全有効blockを一度ずつ担当する。
query/stepsとscoreのstrideは元のcapacityのままで、Gに変えない。nb<Aの検査も保持する。
其の値をnb<Gへ置換すると長文を誤って拒否するため、proposalでは区別している。
全laneが同じquery/nb/loop条件に従い、shuffleのmaskも同じで、
[SYCL group functionの収束条件](https://github.khronos.org/SYCL_Reference/iface/group-functions.html)
を保つ設計になっている。sourceだけではdevice bitwise parityや速度を認定しない。

Solのpatchはexact-1 flagを最初のvalid callで固定し、OFFなら元のkernel bodyを通る。
ONのFP statements、tail dead/biasとscore storeは元bodyから移しており、helperの実traversalを
CPU fixtureからも呼ぶ。独立oracleは元のgroup enumerationからcoverage/ownershipを比較する。
Mainはbase source hashと新helper未存在を確認した。fixture/structural checkerは未実行。
対象PCは64-bitであり、R551の32-bit size_tの一般論を此の環境の確認済み不具合とは扱わない。

flagは共通score APIを変えるため、captured decodeだけでなくexplicit active boundのprefillも
比較する。score、top-k IDs、attention/verifier結果、capture replayとfull contextの各gateが要る。
group数の減少はsource-derived countで、現在のdecodeの実ボトルネックという証拠ではない。

## Indexer storageの品質根拠

[vLLM PR #54890](https://github.com/vllm-project/vllm/pull/54890)の本文とcommit説明をMainも確認した。
Qwen3.8のFP8 indexerのtask結果は有望な候補根拠だが、本文のGSM8K/LongBench評価は
**thinking=false**、比較referenceはBF16である。現在のStrataのFP32 indexer／thinkingONと同じ
条件ではない。短いneedleとmatching tracesも256K全体の無劣化を証明しない。
別GPUのkernel gainを此のPCへ移さない。FP16/BF16 key storageを検討する理由として使い、
thinkingONのpaired task品質、long-context、score/selection/logprob差で判断する。

configured max context262144でpooled keyを16-bitにすると、source容量は192 MiB+6 KiB減る。
decoder queryはlayer毎のpersistent cacheではなく、最大T4のshared scratchなのでsavingは
4 KiBに留まる。key-only、query-only、combinedを分ける。
selected-ID変更は診断とし、それだけで品質不合格／合格としない。
既存`STRATA_IDX_FP16_CHECK`は同期と大きいtemporary allocationを伴うprefill-only debug hookで、
BF16、query narrowing、decode、task品質を検証しない。今回其を実行していない。

## Mainが確認したprefillのbatch API

`src/prefill/prefill.cpp:2071-2091`ではdefaultで`native_qsa_indexer_append_batch()`を呼ぶ。
呼び出し側のcommentはthree launchesと説明するが、実際にbuildされる
`src/kernels/sycl/native_qsa_indexer.cpp:183-203`は、for(i<n)で`append()`を繰り返す。
`append()`は其の都度queue.submitする。従って4096-row chunkなら此のAPI内で4096 submissions、
12 QSA layersなら49152 submissionsというsource-derived countになる。
全batched rows34560なら414720 submissions。実timelineで採取した数／時間ではない。
此のsource resolverの差はheader/commentだけでbatch最適化済みと判断できない例でもある。

別Sol implementation agentへdefault-OFFのbounded batch proposalを依頼した。
初期deadを確立し、完成したfour-cell blocksを並行にpool/norm/rotateした後、最終tailと
spare pooled row／block_posを更新する。chunkの最初がunalignedならprefixは旧tailを読むため、
tailを先に更新しない。初期token、partial/no-complete chunks、chunk境界、final full contextを
含め、現F16入力丸め・FP32 mathとserial最終stateを保存する。報告は別dirで受け取る。
これは未完成の実装依頼で、適用や速度向上ではない。

## 実行環境の観測と次の順序

同じowned suiteに対して/proc、sysfs、HTTP statusのみを読んだ。native start identityは一致。
thread affinityはCPU0に6、各CPU1..5に1、0..11に18だった。thread名は全てstrataなので、
TIDだけからStagerとruntime helperを識別しない。CPU時間はcumulativeでservice時間ではない。
GPUの瞬間的なclockや温度、memory PSIだけではthermal/CPU/PCIeの原因を決定できない。
GPU API、profiler、追加testやhardware/service設定変更は行っていない。

後のthinking decode時に、CPU0の三つの高CPU-time threadはfutex waitにあり、追加CPU ticksは0。
同じ期間にnative mainとCPU1..5の五threadsのCPU timeが増えた。最初のresource snapshotから
のdeltaはphaseを跨ぐ可能性があるため、其をdecode-exclusive serviceやcore利用率と呼ばない。
また、此のsourceのExpertPoolはdefaultでhostもdrainへ参加する。CPU0のmainが動くことを
「一coreがspinだけで無駄」と判断しない。未確定のStager affinity問題でdecode退行を説明しない。

現在のSYCL Verifierは`STRATA_VERIFY_PROFILE`が有っても`prof_on_=false`へ固定する
（`src/core/verify.cpp:383-387`）。既存CUDA timestamp reportを有効にするだけでは此のbackendの
kernel breakdownは得られない。既存MTP observerはhost/window countersを補うがdevice kernel
時間ではないため、admitted software traceまたは別のdevice attributionも必要となる。

user指定の順序を保つ。長い生成benchmarkの検証を閉じ、PR #6／公開Codexログの高速prefillを
監査・回復する。其の実sourceと既存のbatch処理を比較し、同等の部品を重複実装しない。
必要な候補のみをMainが直列でCPU/build/GPU検証し、prefill/decodeを別々に評価する。

Main review source pins:

- `src/kernels/sycl/qsa_select.cpp`: `45084756ca5e8e36085dae46581049c81366a45cdfde23e8c6f38defbba0c809`
- `src/kernels/sycl/native_qsa_indexer.cpp`: `d87abcf8aece04bc1566f4310e0dee546b7c80e0b711d75d118551b606d38685`
- `src/core/verify.cpp`: `640e0270be833ddff0218331a7db18043605af3ced2682787f6b06abdcf6316c`
- grid-stride patch: `fc44bc560960cbe02f79467713d9577b37a5d949a209b8cce4f8f1507ee0b8b4`
