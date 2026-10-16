# 長大prefillの計算資源別測定

2026-10-16、Ryzen5600X・RAM128GiB・Arc B57010GiB。PCIe外部経路は
Gen4×4。現行source63326f98、engine6334491b、BF16/vLLM、K8V8、chunk8192、
GPU残差全保持、softmax候補OFF。公式bench.shのprefill-only・thinking ONを使用し、
実入力39262、batched39261、5chunks、出力24。これは計装した診断である。

`own-vllm-resource-timeline-model-v4.json` はPASS。native・wrapper・server・benchが
正常終了0、所有PID/start消滅、同一boot、新規fault0、dumpなし。534498件の全GPU行を
154種類の公式summaryへ正規化し、件数・合計・最小・最大・総時間が整数nsで完全一致。
`hardware-resource-profile-v1.json` と解析sourceに再計算と全個別sample表のhashを残した。

## 観測した長入力区間

60回の長入力QSA Attentionをsourceのquery数/launch形状と照合した。
最初のAttention開始から最後の終了までの94.945秒は途中のMoE/GDN/転送を含む。
前後の仕事を除くため**全prefillの境界ではない**。

| 観測区間内の資源の占有 | 時間 | 区間比 |
| --- | ---: | ---: |
| GPU compute commandの和集合 | 73.403秒 | 77.31% |
| copy commandの和集合 | 8.649秒 | 9.11% |
| compute/copyの重なり | 0.000007993秒 | ほぼ0 |
| 観測commandが覆わない時間 | 12.893秒 | 13.58% |

この区間に完全に含まれるM2D copyは49.697GB、service8.214秒、要求payload/serviceは
6.050GB/s。D2Dは72.473GB、0.435秒、166.768GB/s。MはLevel Zero allocation queryの
UNKNOWNであり物理配置カウンターではない。host expert転送とのsource対応も併用し、
これらをPCIe/VRAMの物理trafficとは呼ばない。全期間のH2Dは6.354、M2Dは5.989GB/s。
同stackの320MiB通常H2D component6.015GB/sと整合する転送率だがサイズ/経路は別条件。
D2Dの論理read+write換算もcopy操作の値で、AttentionのDRAM帯域を測った値ではない。

従って転送は帯域を使っているが、PCIe転送だけが支配的という説明は支持されない。
転送と計算の重なりは現状ほとんどなく改善候補だが、それだけで600/1000tok/sに届く
根拠にもならない。QSA kernel自体の60回serviceは33.963秒。GPU computeが長い理由が
XMX/ALU、VRAM/cache、SLM/barrier、依存のどれなのかは、時間測定だけでは判定しない。

## CPU、RAMと未計測の資源

prefill105.007秒に対し発行threadのCPU98.420秒、そのうち明示GPU待ち69.696秒、
待ち以外28.723秒。host grouping0.106CPU秒に対し、GEMM投入123192回で5.643CPU秒、
upload投入5.753CPU秒、source copy4.846CPU秒、未分類12.329CPU秒が残る。
1core占有の多くはpollingだが、発行/packingによる空白の因果はまだ検証が必要。
12.893秒のcommand空白を全てCPU律速やGPU idleへ帰属しない。
RAMの物理traffic、各kernelのVRAM/cache帯域、XMX/occupancy/stallはこのrunでは未取得。

## 実際のメトリック列挙と次の判定

host-only helperをreview/build後、同じB570のvendor/device/PCIを確認して非rootで実行した。
`own-metric-enumeration-query-v1.json` はENUMERATION_PASS、native0・fault0・所有終了。
24groupsと各group先頭8件、合計192件のmetric propertyを取得し、compact evidenceに残した。
ComputeBasic、MemoryProfile、DeviceCacheProfile、VectorEngineStallsにはevent/query版と
time版が実在する。`observation_paranoid=1`だけから「利用不可」とした過去判断は撤回する。
ただし列挙はactivation/collection成功ではない。次はComputeBasic EVENT_BASEDの小さな
既存FP64-oracle fixtureで実収集を資格付けし、その後にGPU内部の律速を調べる。
権限・sysctl・service・GPU resetの変更はしていない。

## 失敗、性能と保持

v1 tinyはverbose property名の想定違いでFAIL。v2 modelは128MiB budgetで停止しFAIL。
v3 modelは全要求完了後のreportingで終了期限を超えてFAIL。元statusを変更していない。
公式PTIの名前集約は二重ループで、詳細名8467種を毎比較時に生成するsource上のコストがある。
DeviceTimelineはsummary verboseと独立に詳細名を記録するので、verboseを外し、公式規則の
suffix正規化後も全件照合するv3 sourceでv4を正常完走させた。旧timeoutのstackがないため
停止箇所の因果を確定したとは言わない。新規GPU faultはいずれも0。

今回の計装速度をclean比較に使わない。過去source6354のclean BF16参照は
402.6/403.0/403.4、median403.0。現source63326の採用対象OFF clean403.4は1件で、
cohortは別のdoc pin変更による失敗で未完了。softmaxや新たな性能向上を採用していない。
一般品質・全physical262144・RESTORE・旧PR6安全化の実行資格は引き続き別gate。

53.624MBのcanonical個別sampleはGit外でMainの次の内部counter/投入gap比較のため保持し、
2026-10-17T03:00Zまでに再評価する。成功raw timeline/APIと重複・解決済みbudget失敗logは
抽出/hash照合後、測定lock下で削除manifestを残して退役する。既存の失敗receiptは不変。
方法は `docs/RESOURCE_PROFILING.md`。
