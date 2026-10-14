# R576–579 Main review

2026-10-15 JST。対象は実際のroot SYCL engine、Arc B570 10 GiB / Ryzen 5600X / 128 GiB RAM。
研究レポートを実装・実測の証明として扱わない。各原本は訂正せず保存する。

## R576: actual-prefill profiler

一つの実model layer/chunkを有限区間で収集する案は有用。従来のhot GEMM 739イベントは
実modelのDown phaseやGPU全体のcritical pathを示していない。kernel名filterだけでは
source roleを区別できず、copy/dequant/gather/GUを落とす。same-queue gapとGPU全体のidleも区別する。

原本の **8 GiB RSS / 300 s wall** はこの実modelに適用しない。既存の実model起動は
約48 GiB以上を必要とし、8 GiBでは正常な起動を打ち切る。Main所有の実model診断と同じ
**88 GiB RSS**、段階に適した有限wallを採用し、開始前にreceiptへ記録する。
64 MiB初回trace上限、512 MiB全prefill例外上限は保持するが、PTI per-thread event capは
完全性の証明にはしない。5% overheadは研究者の提案であり、実測値・ユーザー指定ではない。

ITTは現在の製品依存関係にない。ソース変更前に、既存providerの条件付き収集と
低頻度host RAW-spanだけで同じ操作のjoinができるか確認する。start-pausedではallocation
prefixが欠落する可能性もあり、copy endpointはsource/collectorの実証が必要。
`UNITRACE_SystemTime=1`のCPU-header検査だけでGPU clock gateを通さない。新しいGPU実収集の
RAW時刻・PID/device/queue/flow対応を照合し、古いUTC gateはfalseのまま保持する。
製品profile変更・GPU collectionはまだ行っていない。

## R577: residual transfer

現runtime/安全設定での256 MiB controlを採用する。H2D ordinary 2.24–3.06 GB/s、
Host-USM約2.71–2.75 GB/s、D2H ordinary約5.05–5.17 GB/s、Host-USM約5.66 GB/s。
過去の6.447 GB/sは現在のresidual routeの達成帯域やcritical-path下限へ流用しない。
実expert小コピーのbatch依存もあり、256 MiB controlも160 MiB residualの実rateではない。

有限Host-USM stagingはCPUコピーとその寿命管理を追加する。まずFP16保存と不要VRAMの
返却によってhost tail自体を減らす。必要なら後で同じ160 MiB/最終partial形状の
ordinary/Host-USM roundtripを比較する。全host tailのpinningや新しいqueue overlapは採用しない。
原本の「FP16は自然入力未評価」は調査時の状況。Mainのselected-prefill-long-v1では
34603入力/19位置の分布比較が終わったが、独立task品質や全token PPLは未評価のままである。

## R578: MTP expert lifetime

KV-only `record_forward(T,-1)`はexpertを読むfull forwardへ入らない。退役対象は
normal/coupled round/step graphの四familyであり、dense/KV用graphやVerifierの`final_R`
borrowを残す現実装と整合する。private RAM vectorからchecked H2D後にpublishすることも確認。

false-return時の部分capture状態は一般的に再利用可能とは証明されていない。ただし現serve
`generate.cpp`の`!drafted`分岐は`ERR`後に **return 1** でプロセスを終了し、同じMTP ownerで
次requestへ戻らない。現callerの契約として記録し、汎用のrecoverable retryを主張しない。
capture例外/不明なdrain/destroyはownersを保持したfail-stop。fault injectionは実行していない。
初回standalone gateは新しいallocation/graphを作る正常経路だけを確認する。
既存graphを退役する繰り返しserve、coupled、短文bypassは別のMain GPU gateが必要。

## R579: decode CPU roofline — denominator correction

**原本の `q=cpu_rows/R` は誤り。** `R`はVerifierのtoken rows、`cpu_rows`は48層×top-k10の
expert-token rowsを合計した値である。実source
`decode_route_ledger.hpp::record/complete`は、各layerで`n=T*k`、`cpu_rows=sum(job.nt)`、
owner CPU entriesと同値、全owner entriesの和を`route_entries`と照合する。

従って対象の固定geometryでは以下を使う。

```
rho = R / E
q = cpu_rows / route_entries = cpu_rows / (R * 48 * 10)
CPU useful FLOPs / output = 9,830,400 * cpu_rows / E
CPU packed logical bytes / output = owner[CPU].blob / E
```

`q`が割合[0,1]となる分母は全route entriesである。R488のcache hit/miss entry率も
unique jobs率とは異なるが、owner entry率同士ならこの`q`と比較できる。
R579本文のTG70時全expert619.3 GFLOP/sは`rho=1.875`という過去短出力条件付き算術。
CPU分は訂正した`q`を掛ける。0.818 TFLOP/s FMA controlや36.87 GB/s RAM controlは
packed IQ expertの実能力でも同一workloadのceilingでもなく、70 t/s可否を判定できない。

一回の実際の32K以上serveでW/R/D/A/C/E、CPU unique jobs/nt、logical bytes、issuer/worker
thread CPU時間、既存caller spanを得ることが次の優先事項。worker CPU nsの合計をdecode wall
から引いて待機時間にしない。service wallも並列部分のunionではない。
今のobserverはstandaloneでは起動しないため、stdin/stdoutの所有付きserveで取得する。
診断によるthroughputは正式benchの性能採否に使わない。
