# 通常RAMとhost USMのcopy提出・完了比較

2026-10-16 JST。前の実78K診断でhost residual copy提出内に25.793CPU秒を見つけたため、
呼び出しが早く戻ることとコピー全体が速いことを分離した。Main新規MIT sourceのみを使用した。
既存native数学/モデル配置/環境は変更せず、EXCLUDE_FROM_ALLのsycl_host_transfer_probeだけを追加。

## 方法と正常性

production互換cudaMemcpyAsync ->同じRuntime compute queue ->cudaStreamSynchronizeを使う。
2MiBと320MiB（8192*10240*4の実residual fullchunk）をH2D/D2H。2MiBはsize-scale comparisonで
実expert blobの正確なsizeとしては扱わない。通常new[]対sycl::malloc_hostのpersistent host buffers。
双方とも保護領域先頭を4096に揃え、bodyは+128B。同じdevice allocation/verification hostUSMを使う。
C++20 buildを確認し、初期化、allocation/pretouch、検証readback/CPU全値走査はcopy計時の外。
同じCPU0へtasksetし、CLOCK_THREAD_CPUTIME_IDとsteady_clockを提出と既存waitに分ける。
CPU時計のoverheadは測定値に含み、差引補正していない。両armの全source/destination lifetimeを
checked drainで保護し、完了不明ならownerを解放せずfail-stopする。実行中env変更無し。

最初の4MiB小GPU確認は各armをUR tracing/parameter validation付きで実施。性能から除外。
cleanはsizeごとordinary/usm/usm/ordinary/ordinary/usmの6独立process、各process内warmup1と
計測3反復。size/armごと3独立process、方向ごと9eligible copy標本を全件保存した。
全14 process・104copy/check pairs（うちclean eligible72）で全pattern、両端guard、source不変性PASS。
D2H前にhost bodyを必ず別patternへpoisonし、未転送のstale sourceで合格しない。

copy-offload-disable=1、EnableDirectSubmission=0、同queue、GPU selector、runtime files/compiler hashを
維持。初期化後の変更無し。driver/boot/fault boundaries/実exit/owned PID start等はreceiptに保存。
全正常終了、新規fault0、boot不変、dump無し。既存production native SHA8fc08439...はbuild前後同一。
probe binary SHA256 2fccadda6686349ad4dbc081511d99f4f6cb85683055797cc21bb28e15f9894c。
追加device allocation最大335544576B、ordinary/USM hostとverificationは各335548672B。

## 結果

表はprocess内3反復の中央値を、3独立processでさらに中央値にした値。個々の標本はreceiptに残す。
各列を独立に中央値化したため、表のsubmitCPUとwaitCPUがtotalCPUに厳密一致するとは限らない。
per-sampleのCPU/wall partitionとbyte throughput式は全件検証した。GB/sは10^9B/s。

| size | host | 方向 | submit CPU ms | wait CPU ms | 全copy wall ms | GB/s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2MiB | ordinary | H2D | 0.082324 | 0.395762 | 0.484714 | 4.326576 |
| 2MiB | ordinary | D2H | 0.535995 | 0.002826 | 0.542814 | 3.863482 |
| 2MiB | usm | H2D | 0.023595 | 0.461145 | 0.491006 | 4.271134 |
| 2MiB | usm | D2H | 0.023723 | 0.451798 | 0.481168 | 4.358461 |
| 320MiB | ordinary | H2D | 33.288885 | 22.056059 | 55.784387 | 6.015022 |
| 320MiB | ordinary | D2H | 60.182604 | 0.012934 | 60.569935 | 5.539783 |
| 320MiB | usm | H2D | 0.072666 | 65.696501 | 65.862859 | 5.094591 |
| 320MiB | usm | D2H | 0.070211 | 59.311056 | 59.422941 | 5.646713 |

320MiB H2DでhostUSMはcall CPUを約33.289msから0.073msへ減らすが、waitが増え、全体は
55.784msから65.863msへ約18.1%遅い。D2Hはcall CPU約60.183msから0.070msへ下がる一方、
wait約59.311msへ移り、全体は60.570msから59.423msへ約1.9%短い。
2MiBはH2Dに明瞭な改善無し、D2Hは約11%短いが、exact actualweight/cohortではない。

## 判断と限界

単純なhostUSM化を長文prefillに採用しない。submitCPUが減っただけでは、CPU負荷も全phase latencyも
改善したとは言えず、今回320MiB H2Dのtotal CPU/wallはむしろ増えた。D2Hだけのcomponent利得で
同じbankの双方向性能改善を主張しない。idle queue、persistent allocation、fixed alignmentであり、
productionの全先行GPU仕事/512連続upload/実queuebackpressureや非整列host pointerは未検証。
今回もpin/import/staging/cacheやdriver内部処理の原因は特定していない。

実測約5–6GB/sは大きなコピーの時間見積もりに使えるが、mapped kernelのfinegrainアクセスや
host groupingの480同期のlatencyへは流用しない。PCIe linkの仕様上限そのものの測定でもない。
演算fallbackやGPU math分布もこのcomponentからは判定できない。

次は47境界のresidualを16bit storageでGPUに多く保持するquality/admission検証。
FP16/BF16のqualityや実allGPU fitは未証明。NLL案はsource reviewと現行git apply --checkが通ったが、
この記載時点では未適用/未build/未GPU検証。速度cohortと全vocab品質診断を混ぜない。
Niko functional ports/default更新、physical262144とRESTOREも継続義務として残る。

raw成功logsはcompact verification後にlock下で退役した。35 files/444160 logical bytes。
元receipt/status/全個別値は不変、manifestに旧path/bytes/hash/reasonを保存。物理空き容量とは呼ばない。
Orca R764–766は同時に完了し、別fileでMain review、出典適用限界、Task/Dispatch/settlementを記録した。
