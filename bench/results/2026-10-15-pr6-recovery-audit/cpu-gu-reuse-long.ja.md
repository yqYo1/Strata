# 実モデルのCPU GU再利用と採否

2026-10-15、Arc B570 10GiB / Ryzen 5600X / RAM128GiB。
`check_cpu_gu_reuse_v1.py long`が34603入力・128生成・通常MTP4で完了した。
実binaryは92acf71e...、chunk4096、int8 KV、resident32768、context設定262144、
FP32 layer-major残差GPU4096行、Direct1、合計128 GPU expert slots。
診断付き1回であり、正式な速度比較ではない。

128生成IDは既存基準と一致した。W55/R217/D162/A76/C131/E128、
CPU68781jobs/101036 expert-token rows/全route104160が一致。
正常終了・所有process回収・新規GPU障害0。
観測ONのCPU GU状態は1.69MiB程度のmetadataであり、GB単位のcacheは確保していない。

## request-cold仮想LRUの実観測

| payload予算 | all-nt hit / miss | all-nt hit率 | nt1-only hit / miss | nt1-only hit率 |
| --- | ---: | ---: | ---: | ---: |
| 256MiB | 0 / 67790 | 0% | 0 / 45433 | 0% |
| 1GiB | 0 / 67790 | 0% | 0 / 45433 | 0% |
| 4GiB | 18349 / 49441 | 27.07% | 14020 / 31413 | 30.86% |
| 8GiB | 44607 / 23183 | 65.80% | 24630 / 20803 | 54.21% |

未知形式991jobsは両policyともbypass。globalで9816 unique experts、58965 retouches。
このretouch数だけではcapacity付きのhitや展開の償却を証明しない。
nt1-onlyはnt2..4をadmit/touchしない独立状態で、all-ntのnt1行を後から抽出していない。
nt1性能だけが測定済みなので、現段階の採否はnt1-onlyで判断する。
98304個のper-expert nt count、format別hit/miss、fill/eviction/residentの保存は
`cpu-gu-reuse-long-v1.json`に保持した。

## 準備コストを入れると同期cacheは不利

`analyze_cpu_gu_reuse_v1.py`は4個の独立CPU processの単体実測単価と実miss数を照合した。
missの呼出し自身もexpanded kernelで処理し、全eligible jobsで得る計算節約を数えた。
全missが既存mappingのoverwrite refill単価で済む仮定でも、現行GGMLに対する
thread CPU serviceの差は以下となった。負値は仕事が増える予測である。

| payload予算 | 4 processの単価によるnet CPU service差 |
| --- | ---: |
| 256MiB / 1GiB | -13.56〜-11.42秒 |
| 4GiB | -7.82〜-6.51秒 |
| 8GiB | -3.48〜-2.80秒 |

cold単価ではさらに不利で、既存exact helperをpacked対照にした別計算でも同方向。
これはsingle-thread単価の算術scenarioであり、model wallの増減や厳密な下限ではない。
実worker並列、fillの重ね合わせ、persistent cache、選択的admissionやslot poolの物理状態は
別設計である。今回の128生成で不利だから長い全sessionのcacheが不可能とも言わない。

現時点では**同期的な全GU request-cold展開cacheは実装しない**。
packed処理を維持し、実測で有望なAttention Directのclean比較と、
より多くのCPU expertをGPUへ移せるVRAM・cache配分の検討を優先する。
nt2..4未測定、live RAM実cache未確保、全model品質・物理256K/RESTORE未認定。

## host時間調査のMain補正

R604のsource対応表は有用だが、一部の範囲は実native pathに合わせて補正する。
`src/core/verify.cpp`の`ms_wait`開始aは前のgraph launchより後の
`cudaEventSynchronize(layer_done_)`直前。`ms_pool`はbからcallback・flag処理まで。
後続`cudaStreamSynchronize(copy_)`とnext graph launchは両時計の外である。
したがってindex0をgraph launch費も含む時計とは扱わない。
index15もnativeでは`run_split_multi_native`全体の経過時間で、
`run_phase`内部はこの3時計を個別には加算しない。

この実測ではindex11=22.219秒、0=7.643秒、1=8.663秒、2=0.255秒、
draft index13=3.255秒。未計測scopeにはgraph capture/submit、copy同期、最後の
queue同期等がある。異なるthreadのCPU serviceやdevice durationを引いてidleと呼ばず、
sourceで範囲を限定したhost timingを次の診断候補とする。
R606がこの実queue/event所有と未計測scopeを追加調査中。

## 進行中の正式比較

Direct OFF側の公式`bench.sh --long --think --long-generation --runs 3`を開始した。
private outputは`main-primary-prompt-direct-bench-full-0-root-v1`。
同じbinaryのON側はOFF終了を確認してから別processで実施する。
各32K promptと3 agent sessions、thinking有効のstrong/weak長文生成が対象。
実入力token数を報告し、32768以上のlong-generationで長文prefillを判断する。
診断・profiler・ledgerは無効で、既存GPU障害回避設定は維持している。
1000 PP/70 TG、PR6速度回復、品質・物理256K/RESTOREは引き続き未達/未認定。

閉じた52ファイルの重複tensor/log/providerを`retention-v7-manifest.json`で検証して削除した。
論理10777893B、固有file allocation11275264B。空き容量の実増加とは主張しない。
canonical selected-head/first-logit、元の数値FAIL fixture、source/model/binaryは保持した。

## 追加調査606/607のMain確認

R606のqueue/event契約から、graph完了だけで別copy queueやhost callback、
そのsource bufferの再使用、sticky async errorの全てを証明できないことを確認した。
同期を削除せず、まずsubmit/copy同期/末尾同期のhost bracketを測る方針とする。
R606本文の「ms_waitは直前graph submitから」という表現も、前節のsource位置に合わせて
**aはsubmit後でありsubmit時間を含まない**と読み替える。元reportは変更しない。

R607のGPU cache機会をMainが`analyze_cpu_gpu_cache_opportunity_v1.py`で独立再集計した。
CPU histogramだけを後知恵で順位付けする条件付き上限では、追加128/256/512 expertが
観測CPU rowsの12.20%/20.08%/32.09%を覆う。
明示した同順位規則（rows降順、layer/expert昇順）で選ぶglobal128のraw payloadは
266342400B。R607の266343424Bとは一致せず、Mainの構造化再計算を使う。
legacy層別追加のjob数にも同順位規則で差があるため、reportの未指定tieを推測しない。
Mainは追加quotaを先頭層に均すbalanced版も保存した（既存128 residentを概念上維持する案）。
GPU-owned IDsの頻度はcensusに無く、追加候補と実resident集合の非重複も未確認なので、
実cacheの最適解・admission・将来hit率・時間短縮とは扱わない。
page/formatごとの実bytes、live VRAM、graph再capture、quality/RESTOREの検証を要する。
