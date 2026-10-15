# R40960の閉鎖とQSA細分計測

2026-10-16 JST。XeStrataは性能値の参照だけで、今回もsourceを読まず、移植・翻案しない。

## 同nativeでのFP32配置比較

canonical-bank-profile-r40960-model-v1.jsonはPASS。actual/read78453/reuse0、thinking ON、
24生成、C8192、native1694、canonical/profile/Direct1は以前と同じで、FP32 GPU prefixだけを
32768→40960へ変更した。native/server0、48activation→Full復元、全owned identity退去、
sameboot、新規fault0、dumpなし。selected768 FP32行/7864320値はR32768と全byte一致・有限。
same-run stored/input各752行も一致。

| 診断値 | R32768 | R40960 |
| --- | ---: | ---: |
| PP tok/s、各n1 | 331.9 | 332.4 |
| wrapper ms | 235039 | 234702 |
| QSA queue interval ms | 103980 | 104120 |
| MoE queue interval ms | 43937 | 46025.8 |
| residual input queue interval ms | 15883.5 | 13959.8 |
| residual store queue interval ms | 17401.6 | 14991.0 |
| outer restore host ms | 881.363 | 886.808 |

intervalはhost gapを含む非排他的な観測値。GPU中間値は320MiB増え、host input/storeの
requested bytesは各15770583040B減った。wire trafficの測定ではない。
全体の単発値がほぼ同じだったため、効果・分散・採用の結論は出さない。
R32768を次のQSA計測の基準に残す。最大の観測区間QSAの優先度は変わらない。

## 適用・build・host test

6.1 Solから受領したsource-only v1を全レビューし、実dispatch記録と異常系fixtureを
補ったv2を別に受領して全レビューした。途中介入はせず、FINAL後に次の依頼をした。
v2 diff dba9d301385f8d166df50ecd3022d32f0cadf95020a7d82d45816ef488bf37ffを、
全model/comparator閉鎖後にMainがexclusive lock下でactual byte base照合・git applyした。
変更はnew layermajor_qsa_profile.hpp、prefill.cpp、layer_major.incの3fileだけ。

STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE=1を追加。default OFFでcoarse profileを必須とし、
短いwarmupは従来のrun_implを使う。7sectionをchunk単位で14eventにより測る。
projection/KV stage/KV append+querynorm/indexer append/selection/attention/output projection。
per-query/selected-batch/expert markや追加pending wait/drainは無い。
既存checked drain後にtimestampを読む。borrow、Restore、leaseの退役順を維持し、
失敗はincompleteとしownerを早期解放しない。coarse v2 formatterはhash不変。

実matrix_usedからmatrix/fallback chunkとSUM_T、staged chunkとcapを記録する。
全48row+aggregateの固定50行で、QSA各層10chunk/78452queries、aggregate120/941424を照合する。
host spanとqueue intervalはCPU/GPU serviceそのものとは呼ばない。

Mainのhost-shim23caseはPASS。current10chunkとphysical最大32chunkのlayout、
matrix/fallback/staging混在counter、partial creation、Record/Query/Elapsed/Destroy失敗、
NaN/負時間、二重begin、pending finish、sample/cap/query不一致、ordinary/exceptionの
owner lifetimeを確認した。これはruntime/full-model/物理256K合格ではない。

Mainのactual incremental SYCL buildもPASS。new native:
f01e8bd5c9e22f7984594b3928dcefade19b50fb72918c0506986bf25db51241。
recipeと個別command/compiled pinsはqualify_qsa_subprofile_v2.pyと
qsa-subprofile-build-host-v2.json。旧1694 native18671616Bは現observer-overhead/clean比較consumer
向けに32MiB予算で別保持し、fine比較の閉鎖時に再審査する。

## 現在の実モデルgate

run_qsa_subprofile_model_v2.pyをMainが起動した。同78453入力・FP32/R32768条件でfine flagを
追加しただけである。source/binary/link入力を固定し、validation warning/TRACEと有限budgetで監督する。
このreport初版時点では実行中。fineの値・数値比較・速度・障害結果は未判定。
compare_qsa_subprofile_model_v2.pyは閉鎖後に768 selected行とレビュー済み3file変換だけを照合する。

R731/R732はMain全読し、capと算術量をmain-review-round731-732.ja.mdで訂正した。
両reportの提案はfine結果次第。新しいLuna調査R733/R734は最新registryを共有し、
actual fine observer安全性とown prompt kernelのoffline compiled-resource調査に分けている。
Mainは調査と並行してmodel/comparator/reportを進め、GPU実行をagentへ委任していない。

比較済みR40960 raw31457280Bは、同byteのR32768 referenceがありcurrent consumerがないため退役。
canonical-bank-profile-r40960-reference-retention-v1.jsonにpath/hash/replacement/reasonを保存した。
元receiptは不変。残るR32768 referenceとnew fine captureのpaired raw budgetは64MiB、閉鎖後再審査。

600/1000tok/s、task品質、物理262144、RESTORE、実captured SessionGraphs復元は未合格。


## fineモデルの閉鎖と次の実装

qsa-subprofile-model-v2.jsonもPASSで閉鎖した。actual/read78453/reuse0/thinking ON、
24生成、native/server0、48activation→Full復元、sameboot、新規fault0、dumpなし。
compare_qsa_subprofile_model_v2.pyで選択768行/7864320 FP32値は全byte一致・有限、
同runのstore/inputも一致。source変更はレビュー済み3fileだけである。

| QSA細分区間 | queue interval ms | host wall ms |
| --- | ---: | ---: |
| projections | 1856.59 | 66.8032 |
| KV stage | 937.713 | 12.9183 |
| KV append/query norm | 6766.48 | 25.3678 |
| indexer append | 6.69084 | 13.2672 |
| selection | 24399.6 | 133.641 |
| attention | 69909.3 | 9.42802 |
| output projection | 1045.63 | 15.4097 |

各section120samples。実matrix_chunks120、fallback0、matrix query合計941424、
cap2051、staged120を記録した。今度はactual matrix_usedの結果であり、単なる機能存在や
configured Direct flagから採用を推測したものではない。
queue/hostは非排他的な区間で、純GPU service/CPU serviceとは同一視しない。
attention queue区間約70秒に対しhost呼出区間は合計約9.4msで、次の改善対象は実matrix
attention経路。QSAの中では約66.6%、wrapper251647msに対し約27.8%の大きさだった。
selection約24.4秒も次の対象である。KV stageだけを減らしても今回QSA区間全体の大半は残る。

診断PPは331.9→309.9、各n1。coarse QSA103980→104926msに対し、source/input/store等の
区間も増えており、差をfine eventのoverheadや性能デグレだけに帰属しない。
clean反復速度の判定はしていない。追加計測はdefault OFFのまま。

次は6.1 Sol実装agentへown-prompt-k-cellmajor-v1のsource-only提案を依頼した。
R725のown sourceのK読込だけをcell-majorへ変え、同じSLM論理tileへ配置する案で、
V/scales/CH32/WG128/barriers/演算/選択は保持する。SLM conflictによって遅くなる可能性もあり、
component parity、compiled-resource、実component時間とwhole-prefill反復で採否判断する。
R732のCUDA側parity fixtureをconfigured SYCL fixtureと取り違えず、実cmakeと
process環境を途中変更しないtest方式も実装依頼に含めた。
Mainはこの提案が返るまで閉鎖evidence/retention/commitを進め、agentへ途中介入しない。


閉鎖後、Main exclusive lock下で105files/54456525Bのsuccessful fulltrace・duplicate record、
旧1694 binary、review済みfullsource mirrorを退役した。host23caseは各header/aggregateと
対応case順をcompactへ保存し、元receipt全hash不変を前後で検査した。
closed-qsa-diagnostics-{compact,retention}-v1.jsonに証拠を保存する。
最新fine selected raw31457280Bだけを、Main own-prompt-k-cellmajor比較consumer向けに保持する。
このturnの先行二回のraw退役を含む論理削除量は117371085B。snapshot等により実free増分とは異なる。

R734のmetadata queryがcwdへOCL_DRIVER_VERSION15Bを作成していたため、only-report-writeの
記述は正確でなかった。元reportは不変とし、value/hash/reasonを別manifestへ保存して削除した。
以後のmetadata queryもprivate owned directoryで副作用を確認する。

K-cellmajor v1はFINAL受領し全diffをレビューしたが未適用。
6groupしかないnq3 timingをlong-prefillの判定に使わず、normal full correctnessとは別に
nq256のbounded timingを持つv2をFINAL後に依頼した。Mainはその間に本閉鎖記録を進める。
