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
