# Main review: head-split v1

2026-10-15。Sol FINAL受領後のsource review。製品適用/build/test/GPU実行はしていない。
現在のDirect0/1 clean比較の製品source/binaryは固定する。

候補kernelはMode1/Direct exact1/selector exact32または64だけで起動する別instantiation。
旧launcherと既定分岐を保つ。8+4 head、全12head共通Q scale、外部head stride12/24、
subgroup一様のQK分岐・全WG barrier、pad4headのzeroとstore guardはsource上整合する。
CH32のonline softmax/scale group/hi-lo MAD/FP32加算順は保つ設計で、
CH64のchunk変更は別の許容差・品質gateが必要。bitwise/速度/SLM実量は未検証。

**v1のGPU fixtureは起動しない。** `set_arm()`がruntime生成後に
setenv/unsetenvでDirect/head-split設定を入れ替える。runtime/driver/collectorの
背景threadによる環境読取との競合がないことを証明できず、R591でraw environの
collector読取も確認済みなので、実行時の設定はexec時から固定する。
これは未実行fixtureのレビュー指摘で、過去GPU障害の原因が実証されたという意味ではない。
元v1と過去成功結果のstatusは変更しない。

SolへFINAL後の別taskとしてv2を依頼した。v1原本を保存し、v2は各native processを
1armに固定し、exec環境を読取検証するだけにする。決定的入力と全output/guardを有限容量の
captureへ保存し、Mainのoffline comparatorがCH32 bitwise、CH64 toleranceとFP64結果を
照合する。環境を内部切替する既存sycl_prompt_attention_testもv2計画から外す。

v1の『Direct0の終了後に適用』条件も補正が必要である。Direct0とDirect1の両方を
同一92acf71e...binaryで測り、所有processの退役を確認してから候補を適用/buildする。
片方の測定後にbinaryを更新すると比較が成立しない。

legacyのint c0+=32はcountがINT_MAXに近い時の末尾加算にoverflow余地がある。
256Kでは発生しない別課題として保持し、今回の旧body保存と混ぜて変更しない。
論理cap262144/ID262143の小fixtureはfull physical256K/RESTOREの認定ではない。
