# 入力比較の再監査とcache profileの実count照合

2026-10-15。前goal turnはR645/R646を受けて比較toolと失敗記録を修正し、82a98c67まで
commit/pushしたprogress。今回も同じexec16956を直接pollしrunningを確認した。rootの
product/client/native binaryは固定。新CPU test/build/GPU/cleanupは行わず、保存済みの
小さなJSON/count/profileに対するoffline算術だけを行った。これは新しい推論の実行ではない。

## 正式測定の進行

full OFFのrun2 regular-codeがrequestとして完了した。4164 actual completion tokens、
decode9.7 tok/s、accepted/offered2961/3612、eligible=true。run1 code3346/9.8、prose8631/6.2、
run2 prose9056/6.4の後の標本である。3巡目と最終suite/owner/fault gateはまだ未完了。
この同serverの反復を独立processのばらつき、品質合格や改善採用として扱わない。

その後run3 regular-codeも3198 actual completion tokens、decode9.8 tok/s、accepted/offered
2277/2766、eligible=trueで完了した。code3標本は9.8/9.7/9.8 tok/sとなり、最後のvaried-prose
run3のprompt33288を処理中である。code cohort完了をsuite全体の最終合格へ広げない。

## 現profileを過去のCPU expert ID countsへ照合

以前のcpu-gpu-cache-opportunity-v1はCPU routesの後知恵の順位付けであり、現在のprofileを
そのまま使ってslotsを増やした結果ではなかった。今回はclosedのcpu-gu-reuse-long-v1.jsonに
残る実CPU `(layer,expert,nt)` countsと、現在のdata/expert-profile.binを照合した。
analyze_profile_cache_coverage_v1.pyを実行し、profile-cache-coverage-bounds-v1.jsonへ保存した。
入力byte上限はJSON1 MiB/profile256 KiB、実処理は小さなcount列のみ。CLIは
`python3 -B bench/results/2026-10-15-pr6-recovery-audit/analyze_profile_cache_coverage_v1.py`。
モデル・kernel・benchmark・unit testは起動しておらず、GPU_executed=falseである。

元receipt SHA17ddf01d…と現profile SHA8f59b4aa…を固定し、全24576pairの一意性、98304counter、
CPU jobs68781/rows101036/unique9816、layer別CPU count、CPU/GPU以外のownerゼロと全104160
route entriesを照合した。現在のlegacy128 resident集合に観測CPU rowsは0だった。
さらにMainは実cache.cpp/header/slot helper/generate.cppの4source hashが、この元receiptの
source_pinsと一致することを別のread-only照合で確認した。

重要な限界: 元CPU-census producerはprofileのpathを記録したが、その内容hashを保存して
いなかった。したがって再構成したresident集合の結論は「同じ現profileが当時も使われた」
という条件付きである。CPU countとの非重複は整合性であり、過去profile内容の証明ではない。
現在進行中full OFFは同じ8f59b4aa…をpack metadataとしてpinしているが、そのことを過去runの
不足する証拠へ置き換えない。原receiptも新たなhash/PASSで書き換えない。

元のGPU-owned ID別countsは無いので、layerごとのGPU総countを使い、旧resident全保持なら
全count保持、一部だけなら0〜全count、保持無しなら0とした安全な範囲を算出した。
特に旧last layerは34slots/646GPU rowsで、192/256へtotalを変えると4/21slotsへ減る。
追加のcache容量は旧setへの単純appendではない。

| 同じ過去routeを固定した条件付き配置 | Arena B | カバーするroute rows | 全routeに占める率 |
| --- | ---: | ---: | ---: |
| 現profile / legacy128 | 281651200 | 3124 | 約3.00% |
| 現profile / legacy192 | 392908800 | 4617〜5263 | 約4.43〜5.05% |
| 現profile / legacy256 | 533734400 | 5268〜5914 | 約5.06〜5.68% |
| 同requestのCPU count順位 / legacy128 quota | 281651200 | 10159 | 約9.75% |
| 同requestのCPU count順位 / global128選択 | 266342400 | 12325 | 約11.83% |

後二行は同じ要求を学習と評価に使った後知恵で、最良profileの達成値や速度ではない。
現在のprofileのままcache増量する候補は、旧追加128理想上限12.20% CPU rowsをそのまま
得る案ではなかったことが明瞭になった。次のdecode設定試験は容量だけのgridを先に広げず、
小さい固定容量のままprofile順位を別の要求で評価するpilotを候補に加える。profileを変える
場合はGPU Q8_1/CPU IQの算術差と実route/MTP仕事も変わり得るので、task品質とclean反復速度を
確認する。新profile fileは今回生成せず、data/expert-profile.binも変更していない。
実VRAM fit、転送量、CPU/GPU wall削減や70 tok/sをこの算術から予測しない。

## R647 / R648の独立レビュー

Luna2名は過去report置場を共有した反復read-only調査。MainはFINALまで介入せず、上の
実count照合を並行した。R647はv2が本物のv1 reader/6testsを再利用し、fixed/turn0 hashと
全request構造・設定を保つこと、実resultsのagent_filesと不変builder/tagの組合せを確認した。
具体的blockerは無い。focused testがagent行の実reorderを含んでいなかったため、Mainは
該当test内にturn1/2を交換するsubcaseを追加しmanifestを更新した。期待11testsは変わらず、
すべて未実行。source reviewを実host PASSにはしない。

R648は既存IQ3_XXS subgroup v2のsource span/lane/tailの明確な欠陥を見つけず、必要な実行を
整理した。元per-value type18のne64約9.2%低下は不採用のままで、新SG16/32案の勝敗ではない。
実build、device IR/必要幅/collective loweringのMain確認、qualified controllerのquick69/full75
arrays・20negative・F32・route census・正常closure/fault、独立clean shape/whole-route測定が
未完了。SG幅をsource属性だけで支持済みとせず、defaultOFFを保つ。新observerは増やさない。

現在のfull OFFを閉じ、旧nonce回復→host比較tool検証→実gateに応じたIQ ON比較を先行する。
既存候補の検証、profile pilot、MTP固定幅とchunk幅の調整は結果を見て選ぶ。古い失敗や未実行の
別armを永久条件にしない。PR6回復、1000 PP/70 TG、物理262144/RESTOREと品質は引き続き未達。
