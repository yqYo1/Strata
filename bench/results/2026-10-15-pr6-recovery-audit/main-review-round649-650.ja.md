# 静的profile候補の準備と適応交換のsource確認

2026-10-15。前goal turnは実CPU count/profileのoffline照合とR647/R648のレビューを
f792d657まで保存したprogress。今回も同じexec16956を直接pollしてrunningを確認した。
最後のvaried-prose run3はthinking2089から3291まで進行中。新しい完了標本やsuite終端はまだない。
新CPU benchmark/unit test/build/GPU/cleanupは実行せず、現在のroot source/client/b1 binaryと
測定profileは固定している。小さな保存済みcountから別の設定fileを準備した。

## 実際に生成したprofile候補

`prepare_cpu_count_profile_v1.py`を実行し、既存`tools/make_profile.py`のrank_profile(reorder=True)/
write_profileを使って、専用のcpu-history-first-profile.binを生成した。
実行commandは`python3 -B bench/results/2026-10-15-pr6-recovery-audit/prepare_cpu_count_profile_v1.py`。
pathはcpu-count-profile-preparation-v1.json、SHAは
d97b2f483b00341001883537ea4848049b26944c0d14f668b5c43aa88a7a5456。
196632 Bで、全24576pairと逆引き表を含む。既存data/expert-profile.binは変更していない。

入力はclosed cpu-gu-reuse-long-v1.jsonの最初の要求だけ。jobs68781/rows101036/unique9816と
encoded count hashを照合し、観測CPU expert-token rowsの降順・pair ID順で既存writerへ渡した。
CPUで観測しなかったpairは元profile順で後ろを埋める。旧GPU-owned ID別頻度は不明なので、
それらを「実際の頻度ゼロ」として計測したprofileではない。この偏りを明示した実験候補である。
実binaryのheader、全順位、全逆引きを読み直し、legacy128 quotaの選択は128pairs/10159known
training rowsで先の算術と一致した。これは同一学習要求上のcountで、held-outの達成値ではない。

preparation receiptにrecipe/count-reader/writer/input/prior/candidate hashes、Python、準備処理の
scopeとconsumerを保存した。実native loader、モデル、GPUは起動していない。
native_loader_qualified/performance_qualified/quality_qualified/adoptedはいずれもfalse。
sourceとcandidateのhashを別のread-only照合でも確認した。fileのconsumerは同じlegacy128 quotaの
held-out bench.sh pilotで、byte budget196632、判断または置換時に保持を再確認する。

## R649の独立確認

Lunaは前の上下界・quota・同容量比較を独立source/countレビューし、算術訂正を見つけなかった。
R607の旧128を残すappendと、total192/256への再partitionを混同しないこと、過去profile内容を
producerがhashしなかった条件付き性を再確認した。Mainはその限界を候補preparationにも残した。
既存の成功receiptや過去profile内容の証拠を書き換えていない。

## R650のscope訂正と実際の未確認API

LunaのFINAL後、Mainが実active routeを再確認した。v1はstandaloneの行を主に使い、device mapの
失敗がordinary SYCLに古いslotを読ませると説明した。しかしverify.cppはSYCLでdevice_plan_=falseを
強制し、GPU pointerはexpert_source.cppのhost_res→host planから来る。Mainはこの違いとserve側の
実call sitesを伝え、v1を残して別v2の訂正を受領した。調査中の途中介入はしていない。

evicted outは直ちにhost_resで非resident、incoming inはpending完了まで非residentとなる。
必要なresident-complementでは先にD2H checked waitとoverrideを準備する。よってv1の
「no-waitだけで旧GPU slotを読む」という説明もこのhost planには当たらない。source storageや
他のmodeまで安全と一般化せず、今のstatic baselineは変更しない。

Mainが確認したserve側の実不足はgenerate.cpp:5127のcudaEventRecord戻り値無視である。
pendingのH2Dを代表するevent記録が失敗したのに、古い/空のeventを完了として扱うと、
apply_pendingがincoming slotをhost_resへ公開し得る。host planでもこれは後続consumerへ影響する。
serveのdecode前applyは6043、request boundaryは5864、adapt thread開始は6072にある。
standalone6925にも同じevent-record不足がある。ignored res_upload/d_res copyはエラー処理の不足
だが、現在のordinary SYCL verifierはそのdevice mapでexpertを決めない。適応の問題とPCIe miss
fetch修正を同一経路として扱わない。commit_exchangesの短い成功countも別のsource-tier課題である。

これはsourceで確認した潜在的な失敗経路で、過去のGPU障害原因や今回の障害発生の証拠ではない。
現在はadapt-swaps0なので、この発見で実行中測定を止めない。正の適応交換を試す前にevent結果の
伝播とコピー完了/owner保持を修正・検証する。単にAPI失敗でreturnしてunknown DMAのstorageを
解放する修正では足りない。今回、製品sourceの変更や適応を有効にする測定は行っていない。

現full OFFの終端→旧nonce回復→比較toolのCPU検証→実gateに応じたIQ ON比較の順序は維持する。
profile候補は次の別armの材料として準備したもので、現比較へ混ぜない。1000 PP/70 TG、PR6回復、
物理262144/RESTOREとtask品質は未達である。

原R650 v1/v2の末尾空行にgit diff --checkのwarningが各一件ある。codeではなく、agentから
受領した原bytes/hashを保つため変更していない。これら2report以外の変更のwhitespace確認と
実test/buildを区別する。
