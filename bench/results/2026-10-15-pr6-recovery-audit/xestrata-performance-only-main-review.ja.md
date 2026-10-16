# XeStrataを性能参考値に限定する方針と現行sourceの処置

2026-10-15。ユーザーの最新指示は「XeStrataにはライセンスの問題があるので、性能の参考値に留め、移植しない」。法的解釈や互換性の断定で置き換えず、実装・helper・診断・test・保存source提案にも適用する。
AGENTS.md、docs/UPSTREAM_INTEGRATION.md、project-only strata-upstream-sync本体と二つのreferenceへ反映した。
ユーザー空間へskillを追加していない。skill-creatorの既存validatorは正常終了した。

## 確認した取り込みと実撤去

0f816385にはXeStrata表記のdecode_route_ledger.hpp、decode_service_budget.hpp、CPU service_budget.hppが現存した。
R715がown-treeの実callerと履歴を調べ、Mainも全非delete差分をレビューした。
3header、対応fixture、request scopeとCPU worker instrumentation、generate/dispatchの全hookを撤去した。
単なる著作権表示の削除・改名はしていない。

CpuGuReuseObserverはR717でも独立出自を確定できなかった。
元のledgerとのrequest bindingが失われた診断を別名で再構築せず、header・fixture・全専用hookも追加撤去した。
独立したMTP record/upload lifetime・size/overflow検査、GPU callback drain、SessionGraphs/Verifier retirement、
CPU pool本来のwait_done/wait_parked/atomic completion、watchdog join、prefill logitsは保持した。
native pool例外時には最小の独立catchでGPU waitを解放してabortし、借用ownerをunwindしない。

Mainのfresh actual115step SYCL buildは正常終了。その後GU撤去分を実recompileして正常終了。
元v1build receiptは不変、v2には上書きされた中間build productのold/new hashと退役理由を記録した。
現在native SHA256=8fdc241d5227f7bde921a8bbdccf9535d6b263d35237751488338cdf438b89de。
原productの数値演算・MTP/session/safety source pins不変を両buildで確認した。
active include/src/tests/CMakeの退役symbol参照とXeStrata SPDX検索は0件。
これを全歴史・全外部依存に対する法的保証や包括的なsource独立性の証明とは扱わない。

## source artifactと再取り込み防止

既存の派生source・patch・生成/適用recipe、混在した古いfullsource mirror、由来未確定GUのsourceを現行treeから退役した。
86 files / 11413962 B。license-source-retirement-v1.jsonにpath/size/SHA/reasonを保存した。
元測定のFAIL/PASS・個別sample・compactreportは変更していない。Git履歴のrewrite/force-pushは行わない。
旧source proposalを後日使う際は、残った古いmirrorやbaseへ戻さず、撤去後のactualsourceをimmutablebaseとして
独立した変更だけを再構成・レビューする。canonical dense bank v1/v2とphysical consume v3も、
古いfull generator mirrorは退役したため、保存された独立diff/metadataとcleanbaseから再構成する必要がある。
Main適用用removal差分に含まれる削除元code全文も、適用後のcurrenttreeへ恒久保存しない。

## 選択sourceと履歴の区別、調査の訂正

R716のDirect1説明はfallback Qlayoutと混同していた。実Direct1はSTRATA_SYCL_PROMPT_DIRECTで選ぶ
prompt Direct-outputで、own-fork cd4d4d97まで起源がたどれる。84374f68のfallback Qlayoutは別経路。
Mainはactualdispatchを読み、R718/R719にもこの訂正を残した。元R716reportは変更しない。

R718はreachableな旧Xe系統df4f8742/378770aeをcurrentCPU sourceの採用と取り違えた。
Mainのrevision固定blameとR719独立再確認では、pool/native_expert/expert_layout/iq_avx2の現4fileに
両commit由来行は0。Niko/pipeob0等の選択されたsourceとown修正であり、旧系統の履歴保存と区別した。
R718の該当結論を撤回する別reportを残し、現CPU推論部品を誤って旧Xe系統へ置換・巻戻ししていない。
marker不在だけを由来証明に使わず、実blob/hash/blame/CMake resolverを照合した。

compiler/runtimeがXeStrata作業tree内の.tools/intel-llvm-v711/installに置かれているのは別の配置依存。
実resolverはこのforkのroot src/includeと既存ggmlを選び、XeStrata checkoutから製品sourceを取り込まない。
今回toolchainは変更していない。格納pathをsource移植の根拠にはしない。

## 最適化の実進捗と現在の検証

削除前の同native・同78453入力・FP32/R_GPU32768/nativebank1/QSA1で、Direct0/1診断が閉鎖した。
PPは251.8→303.2 tok/s、各一標本・instrumented。全48層×選択16行=768 FP32行はbyte一致・有限、
両runの752 store/input行も一致。native/server0、48activation→Fullrestore、新規fault0、dumpなし。
これはclean反復の速度採用やtask品質・RESTORE・物理262144の証明ではない。
Direct0の31,457,280 B rawは比較閉鎖後に退役し、Direct1だけを新native比較のconsumer向けに残した。

新nativeでDirect1、GPU32768、同official bench.sh/78453入力・thinkingON/24出力を比較する
run_license_clean_prefill_model_v1.pyをMainが起動した。sourceはrun中固定、validation/logはexec前に設定する。
このreport初版時点では実行中で、合格・速度・数値は未判定。閉鎖receiptと比較を次に記録する。
目標prefill反復600超→1000、decode70は未達で、goalは継続する。


## 新nativeの実行閉鎖と前後比較

license-clean-prefill-model-v1.jsonは正常閉鎖した。実78453入力/reuse0/thinkingON/24生成、
48activation→Full復元、native/server0、全owned process退去、新規fault0、dumpなし。
比較recipeはsource変更をremoval v1/v2のexact hash chainに限定し、数値演算/MTP/sessionの
source不変と同入力・設定を確認した。選択768 FP32行は撤去前のDirect1と全byte一致・有限、
両runの752 store/input行も一致（license-clean-prefill-model-comparison-v1.json）。
診断PPは303.2→293.8、各一標本。source撤去の速度効果やばらつき幅をこの二点から断定しない。
clean独立反復とDirect0/1の部品別性能判定は未完了。

閉鎖後にDirect0/Direct1/撤去後の成功trace・requestjournal・外部duplicate record等をcompact化・検証し、
37 files / 33886835 Bを退役した。最新のFP32 selected raw31457280 Bだけを、
同workloadのcanonical dense bank v3比較consumer向けに32 MiB予算で保持する。
closed-direct-diagnostics-{compact,retention}-v1.jsonに証拠を保存した。元FAIL/PASSや個別速度は不変。

canonical extension v3はcleanbase5942f185へ34hunkを再構成したsource-only提案として戻った。
旧診断の復帰なし、TRACE/GpuStage/MTP helper保持hashを伴う。diffSHAは
44ee495f93be0ca4e7f73f5cbed8d2c57cc86784f00f7539d92b6643d56a0282。
Mainの全source/host/component/actualcaller qualification前であり、未適用・未compile・未採用。

R721をMainがsourceへ照合した。FP16はphase-auditを外せば既存typed boundary auditを使える。
保存/復帰の同run mechanicsとFP32対FP16の品質は別。BF16は未対応・parser拒否であり、
boolean half auditをそのままBF16へ使わない。natural long FP16品質とdense NLLは未合格。
R720の最小timer案はMainが却下した。現環境はlayer-major=1で、actualadmissionと全48層完了がある。
layer_major.inc:69はSTRATA_PREFILL_TIMINGを明示rejectするので、「flag追加だけで現在routeを測れる」は誤り。
この未実行案をprofile結果や実行可能recipeとして扱わない。元reportは保存し、訂正を別に記録する。

R722の別訂正reportを受領した。実layer-major routeとTIMING拒否を認め、現在source/binary pinsを使う
unitrace --device-timing集約を未実行の候補とした。全process集計をphase排他時間やcritical pathとは呼ばない。
