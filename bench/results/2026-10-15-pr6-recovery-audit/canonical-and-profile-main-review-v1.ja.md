# Canonical bank v3のMain source reviewと次の実測

2026-10-15。actual HEADは9177ad4fb7cc1587255db7235be474aca3eb4e2a。
XeStrataは性能参考値のみ。source参照・移植・改名・sourceに基づく書き直しは行わない。
撤去済み派生診断を含む古いfull mirrorから実装を戻さない。

## 実行中の比較とsource固定

current native8fdcのDirect0/1を事前順0,1,1,0,0,1で各3process測定中。
actual source、link inputs、client、modelは全cohort閉鎖まで固定する。
計測中にbuild/test/別GPU/cleanupを行わず、Mainの作業はsource/report reviewと次recipeの作成のみ。
初回Direct0はactual78453/read78453/reuse0、thinkingON/24出力、PP278.2、native/server0、freshfault0。
一標本を速度採用・ばらつき幅の確定・decode品質に使わない。
compare_native_direct_clean_v1.pyは六標本のidentity/閉鎖/個別値を照合する未実行recipe。

## Canonical bank v3

clean immutable5942f185上のsource.diff44ee495f93be0ca4e7f73f5cbed8d2c57cc86784f00f7539d92b6643d56a0282を
Mainがpolicy、全210行のbank、Weights capture、NativeDense prepare/cancel、Verifier commit retirement、
Prefill PLE分岐、generatorのstartupとtransactionまで読んだ。actual callerのweight accessも照合中。
未適用・未compile・未実行。source上で通常returnにより半端なownerをdecodeへ渡す具体例は未発見だが、
これはruntime/graph/完全contextの安全性や速度の合格ではない。

canonical prepareとnative reserveはreader retirementより前。確定OOMではFull publicationを維持して
reservationを取消す。不明statusやpublication後のupload/drain/free/restore失敗はfail-stop。
Restore guard→Reservations→Exclusionの逆順破棄で、retired後の例外は復元を先に行う。
HC96 norms/lookaheadとPLE/nonlayerをpersistent poolへ残し、private overlayへ当該canonical/native層を合成する。
Verifierのcommit graphもcanonical normを捕捉するためretire対象。KV等のsession stateはrollbackしない。
Full復帰後PLE全pointerをbindしてからpoolをfreeし、SessionGraphs再capture→Verifier復帰の順。

R723はfixtureのpack payload reread/conversionがloader由来descriptorを使う点を指摘した。
独立metadata oracleとは呼ばない。Mainのcanonical_dense_bank_component_fixture_v4.cppで、
BF16/quantized canonical PLE key、value、3norm、conv1dのname-to-pointerを明示照合するようにした。
Bank、Full再bind直後、pool退役後で照合する。現在packはBF16 keyであり、このfixtureだけでは
native quantized PLE variantや実Prefill/graph/MTPを検証できない。v3fixture原本は不変。
R724のfailure state reviewをMainが全文読んだ。source上の具体的UAF/double-freeは未発見だが、
canonical second OOM、native reserve OOM後rollback、partial upload/graph失敗、callback false/throw、
両Full restoreとPLE/recapture失敗の実注入は残る。allocation shimだけで全てを検証したとはしない。

## 反復研究とprofiling

R725はown prompt mode1のK gatherだけがdim-major lane mappingでrow間をstride256B移動し、
V gatherは既にrow内連続であることを区別した。Kだけcell-majorで読み、同じSLM layoutへ置く案は、
global coalescing改善とSLM store conflict増加の両方を測る仮説。未実装・速度未判定。
R726はactual layer-majorにcompatibleなcoarse observer案。旧TIMINGは引き続きrejectする。
Mainは別の6.1Solへ固定9177baseのsource-only最小計測案を依頼し、全test/GPUをMain管理にした。

R726の「observed git HEAD=bd136499...」は誤記。この64桁は実layer_major.incのSHA256そのもの。
Mainのgit rev-parseは9177ad4fb7cc1587255db7235be474aca3eb4e2aで、report記載6source SHA256は全て
actual current sourceと一致した。workspace変更の証拠ではない。原reportは変更せずこの訂正を残す。
coarse queue intervalはhost submission gapも含み、pure kernel/exclusive critical path/PCIe wireとは呼ばない。
特に既存PfTimerのperexpert eventを丸ごと有効にせず、有限の再利用eventと既存drainだけで観測する。

次は全clean比較の閉鎖・証拠commit、その後Main serial下でcanonical候補の実ABI buildと
host/component/fault gates。600/1000目標、natural長入力のFP16品質、実262144/RESTOREは未達。

## Source-only結合案のMain準備

別実装agentからcoarse profile v1を受領し、Mainが全文を読んだ。v1のshort拒否は
公式benchの74token warmupを壊すため採用しない。別v2でshort/defaultを元の経路へ流し、
GDN blockのpairを足した。v2 source.diff80ccb01e06005d8ed1cc20a8aeaf151092d3820b54e1549852643fb2704d1f18。
Mainはv2の全差分/header、own compatのCreate/Record/Query/Elapsed/Destroy実装を読んだ。
12eventを固定再利用し、timestamp取得は既存のchecked drain後のみ。旧perexpert timerは有効にしない。
NativeDensePhaseのhost-only entry/run/restoreも追加。新たなGPU APIやqueue drainはない。
run区間はMTP解放・prefill/callback・MTP復元を含み、pure prefillやdevice serviceとは呼ばない。
OFFはclock/events/logなし。これは未compile・未実行のsource reviewであり、actual API合格ではない。

Main fixture v4に実Prepared cancelと、canonical Prepared後のnative reserve OOM/null→取消し→
通常48層enter/restoreを追加した。既存native専用allocation shimを別targetだけで使い、
canonical-inline allocatorとは分ける。runtimeのstaging/copy/drain/graph失敗やactual generator guardsの
全注入をこれで網羅したとは扱わない。

assemble_prefill_bank_profile_v1.pyはimmutable9177のblobと両提案をRAM内で結合した。
15file/61proposal hunk、combined source.diff25ffa24bd9f4bc195a87836e853a6a529e74f7e5711d13e3657d03b993b4b556。
2箇所だけ明示context調整した。canonical enter/PLE bind完了後へentry_endを置き、
canonical/private overlay callbackを保持したままrun_beginを追加する。その他は一意exact context。
初回assemblyはcallback contextの不一致で出力前に停止した。Mainが明示contextを加え再生成した。
actual product/原提案/測定は変更なし。Fullsource mirrorは増やさずcombineddiffとhash ledgerを保持する。
4つのmanual EXCLUDE_FROM_ALL target（policy/canonical/canonical allocation fault/native reserve fault）を含む。

build_prefill_bank_profile_v1.pyは未実行のMain supervisor。六clean比較完了、全owner退去、
sameboot/fault0、Mainによるexact15source適用を前提に、新build rootで全ABI callersを再buildする。
旧8fdc参照binaryは保存し、CUDA/HIP OFF、既存compiler/znver3設定・own resolverを固定する。
この準備にGPU/build/testは一切含まない。profile host fault/lifetime/output bound qualification、
実componentとモデル/quality/262144/RESTOREは後続Main直列のgateとして残る。

## Clean cohortの失敗閉鎖（上記実行中planを置換）

事前順の4回目Direct0/r1は150042UTCに閉鎖FAIL。長promptの演算前に、指定R_GPU32768と
expert cache/512MiB余裕のadmissionが拒否された。必要3242196992B、報告free3120730112B、
不足121466880B。engine exit1、server0、全owned process退去、sameboot/newfault0/dumpなし。
元FAIL receiptは不変でnative-direct-clean-cohort-closure-v1.jsonに機序と閉鎖を記録した。
残り2回は未実行、自動再試行なし。3成功値はDirect0=278.2、Direct1=273.9/298.4で、
各3回の比較未完了、ばらつき除去・優劣・性能採用の主張はしない。

4回目はstartupfree967MiBに対し先3回1180MiB。Mainの失敗後/proc fdinfo読み取りでは、
B570の別client WebKitWebProces(pid2325671/start_ticks42493913)がVRAM252436KiBを保持していた。
第3native開始42466754の後、第4native開始42504212の前に作られたclientであり、free低下と整合する。
これはexclusiveな原因証明ではない。閉鎖json作成時にはclientは既に消えており、その時点の観測は空。
他process/serviceに対するkill/停止/設定変更やGPU resetはしていない。GPUハングとは扱わない。

source freezeはこのcohortの失敗閉鎖で終了。build supervisorの前提を「全6成功」から
「この確定admission失敗を含む全4processの安全閉鎖」へ明示変更した。性能gateを合格へ変えず、
未完了比較を採用せず、独立したdefaultOFF実装のcompileを許す判断である。
新配置で空きを増やす検討は継続し、後の比較では同じ設定・必要な余裕・新しい独立反復を揃える。
