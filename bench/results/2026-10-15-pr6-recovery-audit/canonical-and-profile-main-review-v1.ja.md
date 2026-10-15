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
