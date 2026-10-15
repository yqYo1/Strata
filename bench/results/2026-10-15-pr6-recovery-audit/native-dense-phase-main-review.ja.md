# NativeDense phase v2のMainレビューとhost検査

現在のintegration状態: Mainが全6 QSA clean processと比較の正常終了後に実rootへ適用した。
`apply_build_native_dense_phase_v1.py`が別buildで全ABI callerと二つのmanual GPU fixtureをcompile中。
この段階ではcompile/link、実GPU payload、generator transaction、VRAM回収はまだ未合格。
以下の「未適用」は初回レビュー時点の記録。defaultOFF、head-release併用拒否は維持する。

2026-10-15。Solのcaller接続済みproposalは`prefill-native-dense-phase-v2`。
source.diff SHA256 `03a385d0ef0b771d4aa89df18090fe7c602bd81eaa14c9bfa74a8e092f0caffd`。
Mainは14 production fileの全diff（594挿入/32削除）、全fixture/shim、manifest、review、recipeを読んだ。
実production treeへの適用、SYCL build、GPU component/model実行はまだ行っていない。
同時に走るQSA clean測定のsource/binaryは不変。
Mainは全base source hashがreachable commit `77be46f2f46512064bb79954faccc6e0f5c65605`と
一致することも確認した。新policy headerは同commitに存在しない。
Gitには差分/manifest/fixtureを保持し、冗長なproposed全source mirrorは現在のMain検査用working copyとする。

## 実callchainと所有境界

R703/R704は別scopeのLuna read-only FINALで、Mainも原本と実sourceを確認した。
排他的なrequest loop、token/split/peer/window等のguard、Verifier全weight-reader graph退役、
SessionGraphs全配列退役、MTP同期callbackという契約下で、具体的な新規owner違反は見つかっていない。
これはreader完全性の形式証明でも、実driverのretirement結果でもない。

実batched Prefillはcore block helperを呼ばず、current layerとlookahead normの二箇所で
LayerViewを作って直接operatorを呼ぶ。v2はこの二箇所へprivate overlayを明示して渡す。
R694の「block helper overloadも必要かもしれない」という一般論は現callchainには当たらない。
LayerViewのdefault decode経路は元の単一tableを使う。
PleRunはpointer/scalarの値で、local copyへのnative key差し込みは元のhistory/state pointerを保つ。
ready()を差し込み後に評価し、永続ss.ple.wへbank pointerは公開しない。

entryの一層bank確保は全reader退役後・table撤回前。確定OOM/nullだけ通常falseで戻り、
不明なstatus/owner組合せはfail-stopする。full restoreも全owner確保・upload完了後にtableへ公開する。
layer overlayはpass/callbackのchecked drain後に切り替える。cancel/throw/falseでもfull owner、
PLE別publication、以前capture済みのSessionGraphsを復元してからVerifierを再開する。
状態/KVのrollbackと重み/reader capabilityの復元は別であり、前者の成功は主張しない。

R703文中の「startup OOM classifier」は厳密にはentry-bank allocationのpolicy helperを指す。
startup自体がこのhelperを使って全status/pointer故障を検査済みという意味にはしない。
component fixtureのfault injectionもfirst enterだけを対象とする。

## host検査の実結果

`qualify_native_dense_phase_host_v1.py`の初回は、Mainのfake graph handleを整数型にした誤りで
graph-retirement fixtureのcompileが失敗した。実APIのhandleはpointerで、`!=nullptr`を使うためである。
policyとcomposite resolverの二fixtureはそれぞれ正常終了済み。
`native-dense-phase-host-v1.json`の元FAIL/status/bytesは変更しない。

別の`native_dense_graph_retire_host_fixture_v2.cpp`でpointer handleへ訂正した。
`qualify_native_dense_phase_host_v2.py`は二つの既合格fixture結果をhash参照で再利用し、
proposed session.cppの**実`session_graphs_retire_dense`関数本体をそのまま抽出・compile**した。
CUDA result/handle providerだけfakeで、関数の制御をfixtureへ書き直していない。
mainが同じmeasurement lock下でGPU run間に実行し、**合計43ケースPASS**。

- 既合格policy/resolverの二fixture。
- 空graph ownerと、全37 handleを埋めたownerの全破棄/reset/二回目退役。
- null streamとsync failureがproduction stop messageでSIGABRT。
- 37個のdestroyそれぞれでfailureを注入し、production stop messageでSIGABRT、通常returnなし。

gr.n=3、shared_capacity=5と異なる長さを使い、routes/sharedと全exec/post/preA/preB/preP配列を確認した。
core dumpはchildで無効。全owned host processは退去済み。GPU runtimeを初期化していない。
`native-dense-phase-host-v2.json`に個別exit、stdout/stderr、所有証拠、compiler/source/binary hashを保存した。
この検査は実driver graph destroyやfull generator transactionを通していない。

## 次の実行と未合格範囲

v2はhead-release1との併用を明示rejectする。独立した移行段階の制限であり、
両方のVRAM削減を同時達成したものではない。将来はreader退役/復帰を一つのtransactionへ統合する。
native-only理論回収2066032640 Bにcanonical1538035200 Bは含まれず、
ユーザーのcombined一層bank仮定をまだ実装していない。

次はfull caller compile、actual NativeDense payload/metadataとallocation fault fixture、
実generator transactionのcaptured/uncaptured・pending commit・native PLE・callback/cancel/throw・
OOM/uncertain failureの検証。request serial ownershipと全graph reader coverageのruntime receiptも必要。
その後に長文状態/数値、RESTORE継続、物理262144とclean反復を確認する。
host43ケースを実VRAM回収、モデル品質、物理contextや速度の証拠に読み替えない。

source-only実装担当には別dir `physical-context-consume-v1`でfull-context producerも依頼した。
通常CLI/APIのmax-new意味を変えず、最後prompt tokenを実MTP4 verifierへ通すもの。
それもMainのbuild/test/GPU管理下で検証するまで合格扱いしない。

物理context producer v2のsource proposalにはbase raceが見つかった。
Solが古いproduction由来mirrorを編集する間にMainが上記bankを適用し、diff生成時のbaseだけが新しくなったため、
v2のdiffはbankのclass/関数/caller/declarationを削除する内容だった。Mainは適用していない。
元v2はそのまま保存し、immutable bank-v2 proposalをbaseにした統合v3を別dirで依頼した。
以降、source proposalのbaseはmutable worktreeでなく、reachable commitまたは固定hashのmirrorを指定する。
