# NativeDense phase v2のMainレビューとhost検査

現在のintegration状態: Mainが全6 QSA clean processと比較の正常終了後に実rootへ適用した。
`apply_build_native_dense_phase_v1.py`の120 build stepsとactual generator compile/linkはPASS。
実GPU componentの修正後6armもPASS。実78453入力のgenerator transactionも正常終了し、
driver free増加とselected FP32 captureの保存hash一致を確認した。
全状態の数値・実RESTORE・物理262144・clean速度はまだ未合格。
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

## 実GPU componentとfixture設定修正

初回`native-dense-bank-component-normal-v1.json`はFAIL。fixtureが`generate --native`のskip setを
渡さず、metadata-onlyの`output.weight`をcanonical payloadとして読もうとしてabortした。
runtime API errorではなく正確なloader拒否で、新規fault0、dumpなし、全owner退去。
元のFAIL receipt/statusは不変。productionのbank/数値処理はこの修正で変えていない。

Mainはfixtureにserved_names、native head/embedding skip、非量子化PLE key保持を適用した。
canonical arena1538035200 Bを独立したcensusに照合し、実運用と同じskip setをpool_bytes/loadへ渡した。
`build_native_dense_component_v2.py`は二つのmanual fixtureだけをrebuildし、actual product binaryは不変。

`check_native_dense_bank_component_v2.py`をMainがlock下で直列実行した結果:

| arm | exit | 結果 |
| --- | ---: | --- |
| normal | 0 | GGUF/RAM/device全300 payload・全metadata、48層activation、全restore、再enter/restore PASS |
| oom-null | 0 | Full/table/viewを保持した通常拒否、その後の実enter/payload/restore PASS |
| unknown-null | -6 | 正確なproduction allocation fail-stop、通常returnなし |
| unknown-nonnull | -6 | 同上、nonnull ownerの不確実性を通常拒否としない |
| oom-nonnull | -6 | 同上、OOMでもnonnull ownerを通常拒否としない |
| success-null | -6 | 正確なsuccessful-allocation/null-owner fail-stop |

各armの個別receiptは`native-dense-bank-component-*-v2.json`。
すべて同boot、新規GPU fault0、dumpなし、全owned process退去。初回normalはflushed UR trace/validation、
故障armはwarning validationをexec前に固定した。SIGABRTだけで期待故障PASSとせずoperation messageも照合した。
このcomponentは実generatorのgraph retirementや全モデル数学を通さない。

## 実モデル検証の観測

Mainは`STRATA_TRACE`時だけouter generator transactionのretired/bank/restored各境界を記録するよう追加した。
actual NativeDense state、snapshot/full/bank bytes、activation回数、overlay、graph再capture有無、
Verifier有無とdriverのfree/total queryを同時に記録する。未知のquery completionはfail-stopする。
`build_native_dense_observer_v3.py`はgeneratorのみrebuildし、全component archive/fixtureのhash不変を確認してPASS。
新native SHA256 `291da414dbd55efb3142a47fbca18296e17fa8b2bb680f9e3f1861e01ecb94e1`。

`run_native_dense_bank_model_v1.py 1`はofficial bench.sh、short warmup後の実78453入力、thinking ON、24出力。
C8192、FP32/R_GPU0、alias1、QSA1を固定し、native bank1/head-release0/明示no-token-graphで実行中。
48activation後のFull復元、旧QSA FP32 referenceのselected768行capture hash、同run store/input照合、
実Verifier生成と通常終了/新規faultを確認する。raw referenceは既に退役済みなので、照合は保存hashとの比較と明記する。
instrumented結果をclean速度へ読み替えず、容量変更/R_GPU/16bitの速度実験はその後に行う。

この実モデルrunはその後正常終了した（`native-dense-bank-model-1-v1.json`）。
short warmup後の78453入力と24生成、48 native-layer activation、Full復元、native/server0、owner退去、
新規fault0、dumpなし。selected768 FP32行captureは旧referenceの保存SHA256と一致し、
同runの752 store/input行も一致した。SessionGraphs再capture観測は0で、このrunがcaptured SessionGraphsを
実際に退役した証拠とはしない。warm decode後のVerifierは存在した。
bank前free826142720 B→bank後2666852352 B、driver queryの増加は1840709632 B (約1.714 GiB)。
payload差2066032640 Bとは一致しないため、payload差を物理回収量と言わない。
Full復元後free776048640 B。free差だけからleakの有無やallocation-pool原因は断定しない。
診断PP240.2 tok/sはinstrumented一標本で、速度採用しない。
次は同native・同78453入力でFP32 hiddenのGPU prefix32768行 (1342177280 B) を実際に使う。
現runのraw31,457,280 Bはその比較の明示consumer向けに保持し、paired raw予算64 MiBとする。

run閉鎖後、`retire_qsa_clean_dense_component_logs_v1.py`をlock下で実行し、closed QSA6 sampleと
component7 armの不要trace/journal/protocol/外部duplicate recordを115 files・4255867 B削除した。
元FAIL/PASS receiptは不変、全UR entry/return成功件数等をcompact化し、path/hash/reason manifestを保存。
現在のmodel raw/native診断・source/build/modelは削除していない。
