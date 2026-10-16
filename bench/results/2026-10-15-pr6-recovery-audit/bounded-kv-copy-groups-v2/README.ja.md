# KV copy groups v2 — 明示geometryと完了不明時の所有者保持

2026-10-15。新規v2 directory内だけの未適用proposal。製品変更、build、test、GPU、Git、service操作、cleanupは実施していない。
v1は変更していない。Mainから指定された保存commitは `4b1bdc2733e7cc8e089c04f29e0b075fd01a0821`。
`pins.json` にv1全fileのhashを記録し、`regen.py` はそれらと現在の製品sourceの一致を確認してから、本v2内だけへ候補とdiffを生成する。
Mainのphase診断buildを変更・実行・停止していない。compile、correctness、fault、性能は全て未認定。

## v1から変えた点

v1 componentはruntime初期化後に `setenv/unsetenv` してarmを切り替えていた。
Mainが確認したPTI providerのraw environ並行走査、およびUR/driver等の環境読取と共存させないため、v2 fixtureは**process環境を一切変更しない**。
環境が静的でもABBAを同じRuntime/queueで行えるよう、geometryを通常の整数引数として渡す。

製品側の既存 `kv_stream_resolve` のsignatureとcallerはそのまま。
起動前から固定された `STRATA_SYCL_KV_COPY_GROUPS` を読み、追加entrypoint
`kv_stream_resolve_with_copy_groups(..., stream, uint32_t copy_group_cap)` へ委譲する。
追加entrypointに従来の全validation、resolver single_task、CLOCK、counter、overflow、copy、finishを置き、
production wrapperとfixtureが同じ実装を通る。kernel本体をfixtureへ複製しない。

- `0` は従来の1 slot / 1 copy group。unset/empty/`0` が既定。
- `128`, `256`, `512`, `1024` は `G=min(n_slots,cap)` groupsのopt-in候補。
- 各groupは `k=group_id; k<ctl[2]; k+=G` で全placed pageをcopyする。
- その他の整数capは明示entrypointでenqueue前に拒否する。`nq<=0` の従来no-opは維持する。
- 文字列parserは新規のhost-only header `kv_stream_copy_groups.hpp` に置く。環境へ書かず、SYCLも初期化しない。

追加entrypointは環境を読まない。fixtureのdirect発行とgraph captureは整数を渡すだけで、ABBA中にprocess環境を書き換えない。
通常callerは引き続きproduction wrapperを呼ぶ。起動前のenv固定はownerの契約であり、外部コードによる実行中の環境変更をこの候補が安全化するものではない。

## host staging とUSMの寿命

v1 `Buffer::put/get` のlocal vectorは、`q.memcpy`後のwait例外が関数外へ伝わると、完了を証明できないまま破棄され得る構造だった。
v2ではsubmitとevent waitを `pending_or_hold` の内側で行う。例外をその場で捕まえ、local vectorを持つ外側frameへ戻さない。

`owner_hold` はstderrへ一度 `INCOMPLETE owner_hold: ...` を出し、既存stdoutをflushした後、POSIX `pause()` でownerの終了指示まで止まる。
stack unwind、USM free、vector破棄、再発行、retry、reset、状態変更を行わない。
unknown completionを成功へ変えない。Mainの有限deadlineと既存ownerが、このprocessをincompleteとして記録し停止・reapする必要がある。
自発的な正常終了でも安全回復の認定でもない。Linux用component fixtureの機構であり、製品のerror/lifetime処理を変更しない。

同じ保持方針を次へ適用する。

- `Buffer::put/get`: local staging/readback vectorを保持したまま、memcpy submitまたはchecked event wait例外で停止。
- timed direct/graph発行とmarker/event境界: 例外時にcallerへunwindせず、map/pool/input/graph ownerを保持。
- graph correctness replay: submitと完了確認を保持scope内で行う。
- Buffer破棄前: external queueとruntime compute/transfer queuesをchecked drainする。失敗時はAllocationのdestructorへ進まない。
- `MemoryKind::Host` sourceの直接CPU初期化前: external queueとruntime queuesをchecked drainしてから前後guardと全payloadを書く。初期化直後にguardも確認する。
- sourceのCPU読取前: 同じchecked drain後にguardと全payloadを確認する。

`pending_or_hold` は時刻を待つだけで完了を仮定する処理ではない。エラーの後もvector/USM所有者は同じframeで生存し、outer ownerの有限期限だけが停止を管理する。

既存 `kv_stream_counters` はそのまま。source上は内部stack配列を宛先にmemcpyしwaitしており、wait例外時の内部配列寿命をfixture側の外側catchだけで保持できない。
そのためv2 fixtureはこのaccessorを呼ばず、保護された `Buffer::get` でctlを読んで同じu64 counters/overflowを照合する。
accessorの通常caller契約（external queueを先に完了、runtime device0 context）は維持する。
この既存error経路が実際に故障したとは報告しておらず、製品側での変更・検証も本proposalの範囲外。

## 共通source、caller、制約

root `CMakeLists.txt` のSYCL分岐は `cmake/sycl_backend.cmake` をincludeする。
その `strata_kernels` に実際の `src/kernels/sycl/kv_stream.cpp` が入り、engine/prefill/strataへlinkされる。
CUDA root generatorからArc機能を推定していない。

| caller | 維持する契約 |
| --- | --- |
| `src/core/layer.cpp` `qsa_kv_resolve` | mode1だけproduction wrapperへ渡す。map/pools/host/fmtは同じstate。 |
| 同fileの単token QSA | nq1、selection後、同streamのattention/gather前。 |
| `src/core/verify.cpp` verifier QSA | nq=`n`、tb offset付きIDs/steps、同cs上でbatch attention前にunionをresolve。 |
| mode0 / mode2 ring | wrapperへ到達せず従来動作。ring restoreとprompt stagingも対象外。 |
| component | 明示cap entrypointをdirectまたはnon-native graphで使う。同じaddress、validation、single_task、copy実装。固定envのwrapperとも全state/byte oracleで比較する。 |

in-order queue内のresolver → copy → attentionの依存、nonnull streamで非同期return、null streamで既存finishの完了待ちを維持する。
製品に新queue、wait、device stateのhost read、flag reset、memory owner変更を足さない。
componentに追加する保護境界は製品の計時やserviceには入らない。

`0<=placed<=n_slots` の全範囲に対し、各kは一意の `group=k mod G` に割り当たる。
Gを超えるmiss数、full slots、非連続duplicate、任意valid IDを切り捨てない。
`ceil(width/page_size)` をgeneric IDのpage数上限へ使わない。
overflow時は従来のsticky flagと置けたsubsetを保持し、全selectionの成功とは扱わない。

FP16のK,V、int8のK,Vと各scale、Q4_0のscaleを含むpacked K,Vのcopy spanは同じ。
real shapeで1 pageの論理payloadは8192/4224/2304 B。
hybrid K8V4 streamingの既存拒否は維持する。append量子化・回転・host正本への書込・reject/commitは変更しない。
論理262144 cells、任意の割当済みpage、partial pageの全copy、host backing容量も変更しない。

## fixtureとMainの後続候補

CMakeにはEXCLUDE_FROM_ALLの二targetだけを追加し、自動CTest実行は追加しない。

`kv_copy_groups_host_fixture` はSYCL/runtimeなし。pure parserへnullptr/空文字/0、全valid cap、無効文字列と無効整数を渡す。
実process環境は変更しない。その上でv1の独立oracle/全byte/coverage fixturesを維持する。
全miss0..513、cap周辺、full8192、duplicates、eviction、hit保護、epoch/u64 carry/overflow、任意IDとpartial pageを含む。

`sycl_kv_copy_groups_bench --host-env` はactual envの文字列をread-onlyにparseして、Runtimeを作らず終了する。
Mainが必要ならunset/empty/invalid値を**別processの起動前env**に入れて、このmodeを実行する。
無効値はRuntime生成より前に拒否する。fixture自身は値を変えない。

`--validate` / `--validate-exhaustive` は全format、17slotsと1031slots、8192slots/65536 logical blocksの実kernelをoracleに照合する。
miss0/small1..513、128/256/512/1024境界と上側、full slots、invalid dimensions/aliases、sticky overflow、二partial blocks、262143を含む。
synthetic T4は各width2051、union2052 pagesであり、actual verifierのhit率・miss分布を観測したものではない。
graph fixtureは同じUSM addressでIDs/width/ctlを変え、miss513 → hit0 → 広いselection → slots全幅をreplayする。
環境値は全期間固定で、geometryはcapture時の明示整数から決まる。

通常validationの一部で、process起動前capに対応するproduction wrapperと明示entrypointを同じoracleへ照合する。
wrapperの全cap確認が必要ならMainがcapごとに別processを起動する。単process内で環境変更して代用しない。

Mainがidleになりreview/apply/buildを管理する段階の候補:

```sh
cmake --build "$MAIN_BUILD" --target kv_copy_groups_host_fixture sycl_kv_copy_groups_bench
"$MAIN_BUILD/kv_copy_groups_host_fixture"
"$MAIN_BUILD/sycl_kv_copy_groups_bench" --host-env
"$MAIN_BUILD/sycl_kv_copy_groups_bench" --validate
"$MAIN_BUILD/sycl_kv_copy_groups_bench" --validate-exhaustive
"$MAIN_BUILD/sycl_kv_copy_groups_bench" --graph-benchmark 128 40
"$MAIN_BUILD/sycl_kv_copy_groups_bench" --graph-benchmark 256 40
"$MAIN_BUILD/sycl_kv_copy_groups_bench" --graph-benchmark 512 40
"$MAIN_BUILD/sycl_kv_copy_groups_bench" --graph-benchmark 1024 40
```

未実行のコマンド例。build/test/GPUはMainが直列化する。
`--benchmark` は明示entrypointのdirect発行、`--graph-benchmark` は同じ実装でcaptureしたgraphのreplay。
各40標本はA0/Bcap/Bcap/A0を10回、各arm20標本。同じRuntime/queue/USMのまま整数引数だけを選ぶ。
40..80の4倍数だけ受付。各標本の初期map/poolを復元し、capture/初期化/warmup/oracleは計時外。
環境parserのhost overheadは明示entrypointのdirect計時には含まれず、production wrapperの未変更binaryとのdefault退行は別gateである。

`--event-benchmark` / `--graph-event-benchmark` は別processのmarker付き診断。
`host_operation_submit_ns`、`host_marker_submit_ns`、`host_whole_queue_ns`、`device_marker_envelope_ns` を分離する。
device envelopeはresolver + copy + queue gapで、copy単独時間でもGPU active timeでもない。
host/device absolute clockをjoinせず、marker付き値とclean値を混ぜない。

全個別JSONLにphase/directまたはgraph、format/shape、各width、source上のgroup/workitem形状、
実ctl readbackと一致したplaced、logical host-read payload、全counter/overflow、exact oracle fingerprintを残す。
全hitのpayloadは0、帯域列は全caseでnull。logical byteをDMA/PCIe wire byteと表示しない。
`lookups` はpage run参照数であり、任意の非連続duplicateは再計上される。lookups-missesからhit数を推定しない。
overflow caseはcorrectness結果として保持し、速度benchmarkはoverflowを拒否する。

## frozen env、owner、保持

`pins.json` は完了済み `decode-format-existing-exact-normal-v1.json` のhashと次の実際の値を記録する。
fixtureのread-only `frozen_env` はRuntime生成前にこれらを確認する。

| key | Mainの既存値 |
| --- | --- |
| ONEAPI_DEVICE_SELECTOR | level_zero:gpu |
| NEOReadDebugKeys | 1 |
| EnableDirectSubmission | 0 |
| UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD | 1 |
| STRATA_SYCL_VERIFY_NATIVE_CAPTURE | 0 |

Mainは同receiptの全runtime/driver/validation/loader recipeを起動前に設定して維持し、compiler/AOT/source/diff/binary/deviceのhashとidentityを保存する。
この表だけで完全なenv recipeとは扱わない。process environの走査や書換えはfixtureに追加していない。
providerがmainより前から動く場合も、fixtureが環境を変更しない設計は同じ。

component専用の候補上限はprocess毎wall1800秒、RSS4GiB、text64MiB。
最大fixtureはhostsource512MiBを含めlive1.5GiB未満の設計で、model用88GiB等の予算とは別。
Mainの既存measurement lock/ownerが全工程を直列化し、fresh fault/dump/boot、期限、正常またはincomplete終了、owned PID/reapを記録する。
`owner_hold` で保持されるUSM/local vectors/graph/inputが現在のconsumerであり、ownerの終了・reap確認までcleanupしない。
保持のbyte上限は同じ4GiB RSS、review点はそのprocess停止と診断のclose。
owner_hold、期限超過、欠けた標本をPASSへ変えない。通常stdout PASSだけでfaultなしや正常reapを証明しない。

source/commands/environmentと全標本、exact差分/fault/owner/採否をcompact receiptへ保存する。
raw profilerを取る場合も別process・同じ有限budgetで、要約/hash後のretention判断はMainが行う。
本proposal作成では消したartifactはない。

C++/CMake compile、pure host cases、GPU/graph API互換性、exact state/byte、failstop故障注入、fault、性能は未検証。
7秒台のGPU境界host waitの原因を確定したものではない。
component改善後もnormal MTP4 actual decode、公式bench.shのprefill/decode/agent、model品質、物理256K、RESTOREは別gateであり、完了・高速化・採用とは主張しない。
