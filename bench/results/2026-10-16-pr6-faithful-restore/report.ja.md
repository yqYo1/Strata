# PR #6 忠実復元と安全性監査

この最初のコミットは source-only unsafe anchor。歴史ソース197ファイルをそのまま復元した。
build・test・GPU実行は未実施、>600 tok/sは再測定していない。
復元後の監査・修正・現行比較・Main検証計画を後続コミットで追記する。

元ソースとdependencyの全blob/SHA256、resolver選択は `historical-closure.json`。
外側CMakeの互換変更とcommit pinは `manifest.json` と `validation/pr6-restore/README.md`。

初回 `git diff --cached --check` は歴史ソース由来の trailing whitespace 6件を検出した。
blob一致を優先し、元バイト列を変更していない。新規wrapper/記録には該当なし。

## 修正A: draft upload のhost source寿命

歴史 `prefill.cpp:1321-1343` はループ内の `bh[4]` と次の反復で書き換える
`tk` を非同期memcpyへ渡し、完了waitはループの後だけだった。複数batchでtokenを
早く上書きし、boundsは単一batchでもscope終了後に読む可能性がある。in-order queueは
CPU側の書換え・scope終了を待たない。歴史の timed >600 tok/s はMTPなしのため、
この欠陥と当時のGPU faultの関係は未証明。

63326f98の `include/strata/prefill/draft_uploads.hpp` と `tests/cpu/draft_uploads.cpp` を
明示的に移植した（独立した当forkの修正、MIT repository LICENSE）。全batchのsourceを
immutableにし、正常時は元々の最終waitだけ、早期return/例外ではownerより先にdrain。
drainがthrowすればsourceを解放せずnative processを終了する。kernels、精度、batch順、
layer-major schedulerは変更しない。fixtureはdeferred read、partial/full/262144、
early return/exception、failed-drain fail-stopを対象にする。すべて未実行。
歴史DPCT async handlerが例外を表示して握り潰す問題は本修正で解消したとは扱わない。

## 修正B: 未発行のstager完了番号を待つ失敗経路

歴史 `Stager::issued_one` は `want[slot]=s` を先に書き、続く `copy->fill` の
受理後に `issued` をreleaseしていた。DMAが受理されてsignal fillだけがthrowすると、
RestoreがDMAをdrainしてfinishの `issued=1<<30` でworkerを進めても、そのworkerは
未発行の `done_seq>=s` を待ち続ける。少なくとも同じslotへ次のjobがclaim済みの
ring wrap時に成立する。finishはそのworkerのactive==0を待つため正常な解放に到達しない。

signal commandの受理後にだけwantをpublishするよう順を変えた。失敗時の旧wantは
前回の受理済みsignalだけを指し、既存Restoreのcopy drain→finishの順でhost bufferを
安全に返せる。通常時のGPU command列・コピー数・並列staging・kernel・精度は同じ。
`Restore` はwaitがthrowした場合、ownerを解放せず明示的なfail-stopにした。旧destructorも
暗黙noexceptでterminateするため、これを「旧版が必ずUAFする」とは主張しない。

これはsource-proved failure-progress bugで、過去に当該fill失敗が観測されたという
測定ではない。Mainはring2等の小規模fixtureでDMA受理後/signal enqueue失敗と
早期cancelを注入し、worker終了・コピー元不変・owner存続を別途検証する。未実行。
DPCTの非同期error握り潰し、無期限のdevice wait/pollは残る制約。

## 修正C: MTP fallbackのrecord upload早期解放

歴史 `MtpDrafter::prefill` の非per-group-sync経路はlocal `rec` を非同期uploadし、
後続のcapture/copy/graph submissionがfalseになればfinal waitを通らずreturnする。
`rec` はそのreturnで破棄されるがMTPの別queueは未完了の可能性があり、
Prefillのcompute/copy drainはこのqueueをretireしない。drafter destructorのwaitは遅すぎる。

現行63326f98 `src/core/mtp.cpp:1212-1253` の独立したrecord lease修正の意味を
歴史SYCL APIに適合し、修正Aのguardを `rec` の後に作った。upload前にarmし、
正常時は既存final waitでretireする。early return/throw時だけ先にMTP queueをdrain、
完了が不明ならowner存続中にfail-stopする。E-4のgroup順、固定graph引数、精度、
通常時のwait回数、`pf_dev_` 再利用方針は変えない。主経路の `draft_kv` が断る
構成でもこのfallbackは使われるため、修正Aとは別の必須修正。未実行。

## 修正D: allocation失敗とpartial-init cleanup

歴史の `DPCT_CHECK_ERROR(p=sycl::malloc_...)==0` はthrowがなければ0で、
`nullptr` も成功にしていた。新しいprivate layer-cacheの `open_sized`、
prefill Alloc、stager pinned buffer、PLE host rows、group host buffer、token buffer、
hand-off、streamed identity table、MTP mapped allocation/record bufferに実在する。
nullのままmemset/memcpyまたはhost writeへ進むか、必要なpageable fallbackを飛ばす。
戻り値の非nullも検査し、元のfail/fallbackへ入れる。正常allocationで処理順・計算は同じ。

Stager::initでvector初期化が途中でbad_allocとなる場合、旧destructorは常に
kRing個の `dma_done[i]` と `buf[i]/pinned[i]` を読んでいた。実際に構築された
要素数だけをcleanupするよう変更した。これはallocation失敗下のbounds UBの修正である。

[SYCL USM allocation仕様](https://github.khronos.org/SYCL_Reference/iface/usm_allocations.html)
は資源不足時のnull返却を定義する。Mainは各null-returnとvector部分初期化例外を
注入し、false/fallback/cleanupと再initを確認する。全て未実行。歴史engine全体の
allocation呼出しを機械的に書換えた修正ではなく、上記箇所が対象。

## 修正E: prefillの入力/physical範囲とqueue契約

旧public `Prefill::run` は未初期化 `impl_->T` を使い、負数/overflowや
`pos0+n > ss.max_cells` を検査せずbatched KV append、RoPE、steps生成へ渡していた。
`run_impl` にも入口range checkはなく、単一tokenのqsa_layer検査はこのbatched経路を
保護しない。pos+nを作る前に減算で検証する。actual prefixの上限262144とsession容量を
別々に守り、歴史の262146 allocation reserveを理由なく禁止しない。int32入力へ
表せないtokenはcast前に拒否し、host/device残差のサイズ積をcheckedにした。

旧layer-majorはin-order前提をcommentに持つがpublic initのstream引数を確認しない。
out-of-order queueならH2D→kernelに依存がなく、別contextならdefault queueで作った
USMへアクセスできない。layer-major入口で2queueのin-orderと同allocation context/device、
固定workspace geometry、resident page table/poolの範囲を検査する。既存正常recipeの
command順・mathは変えない。page table内のdevice page IDの実値はこのhost検査で
検証したことにせず、session initializerのidentity-table契約とMainの実page fixtureが必要。

`fixtures/range_guard.cpp` は65538条件、continuationがちょうど262144で終わる境界、
1超過、INT64_MAXによる加算overflow、負数、size積overflowを含む。新規helper/fixtureは
この歴史sourceの契約から独立に作成したrepository LICENSEのコード。未実行。
[SYCL queue仕様](https://github.khronos.org/SYCL_Reference/iface/queue.html) の
in-order保証は同じqueueへ順にsubmitしたcommand間のもので、CPU書換えを同期しない。

## 修正F: MTP別queueが読む借用residualの早期解放

MTPのper-group-sync fallbackでも、`R_rows`→`Rin_` のcopy受理後にgraph launchが
失敗すればそのgroupのwaitを通らずreturnする。MTPは `cs_`、caller Prefillは `m.cs` を
使う。standalone generatorの宣言順はMTPが3143、Prefillが7370なので失敗return時に
Prefillが先に破棄される。MTP destructorが後でwaitしてもPrefillのworkspaceを
読み終わる前にfreeし得る。これはlocal recとは別の、借用device ownerの欠陥。

per-group分岐にもcopy前arm/return前drainのguardを入れ、正常時は既存group waitの
後にretireする。kernel・graph・batch順・正常wait回数は変わらない。修正Cの
record guardは非per-group分岐でこの借用leaseも一緒にretireする。
queue admissionもcopy deviceまで一致を確認するよう補完した。全て未実行。
