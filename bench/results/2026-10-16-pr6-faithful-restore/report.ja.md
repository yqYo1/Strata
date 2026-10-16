# PR #6 忠実復元と安全性監査

2026-10-16 JST。worker `task_00abc6a80d11` / dispatch `ctx_a9c4d296a850`。
作業先は `verify/pr6-prefill-restore`、開始commitは
`08874a80989c13f3fe4b8e13c085e69f6fdea5f2`。

**旧engineの実ソース一式を復元し、後続commitで具体的な寿命・失敗処理・範囲の欠陥を修正した。
全てsource-onlyで、build・compile・test・GPU・profilerは未実行。>600 tok/sは再測定していない。**
Mainのsource/build出力、default mirror、feature/sycl、Git設定、serviceには変更を加えていない。
push・PR・統合・子branch退役はMainが担当する。保護対象の外部recordは開いていない。
新規コード・fixtureはrepository LICENSE（MIT）に従う。XeStrataのsource・helper・test・
保存用派生proposalは実装参照にも復元元にもしていない。

## 復元の固定点とbuild経路

| 役割 | immutable revision |
| --- | --- |
| 当fork PR #6の実装 | `f263d87d44aa671d056c33e6c6b6b0635fc25b8c` |
| PR最終head / 復元の元tree | `d0c5dda0dc2e8d82458c4b0ff31c4cef44c40b32` |
| PR merge（履歴の参照） | `6151f24defba2b812a7c139393b4218b5436d051` |
| 今回の修正前復元commit | `efba8d47` |
| 比較する現行engine source | `63326f98c36656cf3a3e1adcc54c3c25a02aa6ee` |
| 403 tok/s測定のsource / binary | `6354f0b4ff3731466795410e11ca9ef785c9d326` / `85643f39…04ad191` |

PR #6は [個人forkのPR](https://github.com/yqYo1/Strata/pull/6)。唯一のupstreamはNiko1221/Strata。
`f263d87d` と `d0c5dda0` のengine sourceは同じで、変更はprofile helperのRAM PLE/startup記録。
最終headのhelper SHA256はtimed receiptの `ecd1fb9d…d4c42f` と一致する。

歴史のentryは `sycl/src/program/generate.cpp`、targetは `strata`。
旧 `sycl/CMakeLists.txt` の `strata_resolve` は `.cu` を同位置の `.dp.cpp` へ、
`.cpp` はoverlayがあればそれへ、なければ歴史rootへ解決する。
headerも `sycl/include` → 歴史 `include` の順で、現行root headerとの混在はしない。
今回 `validation/pr6-restore/` の内側に、この解決結果83 translation unit、
必要なproject header/内部include、CMake、generator、migration/fixup、profile helper、LICENSEを
**197ファイル・4,006,090 bytes**、元blobと同じバイト列・file modeで復元した。
CUDA/HIP engineのimportではない。`sycl/src/kernels/cuda/*.dp.cpp` は歴史のSYCL生成済みsource。
元のCUDA generatorをこのSYCL engineの機能実装とみなしていない。

- `historical-closure.json`: 全197ファイルの元blob/SHA256/mode、resolver各group、external header inventory。
  この一致契約は**修正前 `efba8d47`**に対するもの。修正後の4 engine/header fileは差分として別管理する。
- `current-source-pins.json`: 現行63326f98のresolver、target source、到達header等221ファイルのpin。
  optional vLLM objectを区別したstatic closureであり、configure結果やcompile-commandの代用ではない。
- `historical-evidence.json`: 元receiptへのrevision/path/blob/SHA256、六つの個別sample・command/env、
  input/timing/fixture identityとAOT記録のcompact抽出。元logやbuild/dependency/modelはコピーしていない。
- `fixed-source-pins.json`: 修正後file identityと移植helper/fixtureの出自。
- `manifest.json`: restore/fix commit、互換変更、検証statusと未了gateの入口。

外側の新規 `validation/pr6-restore/CMakeLists.txt` だけがbuild provisioningを変える。
内側の歴史CMakeは変更していない。Main-reviewed build gate、外部vendor include path、
既存offline llama.cpp必須、歴史parity target group OFFが唯一の互換差。
DPCT 46ファイルとroot vendored ggmlの3ファイルはtreeへコピーせず、外部snapshotを全49 SHA256で照合する。
外部llama.cppのpinは `3cf03257f219afbe7334045ff7c6a06ac68c627d`、
MainがHEAD・dirty diffと実libraryを照合する。新規dependency downloadはない。
元migration toolは出自を示すため保存したが、blob再現に再migrationは不要。

旧CMakeが指定するbuild契約はC++20、Release、`-fsycl`、global subgroup32、
`-fp-model=precise`、per-kernel device image split、JITのcorrectly-rounded divide/sqrt、
prefillのcompile/link `-qmkl=sequential`。AOTは `STRATA_SYCL_AOT=bmg-g21` と同丸め指定。
CPU expert TUのAVX-512/AVX2指定、Threads、ggml-cpu/base linkも元のまま。
これらは**sourceが指定するflag**で、今回の有効compile commandを実証したものではない。

## 歴史の測定事実と未解決の再現情報

主参照は `main-review-round559-560.ja.md`、歴史 `paired-normal-wall/summary.json` と
六つの元 `run.json`。pair 2は実行順が逆で、labelを入れ替えない。

| 実行順 | label | elapsed ms | tok/s |
| ---: | --- | ---: | ---: |
| 1 | baseline-p1 | 72286.6 | 453.3067 |
| 2 | layer-major-p1 | 51172.7 | 640.3414 |
| 3 | layer-major-p2 | 50688.0 | 646.4646 |
| 4 | baseline-p2 | 72288.7 | 453.2935 |
| 5 | baseline-p3 | 72288.7 | 453.2935 |
| 6 | layer-major-p3 | 50736.4 | 645.8480 |

中央値はbaseline **453.2935 tok/s / 72.2887 s**、layer-major
**645.8480 tok/s / 50.7364 s**、elapsed減29.8142%。これは歴史の実測で、今回の実行結果ではない。
Arc B570 10GB、Ryzen 5 5600X、installed RAM 128GB、native IQ3_S pack、
oneAPI 2026.1系と選択compute runtime 26.35.39758.10の記録。
JIT binary SHA256は `eae0f3ed824d13cf25033e7dc60d1f8e69bb770b0b764ea6600a9900969b7ed9`。

helperは `tokens[:n+1]` を渡す。`n=32768` はCLIが計時するconditioning/batched token数で、
入力file自体は**32769 token（最後の1 tokenをheld-outとしてdecode側へ）**。
input SHA256は `7050f0ae22c043073278ac617ddbdb52e7bcfedb0880ff0f10a44e059596cf94`、
65,537-token source streamのSHA256は `aea4383be22560c13bcdf3653215e1c6c4f1bbb24c472d4667b7fdc03df597a0`。
source excerpt/tokenizerのhashはfixture manifestにある。chunk8192×4、context65538、K8V8全量resident、
expert-cache128、no-prefill-borrow、pool-workers5、pcie-frac0、adapt-swaps0、direct PLE、
max-new1、greedy、suffix-draft0、MTPなし。共通envはFIRST0、RING8、ATTN_BATCH128、
ATTN_LAYOUT4、TOPK_TUNED1、GDN_KEYHEAD0/KEYHEAD_TUNED1。比較は
COMPACT0/LAYER_MAJOR0/R_GPU0 と COMPACT2/LAYER_MAJOR1/R_GPU32768を同時に切り替える。
compactだけ、schedulerだけの効果として29.8%を帰属させない。

CLI timerはinit/model loadの後、`Prefill::run` の直前から、必要なlent-cache refill後まで。
今回はborrowなしなのでrefillなし。private layer-cache/GPU residualのalloc、各層upload、
PLE gather、residual copy、callback/retirementはrun内で計時される。head dump、
最後のheld-out token、生成token、モデル読込、RAM PLE startupはそのprefill値の外。
process wall/TTFT/phase markersの時間をこの値へ混ぜない。
元六receiptは248320個のfinite first head hashと生成ID `[63]` が一致。
full residual/persistent-state equalityは別の257/4096/8087等の歴史qualificationで、
六つのtimed runに毎回state dumpが含まれていたという意味ではない。

PLE tableはpaired run前に全体がtouch済み、embedding service停止のquiet条件。
同条件系のAOT direct646.90、RAM652.16は単発で、反復した追加gainの根拠ではない。
以前の64K/chunk8K約417 tok/sは別のtransfer-timed/warmth条件で維持する。
262146 context・chunk1024の歴史capacity成功は**入力2 token**のallocation確認であり、
physical262144全prefix・RESTORE・速度の証明ではない。

未解決: JIT binaryの全compile database/cache/toolchain executable hash/完全linked-library inventoryは
六receiptにはない。selected engine sourceのhashは一致するが、全build inputからbyte-identical binaryを
作れるとはまだ証明していない。AOTのbuild receiptをJITの完全build証明へ転用しない。
旧warm PLEのphysical SSD trafficと待ち時間変化の原因も分離されていない。
原token fileは今回取得・再生成していないので、Mainの再現では既知hash一致を必要条件にする。
古い記録を読み直したことはupstreamの新規checkではない。今回fetch/最新upstream確認は実施していない。

## 現行実装とのsource比較

この表の「現行」は63326f98のsource事実。2026-10-15初回reviewにあった
「rootにlayer-majorがない」はその当時のsourceについてで、**現在はrootの
`src/prefill/layer_major.inc` に実装され `prefill.cpp` からincludeされる**。
sourceの有無と、過去のcompile/run qualificationを分ける。

| 項目 | PR #6 / 今回の忠実route | 現行63326f98 |
| --- | --- | --- |
| resolver / entry | standalone sycl overlay+歴史root / sycl generate | root CMake→cmake/sycl_backend.cmake / root generate+dense_prefill_nll |
| 処理順 | =1で複数chunk、=2で単chunkも強制。layer→全chunk、各層全512expertを一度upload | =1かつn>chunkの長入力候補。既存run_implを一層ずつ使う点は同じだが独立したadmission/lease |
| scratch | COMPACT2がHCまでalias、embedding/mixed/outputを共用。8K元3,353,103,360B→1,895,413,760Bは歴史receipt値 | compact FP16 MoE、scratch-prefix alias、native16 viewを持つ別carve。reserved arenaをnative16化だけで全縮小はしない |
| residual | 全層間FP32、GPU完全chunk prefix+host tail。copy後wait。元RはFP32 | FP32 reference、実験FP16保存、native16 BF16 producer/consumer。dtype別plan、autoGPU placement、FP32 bridge |
| projection | BF16/FP16入力→FP32出力、native weight dequant→FP16、dpct→oneMKL | 既存oneMKL FP32出力とnative16 GEMM、optional vLLM Xe2 AOT object。FP32 local accumulate/router/recurrentと保存dtypeを区別 |
| QSA attention | XMXはSTRATA_PROMPT_ATTN_XMX明示時だけ。旧timed env+dispatchからlayout4 SG16 fallbackを推定 | prompt matrix default/capability dispatchとfallback layout1、Direct/Mapping等。現在profileはINT8 matrix launchを実device計測。QSA16別opt-in |
| GDN | gates/conv/recurrence kernelとKEYHEAD/TUNED env、FP32 state | root SYCL kernel群。native16でもrecurrent state/local reductionはFP32。単なる同名でkernel同一とはしない |
| PLE | layer1だけでtoken列からgather、double host buffer、direct/mmap/ram、host→GPU | canonical/native dense bankやnative16 bridgeを持つ。PLE/final callbackのFP32境界が残る |
| routing / expert算術 | GPU router、IDをhostへ渡しCPU count/prefix/row grouping、expert別dequant+GEMM | 同種host groupingが残る。並列dequant/native16/Dm16等の別候補がある。all-GPU arithmeticとcontrolの意味を分ける |
| expert upload | Stager worker→pinned/pageable ring→独立copy queue→私有全layer cache。layer publication前drain | owned whole-layer host slab、same queueを要件にしchecked upload/drain後map公開。layer-majorはStager jobを使わない |
| KV範囲 | layer-majorはmode0 residentのみ。streamed/ring拒否は保持 | mode0とmode1の完全logical host/stageをadmit。QSA16はresident INT8等へ限定 |
| decode cache / MTP | decodeのcache pointer/host mapをRestore。timedにはMTPなし。古いdraft/fallback graph群 | 継続所有・checked retirement、expert/headのrelease/restore/recapture等の別機能。旧版にそのfeatureをあることにしない |
| callback / RESTORE | 最後のlayer・最後のchunkだけcheckpoint-ready。PLE prevは各layer前に入口値へ戻しrun_impl末尾で更新 | final callback順検査とPLE prev公開、chunk cursor、guarded native16 callback等。checkpoint format/APIも別 |
| queue / context | dpct default in-order+copy queue、USMはdefault context、旧async handlerはログのみ | runtime/compatがcontext/alloc registry/sticky async error/checked cleanupを持つ。CUDA名のadapterでも実SYCLで、CUDAengine採用ではない |
| flag | global SG32/precise/qmkl/per-kernel split/丸め指定 | size_t SYCL id contract、target/source単位nonfast/ffp-contract、MKL dynamic/sequential。vLLM objectだけ別AOT image |
| workload / clock | CLI diagnostic 32768 batched / context65538 / output1 / MTPなし | official bench actual39262 input /39261 batch、context65536/chunk8192、thinking ON/output24/MTP4等。HTTP/backend clockとTTFTも別 |

現行の403.0は `own-resident39k-summary-v1.json` にある3つの独立processの
402.6/403.0/403.4 tok/s。BF16/vLLM/QSA16=0、全39261 residual rowsがGPU、source6354f0b4。
`own-current-device-profile-summary-v1.json` のAttention **33.965989345 s** は
別のfixed32768-residual診断で、load・warmup・24出力・teardownを含むdevice duration集計。
critical pathやGPU busy unionではなく、403のall-GPU配置と同じprofileでもない。
softmax qualificationのsource63326f98はlow-level fixture合格のMain記録であり、
この復元の実行やwhole-model品質/full256K/RESTOREの合格ではない。

仮説: attention route、GEMM/backend/shape、scratch・precision、host staging・queue、
入力長・context・MTP・PLE warmth・compiler/runtimeが速度差へ寄与し得る。
**現行403と歴史645.85はmatched regressionの証明ではなく、原因の比率は未解決。**
単一CPU coreがbusyというだけでCPU算術がbottleneckとも言わない。
old/newともoneMKL GPU queueだけを根拠に「全算術がGPU」とは断定せず、Mainがactual device kernelと
CPU library/fallback、wait内thread CPUとhost computation/submissionを分けて評価する。


| 修正 | commit |
| --- | --- |
| A | `aceb6b0f` |
| B | `47fb8e05` |
| C | `9e770e33` |
| D | `edf7cd79` |
| E | `1ecced66` |
| F | `00ae5188` |
| G | `4338c60f` |

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

## 修正G: privateなopaque typeをfree helperが名指しするaccess違反

歴史 `include/strata/prefill/prefill.hpp:163,184` は `Prefill::Impl` をprivateに宣言する一方、
`sycl/src/prefill/prefill.cpp:696-731` の通常free function `compact_take`、`take_compact_hc`、
`take_compact_base` はそのtypeとmember pointerを直接名指しする。friend宣言もない。
これはstandard C++のaccess規則に反する。歴史sourceを復元したことと、保存されていない
完全なJIT flagsでどうcompileされたかは別問題であり、当時の成功binaryを否定はしない。
`-fno-access-control` 等の有無も未解決で、今回そのflagを追加して隠さない。

opaqueな `struct Impl;` の前方宣言だけをpublicへ移し、definitionとunique ownerは隠したままにした。
object layout、symbol、kernel、allocサイズ/順、precision、ABI上のdata配置は変えない。
sourceのaccess問題を解消する最小互換修正であり、実compileはMainのgateに残る。

## 寿命・state監査の残る限界

修正は歴史routeのkernel math、precision、layer/chunk順、原env selector、正常時のupload/wait数を
維持する範囲に限定した。new guardはerror時にsource/caller ownerより先にdrainする。
現行のnative16、bank、same-queue scheduling、MTP expert release等を混ぜて数字を追ってはいない。
short-inputの旧chunk-major/decode weight/cache経路も残る。このbranchをfeature/syclへ丸ごと
取り込むことは提案しない。

| 対象 | sourceで確認したこと | 未了 / Mainの確認 |
| --- | --- | --- |
| layer expert cache | cache・residency・rows・timerはRestoreより先に宣言され、通常のreverse destructionではdrain→pointer復帰→free。各layer upload完了後にresident map公開、次layer上書き前もdrain | native queue failure、allocator failure、partial enqueueの実注入。free/destroyがthrowする場合のcontextとowner保持 |
| host staging | blobはissuerでresolve。transient sourceはcopy_blob、stager bufferはissued+done_seqで返却。512-ringのGGUF sourceに対し1layer512だけをresolveしてからdrain | 同時adapt/blob mutationなしの排他契約が必要。arbitrary ExpertSourceをstableだと仮定しない。source/copy worker例外と停止時の所有を実確認 |
| completion sequence | signal未受理のwant公開は修正B | 既存Stager/PLEのhost volatile uint64 readとGPU fillの可視性/atomic性は標準portableな証明がない。USM aspect・system atomics・受理後device failure・無期限pollの資格付けは未了。今回この仕組みを安全だと認定しない |
| async errors | wait_and_throwはhandlerを呼ぶが、歴史DPCT `device.hpp:69-79` は例外を表示してreturnする。create_queue(true)もdefault queueもこのhandler | waitの正常returnを数値成功へ昇格させない。Mainはprovider errorを含む失敗注入でsticky/fail-stop契約を先に確定する。必要なら別の明示patch。今回external DPCT改変はしない |
| graph / expert owner | layer-major自身はgraphをcaptureしない。MTP destructorは自身queueをwaitしてからexec/arenaを破棄。prefill_dev graphは固定tok/step/pos/Rinを参照し、pf_devはgraph外copyのsource | 一般のCapturedGraph::resetやraw session/token graph freeはdelete中心。CUDA時代の「exec破棄がwait」というcommentをSYCLの保証として使わない。実callerのidle/last event、capture失敗解除、複数queue、timeout、再captureをMainが確認。観測なしにUAFや過去fault原因とは断定しない |
| decode-cache / MTP再利用 | 歴史prefillがdecode cacheへ書く代わりにprivate cacheを使い、元pointer/callbackへ戻す。借用cache refillには元から256blob毎のdrainがある。MTP正常prefill/fallbackは完了後に戻り、修正A/C/Fは失敗時もleaseを退役 | current expert release/restore/recaptureは歴史featureではない。現在のgraph/bind stateをこのengineへ持ち込まない。再bind・prefill容量増・decode継続・warm serve・RESTOREで別途検証 |
| bounds / pages | 修正Eでpublic入力の位置/サイズ上限、layer-major resident capacityを事前確認。元router host countにもID範囲checkがある | device page-table実値、pooled indexer tail/dead/capacity、page末尾とchunk境界、physical262144の実tokenを要検証。host fixtureはGPUを保証しない |
| checkpoint / callbacks | layer毎にPLE入口の2tokenを戻し、各layerのGDN/QSA/PLE stateはtoken順で1回だけ進む。外部chunk callbackは最終layerだけ、checkpoint-readyは最後chunkだけ。標準serveはdraft処理後on_chunkでcheckpoint | on_stage→on_chunkの元順は維持。ss.ple_prevはrun_impl末尾に更新される一方、checkpoint restoreは保存idsからprevを導出する。custom callbackの観測・draft state・checkpoint payloadを混同しない。failed/cancelled途中stateはpointer復帰でrollbackされず、再利用せずdiscard/restoreが必要 |
| ABI / dependencies | 元headers/DPCT型/queue pointer/graph型は歴史source同士で閉じる。外部vendorはhash照合 | Intel icpx/libsycl/UR/oneMKLの同一stackでまずcompile。現在LLVM22/libsycl9へ古いUR adapterを混ぜない。host scalar alpha/beta等のlibrary内部stage、BF16/FP16 emulation/fallbackはactual libraryとdevice/CPU証拠が必要 |

[SYCL queue](https://github.khronos.org/SYCL_Reference/iface/queue.html) と
[USM allocation/free](https://github.khronos.org/SYCL_Reference/iface/usm_allocations.html) のAPI契約を
2026-10-16に確認した。freeは利用commandの完了を代行しない。
Intelの[experimental graph仕様](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc)
はAPI契約の照合だけに使い、implementationは採用していない。mutable online仕様が当時の
installed runtimeの実装を同定するものではない。旧faultとPR #6の因果は未証明のまま保持する。

## Main所有のbuild・診断・数値・性能計画（未実行）

これは実行許可の依頼や自動run scriptではない。Mainが既存のserialized実行枠でsource/所有権を
reviewした後だけ進める。特にcompletion polling/async error/graph callerの未資格事項を解決せず、
歴史unsafe anchorを起動しない。

1. **sourceとdependency gate。** restore commitと修正後HEADを別にpinし、全diffをreview（修正Gのaccess解決も確認）。
   original closureはefba8d47のblobと照合。新規helper/fixtureの出自はfixed-source-pinsを照合する。
   49vendor fileをGit旧objectからGit外へmaterializeしhash確認、llama.cppのHEAD/diff/LICENSE、
   compiler binary/version/hash・libsycl/UR/Level Zero/oneMKLの実libraryを記録する。
   旧binaryと同じSHAになるとは要求も仮定もしない。byte-exact reproductionの未証明部分は維持する。
2. **host fixtures。** `validation/pr6-restore/fixtures` を独立configure/buildし、
   pr6_draft_uploadsとpr6_range_guardを実行。deferred read、partial/full/256K、early-return/throw、
   failed-drain fail-stopとrange/size overflowを確認する。ASan/UBSan等は別owned processで必要範囲だけ。
   これがpassしてもSYCL/libraryの寿命は別gate。
3. **native build。** 下記recipeはMain向けに示しただけで、このworkerはconfigureもしていない。
   build出力は独立したGit外dirに置く。最初は歴史Intel 2026.1系の一貫したstackでJIT、
   原 `strata` targetのみ。compile_commands/CMakeCache、per-TU flag、link map/library、全source/envと
   binary SHA256をcompactに保存する。AOT bmg-g21は別arm・別build。
4. **小規模診断。** model-free allocation/copy/graph fixture→2/257/4096/8087のodd tailへ。
   UR tracing/parameter validationはexec前に固定し、sampleごとに新native process。
   `UR_ENABLE_LAYERS=UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION`、
   `UR_LOG_TRACING='level:info;flush:info;output:stderr'` を既知のrecipeで使う。
   DMA受理後signal enqueue失敗、copy failure、capture failure、callback false/throw、cancel、
   null-return、partial-initを注入する。未知のcompletionでfree/returnせずfail-stopすることを検証。
   特にMTP2batch以上・FP16主経路とE-4/per-group fallbackの別queueを検査する。
5. **数値とstate。** 同じ歴史mathのbaseline/compact/layer-majorでfirst full logits、全residual、
   authoritative KV/scales、GDN recurrence+conv、indexer pooled/tail/dead/block_pos、PLE historyを比較。
   257/chunk128、4096/chunk4096、8087/chunk4096、FIRST256/0、host/mixed/allGPU residualを含める。
   旧safety-only差分ではbyte equalityを有用な診断にするが、modern BF16/FP16実装に対して
   自然言語生成やfloating outputの完全一致は要求しない。独立FP32/FP64 referenceと既存の妥当な
   数値許容値・finite/guard/input不変を保持し、held-out loss/意味のあるcoding・long-context品質を別に評価。
6. **履歴条件の反復。** fixture/input hash、model/pack/profile、K8V8/context65538/chunk8192、
   MTPなし/output1、両arm共通env、原helper SHAを合わせる。PLE startup/初回touchとprefill clockを分ける。
   三組をB,L / L,B / B,Lの順でfresh native processとして実行し、raw sample・失敗・faultも保存。
   比較はCOMPACT0+chunk-majorとCOMPACT2+layer-major+R_GPU32768の**組合せ**。
   原32Kとは32769入力/32768batchである。過去head/tokenを現在の別model条件へ強制しない。
7. **matched official bench。** 同一共有server/client、tokenizer/model、固定workload-id、thinking ON、
   no-reuse、actual input/read>=32768（65536も可）、同output cap/MTP/context/KV/chunk/PLE warmthで比較。
   `bench.sh --prefill-only --prefill-tokens 32768 --think --runs 3 --workload-id pr6_restore_v1`
   を各armのMain管理localhost serverへ `--url` で使う。serverのprivate configは `exe` だけ
   reviewed candidateへ差替え、共有APIが使うCLI flags/protocolをsourceで確認してから起動する。
   old engineにない新CLI機能が必要ならcompatibilityを別patch/別gateにし、黙ってflagを落とさない。
   最小32K要求をactual39262等へ膨らませるclientならその実数とrequest/token hashを両armに保持。
   3sample/arm以上を独立server/native processでも反復し、同じwarmup、順序、単一所有枠を使う。
   backend prompt clock、HTTP TTFT、process startup、decodeを分離する。
8. **physical262144・RESTORE・decode。** 実262144入力をpage/chunk境界のfixtureで処理し、
   最終cellのread/writeと必要なgeneration reserveを確認。resident-only歴史routeのVRAM admissionを
   先に計算し、過去のchunk1024 allocation-only結果を完走passへ昇格しない。全FP32残差を
   全262144行GPUへ載せると残差だけで10GiBなので無理に設定せず、host tail/小chunkと別条件として記録。
   session export→native/server終了→fresh process restore→continuation、prefix rewind/分岐、
   warm serve checkpoint reuse、MTP fallback/batch/graph recapture・cache refillを確認する。
   途中cancel/failed checkpointを成功sessionとして再利用しない。cold/warm/restored decodeの反復を
   独立に測り、24tokenのthinking出力やmax-new1をdecode性能benchmarkにしない。
9. **判定と保持。** 全pathのnumerical/state/lifetime/fault gateとmatched32K反復を満たしたときだけ
   >600 milestoneを再認定し1000へ進む。physical/RESTOREの未了を別に表示する。
   各runにboot/owned PID-start/deadline/exit/cleanup/journal境界とfault countを記録。
   raw log budgetは通常64MiB、distinct failure代表は通常256KiB以内。元FAILを改変せず、閉じた成功trace・
   重複timeline・不要tensorは有用な証拠を抽出/hash確認後にMainがserialized cleanupし、削除manifestを残す。

Main向けconfigure例。`PR6_*` はMainが選定・記録する実pathであり、このworkerでは設定も実行もしていない。

```sh
cmake -S validation/pr6-restore -B "$PR6_BUILD" -G Ninja \
  -DPR6_MAIN_REVIEWED_BUILD=ON -DPR6_VENDOR_ROOT="$PR6_VENDOR_ROOT" \
  -DSTRATA_GGML_DIR="$PR6_GGML_DIR" -DCMAKE_CXX_COMPILER="$PR6_CXX" \
  -DCMAKE_C_COMPILER="$PR6_CC" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DSTRATA_SYCL_AOT= \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_SYCL_PARITY=OFF
cmake --build "$PR6_BUILD" --target strata --parallel 2
```

## このworkerが実施した確認

Git object/sourceの読取り、source closureの静的解決とhash採録、diffの全変更箇所review、
`git diff --check`だけ。fixture・build・native executable・GPU/device query・process inspectionは未実行。
初回復元のcached diffには歴史由来trailing whitespaceが6件あり、blob一致を守って変更しなかった。
修正patchのdiff checkには新たなwhitespace問題を加えていない。
今回のsource案の採否・build可能性の実証・速度・quality・full256K・RESTORE・decodeはMainの実行gateに残る。

最終static identity採録: anchorの197/197 blobと197/197 modeが元treeに一致。
実装commit f263d87dとの差はprofile helperだけ（196/197一致）。
anchor後の歴史変更はengine/header 4ファイル、追加helper/fixture 5ファイル。
Main checkoutや通常root buildのsourceは変更していない。
