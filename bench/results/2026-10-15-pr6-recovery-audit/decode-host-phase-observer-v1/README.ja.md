# ordinary SYCL G1 decode host phase observer 実装提案 v1

2026-10-15 JST。**提案のみ・未適用・未build・未test・未GPU実行**。
唯一の書込先はこの新規ディレクトリで、製品/CMake/既存fixtureは変更していない。
Mainの同一92acf71e binaryによるDirect0/1公式full thinking各3 runsの双方が閉じるまで、
適用/build/model/profilerを始めない。この提案には速度改善部品を含めない。

`source.diff`は現行5ファイルの変更と新規header1個の適用候補、`files/`は適用後に必要な全ソースである。
`source-hashes.json`は各現行入力と候補のSHA256、`artifact-hashes.json`はこの提案全体の台帳。
`generate_proposal.py`は候補をこのディレクトリに再生成するだけで製品を編集しない。
`cpu_contract_fixture.cpp`は新規header単独のCPU契約fixture案で、実行していない。
source.diffにはCMakeやfixture登録を混ぜず、Mainが独立コンパイルする。

## 対象とOFF経路

新envはexec時固定の`STRATA_DECODE_HOST_PHASES=1`だけ。実装は`getenv`読取のみ。
既存`STRATA_DECODE_ROUTE_LEDGER=1`と`STRATA_DECODE_SERVICE_CPU=1`が必要で、
そのeligible ordinary native SYCL/Linux・single stage・MTP capacity4・非split requestだけに
同じsequenceで所有させる。両既存envの片方がOFFなら`joined_required`のcomplete0診断、
joined側がineligibleなら`joined_ineligible`、state/output確保失敗ならcomplete0とする。
新envが未設定/0/他値なら追加allocation/clock/outputはゼロ。固定小型RAIIのnull判定は残る。
既存joined/service/その他診断を別にONにした場合の既存allocationや時計とは区別する。

`DecodeServiceBudgetScope`が新stateを所有する。`generate.cpp`はその直後にVerifierへの
`DecodeHostPhaseBorrow`を作り、通常close直前に明示detach、全early return/exceptionでは
逆順destructorで先にdetachする。旧pointerの復元はしない。衝突は両observerをinvalidにし、
Verifierの旧slotをクリアして新borrowを拒否する。次request/RESTORE/プロンプト処理には
request内pointerを残さない。captureの外側をissuerで括るだけで、新state/pointerをgraph、
callback、workerへ渡さない。

新stateのwriterは`Verifier::run`を呼ぶordinary issuerだけ。pool worker、DMA host_task、
`publish_plan`/`launch_work`からは新observerを触らない。既存native pool例外のfail-stopと
全job/format/dispatch owner寿命をそのまま保つ。pool completionを証明できない場合も既存の
abortを維持し、observerを解放するための新waitやworker slot検査を足さない。
GPU/stream/event/queue/bufferの操作回数・依存関係は変えない。sticky async error観測の
既存API/return判定、copy source寿命、host callbackの完了境界も保持する。

## 時計の範囲と包含関係

新時計は`steady_clock`のhost nsだけ。device timestampへの絶対時刻joinをしない。
`DecodeHostWindow::phase`は一つの現在leafを閉じ、**同じtimestamp**で次leafを開く。
従ってleaf間の重複はなく、全leafのsumは親`run_total`に一致することをcloseで検査する。
`other`はどの特定API bracketにも属さないhost区間で、差の推測で作る値ではない。
時計取得・counter加算の費用も観測中のhost wallに混ざり、性能比較のclean sampleに使わない。

| phase | source内の始点→終点 |
| --- | --- |
| other | run入口の新observer初期化後から最初のcapture bracket、および下記bracket間のhost処理 |
| capture | `capture(T,err)`呼出し直前→return。既存graphのhit判定だけで終わるcallも含む |
| capture_commit | `capture_commit(err)`呼出し直前→return。capture失敗時は試みない |
| host_staging | capture成功後→最初のgraphsubmit直前。token/position/PLE、flag/last-window staging等 |
| initial_graphsubmit | 最初の`graph_launch_with_completion`呼出し直前→return。直後のstream queryはother |
| layer_eventwait | `cudaEventSynchronize(layer_done_)`呼出し直前→return。aの後、progress呼出しの後 |
| poolcallback | bの直後→既存ms_wait/ms_pool更新直後。plan-slot準備、pool callback、event query、flags、fallback shared/hit launchを含む広いissuer scope |
| copy_stream_sync | 各layerの`cudaStreamSynchronize(copy_)`呼出し直前→return。status判定はother |
| next_graphsubmit | 次segmentの`graph_launch_with_completion`呼出し直前→return。status判定はother |
| tail_compute_sync | 末尾`cudaStreamSynchronize(cs_)`呼出し直前→return |
| tail_copy_sync | 末尾`cudaStreamSynchronize(copy_)`呼出し直前→return |
| finalreadout | 両stream drain後→成功return直前。任意既存profile/sampling/追加sampling sync、out読取、logit/NaN診断、progressを含む |
| run_total | 新window時計の最初→最後のstamp。上記全leafの親で、leafと加算しない |

`poolcallback`内の`publish_plan`からshared/hit graphsubmitが起きることをschemaに明記する。
今回はそのsubmitを別の子時計にしない。worker/callbackから時計を触る拡張と、compat内部の
registry/check/submit内訳は次段階へ保留する。
`finalreadout`には既存optionによる同期やlogit診断が混ざるため、最初の診断ではprofile/logit/
NaN/追加traceをOFFに揃える。final `wait_commit()`はrun外の既存joined時計をそのまま使う。
`run_total`は外側`dt_run`の全scopeと同一ではない（ledger begin/return判定、observer自身の
開始前・終端後処理などが異なる）。二つの時計の差をidle/submitと確定しない。

R604/R606の表現はMain補正を優先する。既存`ms_wait`のaはgraphsubmit**後**で、submit費を
含まない。native poolの`ms_drain`は`run_split_multi_native`全体であり、nativeの各run_phase
wait/drain/reparkを個別分解した値ではない。今回の22.219秒verifier、7.643秒wait、8.663秒pool、
0.255秒host、3.255秒draftの差からGPU idleやsubmit時間を確定していない。
この提案は未計測scopeを実測で分けるための診断である。

## 容量とreceipt

request長に依存するvector/個票/動的window履歴は持たない。64bit count/sum/min/max、
33bin base4 histogram（0単独、以降[1..3],[4..15],…,[2^62..2^64−1]）を使う。
全phaseとT1..4のmarginalにhist、最大64層の4phase（eventwait/poolcallback/copy_sync/nextsubmit）
にはmoments、window ordinalは1 / 2..8 / 9..64 / 65+の固定bandでrun_totalだけhistを持つ。
window個票と全layer×Tのcross-productは保存しない。

stateはheaderで`sizeof < 64KiB`をstatic_assert、close時だけ256KiBを確保する。
出力上限は**256KiB/request**、通常bodyに254KiB、最終complete footerに2KiBを予約する。
formatterは1回1024B未満、全appendを予算で検査する。元のstderr出力前に一つの固定bufferへ
構築し、途中の予算/format失敗はfooterのcomplete0にする。line単位の出力をrequest中に行わない。
最大69 histogram分布（13 global＋52 T＋4 band）と256 layer moments、固定schemaを出す。
各分布はmoments、p50/p95の**区間**、全33bin countを出す。層別にはquantileを出さない。
個票がないためexact median/p95とは呼ばない。異なるphase/viewのquantileを足さない。

complete1には元joined complete1、同じattempted/verified window数、全window成功、
RAII detach済、G1/full-layer/single-verifier envelope、各層4phaseが各window一回、
各window専用phaseが一回、leaf/親のsum一致、global/Tのcount/sum/hist一致が必要。
overflow、逆行/失敗clock、途中return/throw、未detach、missing phase、join不一致はcomplete0。
これらは診断の資格を落とし、元推論のstatus/return/ERR/DONEを変更しない。
異常終了・強制abortで最終footerがないreceiptはconsumer側でincompleteとして扱う。
元native fail-stopで中途observer receiptを出す変更も行わない。

## Mainの直列適用・検証計画

1. Direct0/1双方の公式bench全3 runsとowned processes/fault receiptsのclosureを先に確認する。
   そのbinary/source/設定・個別結果を確定する。今走っている92acf71e binaryを再buildしない。
2. このhash台帳とMainの現行sourceを照合し、baseが一致しなければ適用前にrebase/reviewする。
   source.diffの順序、6変更ファイル、独立headerを確認してからMainのみが適用する。
   CMake resolverからactual ordinary SYCL入口が`src/program/generate.cpp`→`Verifier::run`である
   こと、build recipeと实际コンパイル対象を再記録する。root CUDA generatorで代用しない。
3. CPU fixtureを独立にコンパイルして実行する。通常版、ASan/UBSan版を直列に使える。
   提案例（Mainがprivate output dirを用意）：
   `c++ -std=c++17 -Wall -Wextra -Werror -Iinclude PATH/cpu_contract_fixture.cpp -o PRIVATE/cpu_contract`。
   fixtureはGPU/CUDA/ExpertPoolヘッダを含まず、fake clockだけを使う。
   OFF無clock、明示200ns timeline、leaf/親、T/層/band、base4端点/quantile区間、join拒否、
   unwind/detach/次request、collision、clock逆行/失敗、overflow、出力上限/canaryを検査する。
   実Verifierのborrow wiringと既存pool fail-stopはこのheader単独fixtureで証明できない。
4. 既存joined MTP4 CPU契約/owner-abort fixtureと既存必要checksを直列に実施する。
   CMake/build/元binaryやmodelはこのagentでは実行・変更していない。MainがSYCL build後に
   source SHA、recipe/実compile flags、binary SHA、toolchain/driverをreceiptへ記録する。
5. 同一candidate binaryのOFF/ONを別fresh processで一つずつ行う。`cpu-gu-reuse-long-v1.json`のeffective environment（native capture0等を含む）を固定し、
   新envだけを変更し、
   joined2envは両armでON、CPU-GU observerとその他追加profiler/logitsは両armでOFFにする。
   まず小promptの通常MTP4で正常request、同process次request、request cancel/context limit/
   既存fault fixtureの必要部分を確認する。OFFで追加host receiptがゼロ、ONのみ正常receiptが
   同request identityで出ること、earlyexitがcomplete0で元status不変、全borrow detachを確認する。
   cancellationはエンジンが元々対応する方法で行い、診断のためにqueue waitを追加しない。
6. 既存long consumerの34603入力/128生成、chunk4096、int8KV、resident32768、FP32
   layer-major GPU残差4096行、Direct1、128 GPU expert slotsの通常MTP4を同条件で一回ずつ。
   既存canonical token sequence、W55/R217/D162/A76/C131/E128、68781 CPU jobs/101036 CPU
   expert-token rows/104160 routesは参照候補で、実出力を照合してから同workと認める。
   推論が異なれば異workの診断として残し、時間を性能比較しない。既存consumerのactual
   context envelope/新env allowed listをMainが必要なら明示更新し、その変更も別途reviewする。
7. request IDでjoinedとhostをjoinし、raw phase count/sum/min/max/hist、complete、token/hash、
   prompt/config、正常終了/owned PID回収/新fault0をcompact JSONへ保持する。global/T整合、
   layer counts、band counts、leafsum/run_totalをconsumerでも独立再計算する。
   stopwatch overheadの目安が必要ならMainがCPU-only clock-pair microbenchmarkを別途計画し、
   その値をmodel phaseから機械的に引かない。ON/OFFに時間差があってもobserver overheadと
   work/driver変動を区別するclean設計前に改善/退行を確定しない。

上記が通っても、品質全体、RESTORE、physical64K/256K、速度目標の達成とはしない。
`max_context=65536/262144`設定で短いpromptを走らせるだけではfull physical確認にならない。
最初は既存34K/128 ordinary requestのhost timingが目的で、context境界認定は別consumerが要る。

source変更前の先行screenとして公式`unitrace --host-timing`の小さい診断も選べる。
process aggregateはstartup/prefill/decodeやwindowを分けず、API wallからGPU idleを確定しない。
MainのDirect両armが閉じてから別ジョブで判断する保留候補であり、このagentは実行していない。

## 根拠と保持

`research-index.json`、R604、R606、`cpu-gu-reuse-long.ja.md`、
`cpu-gu-reuse-long-v1.json`、現行verify/generate/expert_source/decode_service_budget/
decode_route_ledgerと関連headerを読んだ。R604/R606原文は変更していない。
このartifactはMainの適用review/CPU契約/model計画のconsumer用。新measurement結果や進捗backup
ではない。Mainの採否境界でsource/report/hashをGitへ統合する（このagentはGit操作禁止）。
実測時はARTIFACT_RETENTIONに従い個別sample/command/env/fault/decisionを保存し、必要証拠を
抽出検証後に成功full protocolや重複traceをMainが排他lockでreviewする。ここではcleanupなし。
