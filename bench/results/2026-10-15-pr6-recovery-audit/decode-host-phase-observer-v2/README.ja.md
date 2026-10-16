# decode host phase observer v2 提案

2026-10-15 JST。**未適用・未コンパイル・未テスト・未GPU実行**。
v1は変更していない。唯一の書込先はこの新規v2ディレクトリ。
Mainの同一92acf71e binaryによるDirect0・Direct1の公式full suite各3 runsの**双方が閉じてから**、
Mainが適用/build/CPU fixture/modelチェックを直列に実施する。このagentは実行していない。

`source.diff`は現行5ファイルの変更と新規header1個の完全な適用候補。
`files/`は適用後の6ソースで、v1を先に適用する必要はない。
`generate_proposal.py`は製品入力SHA256を固定し、候補とdiffをこのディレクトリに再生成する。
baseが異なると生成を拒否する。`source-hashes.json`がbase/proposedの台帳、
`artifact-hashes.json`がREADME・生成器・fixtureを含む全artifactの台帳である。

## Mainレビューへの修正

1. `DecodeHostPhaseBorrow`の衝突拒否ではslotのpointer値だけを読み、pointeeを絶対に読まない。
   fresh ownerをinvalidにし、slotをnullにしてborrowを拒否する。
   旧slotがfreed/unprovenなaddressでも`slot->invalid`は行わない。
   まだ生きている旧borrowは、detach時にslotの不一致を見て、自身が所有寿命を知る旧ownerを
   invalidにする。旧pointerを復元しない。通常close前の明示detach、early return/exceptionで
   scope ownerより先にdetachするRAII順序はv1のままである。
2. `structurally_complete`にwindow bandとlayer marginalの保存則を追加した。
   4bandそれぞれのhist countがmoments countと一致し、合計count/sum/histがglobal RunTotalと
   一致する。ordinal bandの期待countは1、min(W,8)−1、max(min(W,64)−8,0)、max(W−64,0)。
   4個のlayer phaseそれぞれについて、各層count=Wを維持しつつ、全層count/sumの合計が
   対応global phaseと一致する。全加算はUINT64_MAXを超える前に検査し、wrapを許さない。
   不一致は診断complete0とし、元推論status/return/errorを変更しない。

receipt schemaは`decode-host-phases-v2`で、`band_and_layer_conservation=1`を明記した。
6ソース中、v1からの意味の変更は新規headerだけ。統合の5ソースはv1と同じである。

## 維持する契約と範囲

新envはexec前に子processへ渡す`STRATA_DECODE_HOST_PHASES=1`。getenv読取だけを使い、
setenv/unsetenvを導入しない。既存joined ledger/serviceの両env=1とeligible Linux SYCL ordinary
MTP4/G1/single full-layer verifierが必要。同じjoined request sequenceを使い、正常完了と
error/earlyexitに対応する。runtime/provider開始後にenvやarmを切り替えず、比較は別owned process。

default OFFでは追加allocation/clock/outputなし。ONでもissuerだけがwriterで、workerや
GPU host_task callbackへpointerを渡さない。追加queue操作/wait/buffer変更はゼロ。
既存pool例外fail-stop、全owner寿命、copy/compute completionとsticky async error観測を保持する。

capture、capture_commit、staging、初回submit、layer eventwait、poolcallback、copy sync、
next submit、末尾compute/copy sync、finalreadout、otherを排他的leafとしてsourceで括る。
run_totalはその親で、leafと足し合わせない。poolcallback内のshared/hit graphsubmitは内包と
明示し、callback内へ新clockを入れない。final wait_commitは既存joined時計の別scopeのまま。
既存ms_waitのaはsubmit後でsubmit費を含まない。native ms_drainはrun_split_multi_native全体。
22.219秒verifierとwait7.643/pool8.663/host0.255/draft3.255秒の差をGPU idle/submit確定と呼ばない。

stateは64KiB未満をstatic_assert、出力は256KiB/request（footer2KiB予約）、request長に依存する
個票なし。64bit count/sum/min/max、33bin base4 histogram、global/T1..4 marginal、最大64層の
4phase moments、ordinal4bandのrun_total分布を使う。p50/p95はhist由来の区間で、exact値を
捏造しない。clock異常/overflow/中途失敗/予算超過/missing footerはincomplete診断になる。

## Mainの直列検証

1. Direct0・1双方のformal full suiteが閉じ、owned process/fault receiptが確定したことを確認する。
   それまで製品/build/binaryを固定する。source pinsを確認してからMainが適用する。
2. configured SYCL resolver/actual generate→Verifier入口を確認し、通常SYCL buildをMainが行う。
   source/binary SHA、recipe/実compile flags、toolchain/driverを記録する。CUDA/HIP検証はしない。
3. 独立CPU fixtureをそれぞれコンパイル/実行する。以下は**未実行のMain用コマンド例**。
   `c++ -std=c++17 -Wall -Wextra -Werror -Iinclude PATH/cpu_contract_fixture.cpp -o PRIVATE/cpu_contract`
   `c++ -std=c++17 -Wall -Wextra -Werror -fsanitize=address,undefined -fno-omit-frame-pointer -g -Iinclude PATH/cpu_asan_borrow_fixture.cpp -o PRIVATE/cpu_asan_borrow`
   同時に走らせない。ASAN_OPTIONS等が必要ならexec前の子environmentで固定する。
4. `cpu_contract_fixture.cpp`はv1の明示timeline/OFF/clock/overflow/output/RAIIチェックに加え、
   band count/sum/hist/ordinalの個別破損、band加算overflow、4phaseごとのlayer count/sumの
   個別破損、global/Tは一致するがlayer totalと違うcount、layer sum overflowを分離して拒否する。
   `cpu_asan_borrow_fixture.cpp`は他fixtureに依存しない。freed heap pointerのslot拒否、live
   collisionで旧ownerをdetach前には触らないこと、元例外type/message/markerと元return値の
   保存、各失敗の後にfresh requestが正常完了することを検査する。v1のslot dereferenceなら
   freed-pointerケースがASanのUAF対象になる。実行結果をここでは主張しない。
5. 既存joined/owner-abortのCPU契約をMainが再確認する。legacy fixtureにsetenv/unsetenvがある
   場合は再利用前にレビューし、必要なarm設定はexec前か明示parameterへ移す。
6. 同一candidate binaryの新observer OFF/ONを別fresh owned processで実施する。既存joined
   2envは両armでON、CPU-GU observerと余分なprofiler/logitsはOFF、基準effective environment
   （native capture0等）と設定を揃える。小promptの正常/次request/earlyexitを先に確認する。
7. 既存34603入力/128生成、通常MTP4、chunk4096、int8 KV、resident32768、FP32 residual
   GPU4096行、Direct1、128 GPU expert slotsのlong consumerで比較する。同workは実ID/hashと
   W/R/D/A/C/E等を照合して判定し、差があれば別仕事として保持する。request sequenceでjoinし、
   count/sum/hist保存則とcomplete、元status、normal exit/owned cleanup/fresh faultsを記録する。
   ONの観測時間から性能採用やGPU idleを結論しない。既存成功full protocolはcompact抽出検証後に
   MainがARTIFACT_RETENTIONの排他手順でreviewする。このagentによるcleanupはない。

max_context64K/256K設定だけでfull physical確認済とはしない。全体品質・RESTORE・速度改善は
未認定。source不要の公式unitrace --host-timingは両Direct suite終了後の先行screen候補として
保留できるが、process aggregateだけではdecode/windowを分離せずGPU idleも確定しない。

consumerはMainのv2適用reviewと上記直列検証。次の見直しはMainの検証/採否境界。
ここにはraw captureや新測定結果を追加していない。Git/CMake/製品/既存fixtureは未変更。
