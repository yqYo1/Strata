# PR6 signed/unsigned64 shuffle oracle 提案 v1

状態は **SOURCE_ONLY_UNSAFE_TO_RUN**。追加はこの directory だけで、Main の
実行中 source、`validation/**`、既存 report を変更していない。
`shuffle64_probe.cpp` と未適用の `integration.patch` を納品する。
configure、前処理、コンパイル、構文チェック、テスト、native/runtime/GPU、
profiler、process 調査、Git、cleanup、代理 agent は一切実行していない。
行ったのは source/API 宣言の読解、新規提案の記述、差分生成と SHA256 記録。

指定 base は `15a444bb91d54dbb86418158d9204308c6eeb5df`。Git では照合せず、
実際に読んだ31ファイルの path/bytes/hash を `input-source-snapshot.json` に固定した。
既存 wrapper の正確な preimage は `base-CMakeLists.txt`。patch はその同じバイト列への
CMake 追記と新規 `validation/pr6-restore/harness/shuffle64_probe.cpp` だけで、
古いファイル全体を置換しない。Main は base の一致と完全差分をレビューしてから統合する。
納品内容・snapshot・完全patch の hash は `contract.json` にある。

## 対象と有限な範囲

本物の `dpct::experimental::select_from_sub_group<long long / unsigned long long>` と
`shift_sub_group_right` を、各型一つずつの WG32/SG32 kernel 内で呼ぶ。
full mask `0xffffffffu` のみ。logical width は8/16/32。
すべての lane が同じ動的 call を通り、shift の境界判定は実 helper 内の
collective 後の元 guard を使う。oracle 側にその演算の置き換えはない。
型生成・結果保存とも `sycl::bit_cast` を使い、64bit payload の符号付き数値 narrowing をしない。

| 内容 | 一つの型、一つの input bank、各 lane あたり |
| --- | --- |
| broadcast select | 各 width の全 valid source、8+16+32=56回 |
| lane-varying select | `(lane*5+3)%width`、各 width 一回、計3回 |
| shift-right | delta=1/2/4、1/2/4/8、1/2/4/8/16、計12回 |

二つの input bank を使用する。bank0 は8セルごとに -1/UINT64_MAX、0、
signed minimum/maximum、2^32+17、2^40+lane、unsigned high-bit と異なる high/low half。
bank1 は全32 lane を識別できる `2^40+lane`。両 bank を両型で処理する。
全 valid source を読むため、broadcast は先頭 lane だけの検証にはならない。
lane-varying remote も0..width-1に収まり、arbitrary mask/divergent 呼び出しや無効 source は作らない。
int/double control は今回追加しない。long-long 分岐の bit identity という有限の問いに限定する。

結果には各 kernel の各 lane について、WG lane/id/size、physical SG lane/id/size/max-size/count、
実際の helper 呼び出し数を記録する。host は WG lane=physical lane=0..31、
SG size=max-size=32、SG id=0/count=1、WG id=0/size=32を全64件で要求する。
physical size が32でない場合は SG 内で一様に helper をスキップし、mapping/count 検証を失敗させる。
この経路は成功候補を出さず、空操作で合格させない。これはこの小さな geometry の確認に限り、
歴史的 WG128/1024 や dormant cluster route の mapping/実行を適格化しない。

host reference は元の `start_index=id/width*width`、
`logical_remote_id=start_index+remote%width` を使う。
shift は `(id-start_index)<delta` なら自身、そうでなければ `id-delta` の bits を要求する。
CPU で実 helper を呼ばず、readback に対する整数 bits の期待値だけを構成する。
全9,088値の64bitを `uint64_t` の完全一致で比較し、7,552 select、1,536 shift、
うち356件の境界 self-value を別に数える。比較は最初の不一致後も全値を続け、
最初の座標と全 mismatch 数を保持し、成功候補を出さず exit1 にする。
mapping が不正ならその時点で失敗し、不正な readback index を reference に用いない。
これは host-reference bitwise 検証であり、model perplexity/quality/numerical qualification ではない。

## build と実行の所有者

`PR6_BUILD_SHUFFLE64_PROBE=OFF` が既定値。ON は既存の
`PR6_MAIN_REVIEWED_BUILD`、Linux、`PR6_OPEN_SOURCE_LLVM_ONEMKL`、
`PR6_LLVM_SHUFFLE64_COMPAT` を要求し、既存 compiler/API/vendor hash gate を通す。
`pr6_shuffle64_probe` は EXCLUDE_FROM_ALL、C++20 required/extensions OFF。
provider probe と同じ `strata_includes` と `Threads::Threads` だけを link する。
GPU core の実 adapter 適用後 COMPILE_OPTIONS/LINK_OPTIONS を取得し、
SYCL、SG32、precise、split、backend と checked preinclude/include 順を引き継ぐ。
core/kernel/prefill/engine/weights/ggml/oneMKL の arithmetic 実装 library には link しない。
既存 wrapper 全体の closure/vendor/llama/MKL configure 検査はそのまま残る。
CTest 登録なし。全 target の生成・compile/link、ODR/ABI、実 flags/ELF/maps は未確認。

Main が現在の cohort を閉じ、完全 patch をレビューしてから、既存 pinned fresh recipe に
`-DPR6_BUILD_SHUFFLE64_PROBE=ON -DPR6_LLVM_SHUFFLE64_COMPAT=ON` を追加する候補。
build は `cmake --build <Mainの新規build> --target pr6_shuffle64_probe --parallel 1 --verbose`。
これは未実行の手順であり、実行中 build を変更する指示ではない。
Main は新しい checked-header/proof と実 compile/link commands を確認する。
既存 shift v2 の full compile/native、native full CLI/decode/D3 doorbell は未適格のまま。

実 child は引数なし、固定 ordinal0。Main が `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` を含む
全 runtime/driver/profiler 変数を exec 前に固定する。プログラムは env を変更しない。
実 checked default in-order queue を借用し、registry、GPU、Level Zero、SG32 対応、USM を確認する。
borrowed queue を本 fixture では破棄しない。三つの flushed input gate は
`PR6_SHUFFLE64_READY before-runtime`、`before-operations`、`after-cleanup`。
各 gate で Main が stdin に一 byte `X` を送るまで進まず、EOF/別 byte は失敗する。

二つの kernel、H2D二回、D2H一回、明示 retirement barrier二回、計7 queue commands の固定経路。
これは fixture の明示 submit 回数であり driver の内部操作数ではない。
入力512B、各 Result 37,632B、output二型75,264B、device 合計75,776B。
host seed/readback/input と USM 合計226,816Bで、static_assert による1MiB未満の上限もある。
tail は固定71 slotを host でも照合し、出力 seed の sentinel と call count で未実行を失敗させる。
可変 matrix/geometry/ユーザー tuning はない。

## lifetime と判定の限界

Probe が host input、immutable seed、readback と二つの USM allocation を所有する。
H2D/kernel/D2H は同じ in-order queue に提出し、最終eventとqueue全体を退役、asyncを消費し、
complete/emptyを確認するまで host readback を参照しない。
copy/submit が event 返却前に throw しても Probe destructor が全 queue を退役する。
失敗時も全 owner がその退役中存続し、不明な完了で unwind/free しない。
最初の runtime 呼び出しより前から既存60秒 process fail-stop deadline を保持し、
cleanup も bounded retirement を使う。Main の direct90秒/total120秒・64MiB log budget は別途必要。

完了不明/query/retirement/deadline/予期しない terminate は owners を unwind せず86。
既存 checked async handler の非空 delivery は87のまま。退役後の free 例外だけ88。
引数/env 不備は runtime 前に64、通常の host 検査失敗は退役/cleanup後に1。
成功候補は cleanupと第三gate後の一行JSON `SHUFFLE64_ORACLE_COMPLETE` と exit0。
実操作数、mapping/value/guard/mismatch counts、gate数、bytes、cleanup 完了を出し、
external fault receipt required と model quality false を明記する。missing receipt/timeout/budget stop は合格ではない。
Main は owned PID/start/boot、binary/env/runtime maps、journal 前後境界、fault0、終了とsurvivorを
別 receipt に記録する。fixture の申告だけでは provider safety や故障ゼロを証明しない。

前提は reviewed checked queue ownership、in-order ordering、実 provider barrier が部分受理も閉じること、
SDK の bit_cast と pinned compiler が要求した kernel を生成すること。非空async87、pending 経路、
provider 自身の部分受理例外、86/88 の fault qualification を本案は実行・再検証しない。
Main が報告した provider 四ケースの整数32セル/128B copy/default/owned/host-unwind
exit0/fault0 は既存の限定証拠として扱い、この新oracleやmodelへ一般化しない。
model quality/state、physical262144、RESTORE、600 tok/s は引き続き全て未適格。

新規 C++/CMake/report は repository LICENSE に従って独立作成した。
参照は自 fork の checked queue/retirement/provider 提案、pinされた歴史的 Intel DPCT 契約、
installed SYCL の public API 定義のみ。外部 implementation/template の移植はなく、
XeStrata source/helper/test/archive は読んでいない。既設 LLVM installation の包含 checkout は参照していない。
project-only upstream-sync skill の provenance/scope 原則に従い、upstream/Git 作業へ広げていない。
Main が統合・測定・保持・Git を所有し、次の review point は完全差分と実生成 command/初回build。
raw trace/tensor/binaryを作成・削除しておらず、既存失敗 receipt の状態も変更していない。
