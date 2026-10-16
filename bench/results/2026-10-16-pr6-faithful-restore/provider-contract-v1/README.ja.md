# PR6 model-free checked-provider probe 提案 v1

状態は **SOURCE_ONLY_UNSAFE_TO_RUN**。Main の実行中 snapshot は編集していない。
追加先はこの `provider-contract-v1/` だけで、実行可能な C++ 提案
`provider_probe.cpp`、未適用の `integration.patch`、契約・pin を記録した
`contract.json` を納品する。configure、前処理、コンパイル、構文チェック、
host fixture、native/provider/GPU/model 実行、プロセス調査、Git 操作、cleanup は行っていない。
Main が独立レビュー、統合、ビルド、各 child の実行と判定を担当する。

既存 source は `input-source-snapshot.json` に全218ファイルの path/size/SHA256 を固定した。
assignment の HEAD 表記は `322e1fd8` だが Git では確認しておらず、完全ハッシュ一覧を
この提案の基準とする。patch の既存ファイル変更は外側 validation CMake への追記だけで、
`base-CMakeLists.txt` がその正確な元バイト列。もう一方は新規
`validation/pr6-restore/harness/provider_probe.cpp` であり、納品 C++ と同一。
Main は元 SHA256 の一致と完全 patch を確認してから適用する。実装中 snapshot が変化した場合に
古いファイル全体を上書きせず、追記内容をレビューして適合する。

## 作る target と依存関係

`PR6_BUILD_PROVIDER_CONTRACT_PROBE=OFF` が既定値。ON の場合も外側の
`PR6_MAIN_REVIEWED_BUILD` と `PR6_OPEN_SOURCE_LLVM_ONEMKL` を通り、Linux の
既存 pinned LLVM adapter が必要。新 target は `pr6_provider_probe`、
`EXCLUDE_FROM_ALL`、C++20 必須、拡張OFF。
`strata_includes` と `Threads::Threads` だけにリンクし、core/engine/prefill、
48-layer、weights、ggml CPU arithmetic、oneMKL BLAS、decode、Verifier、draft、
doorbell の実装ライブラリはリンクしない。wrapper 自体の既存 configure/vendor/
llama/MKL 検査と全履歴 source inventory はそのまま実施される。

GPU core の adapter 適用後の実際の COMPILE_OPTIONS/LINK_OPTIONS をコピーするため、
SYCL、SG32、precise、per-kernel split、JIT/AOT backend 設定は既存設定を使う。
CPU target の private `-fno-sycl` は継承しない。checked-DPCT の forced preinclude と
migrated/project/vendor include 順は `strata_includes` から継承する。
既存12コマンドのhost fixture PASS や SHA33 PASS は別の既存 receipt としてpinし、
新 target の build/provider PASS へ転用しない。parser16 は assignment の情報であり、
この worker が結果を再検証したものではない。進行中buildの成否はここでは更新しない。

Main が現在のbuildを閉じてレビューした後の候補手順は、同じ pinned configure recipe に
`-DPR6_BUILD_PROVIDER_CONTRACT_PROBE=ON` を追加し、生成コマンドを確認してから
`cmake --build <そのbuild> --target pr6_provider_probe --parallel 1 --verbose`。
これは未実行の手順であり、既存buildへの並行変更指示ではない。
Main は新 target の実際の preinclude/private-header hash、C++20/SG32/precise/offload flags、
link closure/ABI/ODR、compiler/runtime/UR/Level Zero の実体を確認する。
過去生成 private header の参照hashは `contract.json` にあり、今回の生成結果ではない。
CTest 登録は追加しない。

## 有限な別processケース

native child の引数は `--case NAME --device ORDINAL`。一度のprocessでは一ケースだけ。
すべてのSYCL/driver/collector変数と `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` は
Main がexec前に固定する。プログラムにsetenv/unsetenvはなく、selector拒否と引数解釈は
最初のDPCT呼び出しより前に行う。deviceは初期化時に一度選び、GPU/Level Zero、FP16、
host/device USM、SG32を検査する。

| NAME | 実際に提出する操作と検証 | 期待exit |
| --- | --- | --- |
| `baseline-default` | checked default queueを借用し、128B H2D、32セル整数kernel、128B D2H、eventとbarrierをretire、全セル照合、free | 0 |
| `baseline-owned` | 同じcontext/deviceのDPCT `create_queue(true)`を使い同じ処理。そのownerをretire/destroy後、registry拒否を確認し、同じborrowed defaultで再びcopy/kernel/照合 | 0 |
| `partial-host-default` | 実copy返却後・event保存前に明示的HOST例外。production DraftUploadGuardのunwindで実queueをretireし、copy結果とsource不変を照合。その後baseline | 0 |
| `partial-host-owned` | 同じHOST例外/unwindをowned queueで検査し、baseline、owned破棄後default survivalも検査 | 0 |
| `async87-host-list` | 現installed SDKのpublic APIでは非空exception_listを作れないため、明示HOLDを返してruntimeを初期化しない | 77 (未適格) |

小kernelの計算は `uint32[i] = uint32[i] * 3 + 7` のみ。入力は小さい既知整数で、
SG32/local32/global32、整数の完全一致を要求する。元sourceのFP32/precisionやモデルmathを
変更しない。H2DとkernelとD2Hは同じin-order queueに提出し、readback eventとqueue barrierの
完了・async消費の後にしかhost出力を読まない。empty operationや例外catchだけで完了とはしない。
全ケースに非空copyと非自明kernelがあり、ownedケースでは破棄後defaultでも再実行する。

状態queryは実eventのsubmitted/running/completeとqueue emptyのboolを別々に数える。
小処理がすぐ完了してpending観測が0でも、その値をそのまま記録する。
これはpending-provider経路の適格化を意味しない。意図的なGPU待ち、無効kernel、device fault、
watchdog hangは作らない。partialは「本物のaccepted copy + 注入したhost境界」の証拠候補で、
provider自身が部分受理後にthrowした証拠ではない。

installed `sycl/exception_list.hpp` はread-only iteration/sizeのみ公開し、PushBackはprivate。
したがって非空host listで実checked handlerを呼ぶ87ケースは本案では実現できない。
private API/ABI操作、empty-list成功、別handler、host_task例外、GPU faultへの置換はしない。
87適格化は厳密にHOLDのまま。将来public APIか別途レビュー済みの安全な実delivery手段が
得られたときに別scopeで扱う。

## lifetime、終了と証拠の限界

Probe内のhost配列とUSM pointerはqueue操作より先に存在し、destructorでqueue退役後に
解放される。submitがthrowしてeventを受け取れなくても、Probeがbufferを保持したまま
queue全体のchecked barrier退役を行う。partial HOST例外ではその前にproduction guard自身の
unwindも通り、drain回数とcopy結果を検証する。sourceは退役前には変更しない。
ownedを借用したraw pointerは `q_of` の実registryに照合し、defaultとownedの
in-order/context/deviceを検査する。owned破棄後は旧raw pointerをregistry照合にのみ渡し、
dereferenceしない。borrowed defaultは本probeではdestroyしない。

最初のDPCT呼び出しより前から既存RetirementDeadlineの60秒process fail-stopを開始する。
完了不明、query/retirement失敗、deadlineは86でbuffer ownerをunwindせず終了する。
checked async handlerの非空deliveryは既存87のまま。完了確認後のfree/destroy例外は88。
closeのwatchdogを作れないなど完了確認前の失敗は86にする。通常のhost検査失敗はownerを
退役・cleanup後に1を返す。引数/selector不備はruntime前に64。
terminate handlerは既存harnessと同じ88 fail-stopで、例外を成功にしない。
86/88の実故障・deadline適格化は未実施であり、意図的に誘発するcaseも追加していない。

成功候補receiptは `PROBE_CASE_COMPLETE` と操作数、実pending観測数、HOST例外/drain数、
default survival、expired registration、cleanup完了をstdoutに記録する。
それはprocess本体の申告であり、driver fault0やprocess消滅の証拠ではない。
Main がdirect90秒/total120秒、64MiB raw budget、journal前後境界、owned PID/start/boot、
実環境、binary hash、runtime maps、exit、survivor/cleanupを別receiptに記録する。
missing receipt、timeout、budget stop、77はPASSにしない。raw logはMainの保持方針で扱い、
workerはlog/tensor/binaryの作成や削除をしていない。

残るHOLDは、実build/link/provider、pending観測、provider由来部分受理/例外、async87、
watchdog86、cleanup88、capture/cancel、partial-init/late-init/relayout、ring wrap/tail、
本prefill/modelの数値・full KV/scales/GDN/conv/indexer/PLE/quality、physical262144、
export/exit/fresh RESTORE/continuation/rewind/branch、historical600。baselineが後日通っても
これら全体の適格化にはならない。D3 constexpr falseはstatic_assertで要求し、
native full-model CLI/decodeも引き続き除外する。original198/vendor49 inventory、historicalな
数値、既存kernel/mathバイト列は変更していない。

新C++/CMake/reportはrepository LICENSEで独立作成し、参照はpin済みown-fork safety API、
historical Intel DPCT dependencyのpublic API、installed LLVM SYCL SDK headerだけ。
XeStrataのsource/helper/test/archiveや外部テンプレートは使用していない。
LLVM installationはdependencyとしてのみ扱い、包含checkoutのsourceを読んでいない。
project-only upstream-syncの範囲原則に従い、この有限source依頼をGit/upstream更新へ広げていない。
Mainがintegration/保持/commit/pushを所有し、次の見直しはこのpatchのレビューと生成command検査。
