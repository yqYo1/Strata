# Mainの直列handoff — 全手順が未実行

## Freezeを解放する条件

同じ`92acf71e`バイナリでDirect0とDirect1両方の公式full clean benchmarkを閉じるまで、
製品source/build/binaryを固定する。thinking有効3 runs、agent session、個別結果と元receiptを
保持し、server/client/benchmark controllerの全owned PID/start ticks/boot IDの退役を確認する。
一方だけ終端、exitだけ、source変更後の再buildへの置換では解放しない。

MainはGit外の新しい実release receiptをfreeze-release.example.jsonのschemaで作る。
exampleのfalse/未完/placeholder値を書き換えて本artifactへ上書きする使い方はしない。
両armのCLOSED_FULL_COMPLETE、同じfull binary SHA256、実receipt path/SHA、thinking/runs/agent、
全owner一覧/退役をMainの実結果から記録する。
`--check-freeze`はschema/hash/owner不在をread-onlyで確認する。元benchmarkの測定値そのものの
正しさはMainが元receiptから判断する。単なる自己申告schemaを独立なbench検証と呼ばない。

以下は将来Mainが実行するための例。ここでは一つも起動していない。

```sh
python3 -B bench/results/2026-10-15-pr6-recovery-audit/int8-prompt-head-split-v2/serial_controller.py \
  --check-freeze --freeze-release "$split_freeze_release"
```

Mainはこの確認を共通exclusive measurement lockの下で行い、その後にのみsource適用/buildへ
進める。controller自体はsource適用、Git、configure、build、サービス制御を行わない。

## 適用とbuild

input-sha256.jsonと現fileを比較してdriftをreviewする。
source.diffとkernel snapshotはv1と同じbytes。既存launch/default/admissionを維持する。
Mainの通常手順でsource.diffをdry-run/applyし、v2 fixture sourceをtools/syclへコピー、
probe-target.cmakeのmanual targetだけを実CMakeへ追加する。
元v1と既存test/fixtureのsourceは変更しない。

```sh
git apply --check bench/results/2026-10-15-pr6-recovery-audit/int8-prompt-head-split-v2/source.diff
git apply bench/results/2026-10-15-pr6-recovery-audit/int8-prompt-head-split-v2/source.diff
cmake --build "$split_build_dir" --target sycl_int8_prompt_head_split_probe_v2
```

再configureが必要ならMainの既存SYCL recipeを維持する。
CMakeListsのSTRATA_ENABLE_SYCL → cmake/sycl_backend.cmake → strata_kernelsの実cppと
qsa_prompt_attn_batch入口、compile flags/AOT arch/GRF、binary SHAを照合する。
64-bit size_t、-fsycl-id-queries-range=size_t、math flagsの比較も含める。
build上限案は1200秒/text8 MiB、失敗/期限/予算停止は未完として保持する。

`sycl_prompt_attention_test`は内部env切替を持つため、このv2計画ではbuild/run対象へ加えない。
STRATA_TEST_PROMPT_MATRIXの既存許容差を参照しても、元testの実行を意味しない。
v2 probe自体の--help/--host-onlyはcontrollerが別processで先に実行する。

## Main fault adapterを用意する

serial_controllerは各native arm前後のfresh fault gateを必須にする。
Mainの現在のjournal/driver診断へ接続する**レビュー済みread-only adapter**を
`--fault-helper ABS_EXECUTABLE`で渡す。helper自体のSHAも固定する。
この外部adapterの実装・実行は本taskでは行わず、架空のPASSも作らない。
helperが無い、scan権限が無い、cursorが取得できない、既存driverのfilterが未確定なら
controllerはnative jobを開始/継続できない。Mainの既存controllerへprobe/comparatorだけを
組込む場合も、下記と同等のcompact process/fault receiptsを作る。

helperのexec argvは`HELPER begin --job-dir ABS`または`HELPER end --job-dir ABS`。
stdout/stderrは各4 MiB以内、deadline60秒（異常退役tail最大40秒）で監督する。
beginはnative job dirへ新規`fault-before.json`を作る。

```json
{"schema":"head-split-fault-before-v2","boot_id":"actual boot ID","cursor":"actual kernel-journal cursor"}
```

endは同じboundaryから実scanし、新規`fault-after.json`を作る。

```json
{"schema":"head-split-fault-after-v2","boot_id":"same actual boot ID","before_cursor":"same actual start cursor","after_cursor":"actual end cursor","filter_id":"Main-recorded exact driver-fault filter/version","error_scan_success":true,"new_faults":0,"status":"PASS"}
```

上記はformat例であって実証拠ではない。filter定義、journal commandの成否と関連entriesは
Mainの通常compact receiptに保持し、permission error/欠損をfault0に読み替えない。
helperはGPU reset、service操作、cleanupを行わないことをMainがsourceで確認する。
helper failure、native abnormal exit、capture失敗後は後続jobを発行しない。
失敗後のfresh fault調査/未知GPU状態の扱いはMainの既存停止方針へ戻す。

## Controller例

Mainが既存共通lockを他jobから解放した後、同じlock fileを以下controllerへ引き継ぐ。
別lock名を作って既存GPU/build/test/cleanupと並列化してはいけない。
出力rootはGit外の絶対・未作成pathで、直近parentが本人所有・他者非writeの場所を指定する。

```sh
python3 -B bench/results/2026-10-15-pr6-recovery-audit/int8-prompt-head-split-v2/serial_controller.py \
  --freeze-release "$split_freeze_release" \
  --binary "$split_probe_binary" \
  --source /home/yayoi/orca/workspaces/Strata/sycl/src/kernels/sycl/prompt_attention.cpp \
  --fixture-source /home/yayoi/orca/workspaces/Strata/sycl/tools/sycl/int8_prompt_head_split_probe.cpp \
  --lock-file "$split_common_lock_file" \
  --fault-helper "$split_main_fault_helper" \
  --output-root "$split_new_private_output" \
  --include-isolation
```

Python base envを一度snapshotし、route2変数を取り除き、Native KV=0を子exec辞書へ入れる。
Q4と診断用runtime envはそのsnapshotから継承して記録する。Mainは既存の明示launch whitelistを
使うこと。秘密を含む全env dumpは行わず、route/native/Q4をtargeted receiptへ保存する。
--include-isolationを省けば3 full armsと比較のみ。付ければ23 pairs/46 separate native processes。
各jobは同時実行せず、source/fixture/binary hashを前後で照合する。

順序はhelp→host-only→legacy28cases→split32同28cases→offline CH32 bitwise比較→
split64同28cases→offline full比較→任意23 isolation pairsの左右別processと各offline比較。
CH32不一致を自動でtoleranceへ緩めたり、split64だけ続けたりしない。
各native900秒、capture8 MiB、log4 MiB、比較120秒、controller全体64 MiB・21,600秒（異常退役tail最大40秒）。
helper2process/nativeも含む最悪byte reserveを次job開始前に確保する。
controllerが受け取ったexit0/owned PID/PGID退役とMain fault gateが揃ったcaptureだけを比較する。
異常停止時は自分のprocess groupだけを有限に退役し、未知状態を正常GPU完了と扱わない。

23 isolationはinvalid split9種、Direct off/empty/legacy-prefix4種×split2、他mode3種×split2。
各pairの基準は同じDirect/formatでsplit unset、比較側だけexec時のsplit値を変える。
片側だけのdeclineでもfail-closed停止。特にMode2/3の既存local-memory/Q4 gateを緩めない。
必要な未測定modeの記録を残し、exit2を全mode PASSへ読み替えない。

## Mainが別途完了させるgate

期待routeと同じ出力だけではkernel選択を証明できない。Mainのbounded診断でlegacyと
launch_mode1_head_split<32>/<64>の実名、他mode/invalid/offで候補が出ないことを観測する。
実resource/driver allocation/spill、packed宣言量とdynamic SLMの区別も確認する。
controller/comparatorのOUTPUT_COMPARE_PASSはモデル/速度/SLM/routeの総合合格ではない。

速度はDirect0/1 clean結果を閉じてから、旧Direct vs split32/64を同一新binaryで分離比較する。
最低32K実入力、thinking有効の同じ公式bench/client、agent session、各3有効sample以上、
個別値/ばらつき/実workを記録する。標準例`./bench.sh --url http://127.0.0.1:18095 --long --think --runs 3`。
標準32Kの実inputが32768未満なら正確な値を残し、厳密32768+追加gateを別に満たす。
CH64は固定履歴のlogits/分布/教師強制top1、retrieval/状態/継続/RESTOREを含む品質gateを要する。
MTP強弱・長時間生成は既存拡張benchのthinkingONで評価する。異なる生成履歴を同じ位置として
比較しない。component比較が失敗した版を速度試験へ持ち込まない。

cap262144/ID262143 small physical poolは論理index検査だけで、物理full256K実消費、最後の位置、
出力余地がある近傍生成、範囲外拒否、状態/RESTOREは別検証。legacyのINT_MAX loop問題は
256Kで起きない別issueであり、今回の既存bodyを変更する口実にしない。

raw full arraysは現在のoffline比較/最初の失敗調査consumer向け。比較とMain診断を記録し、
必要な最後のfixtureを残した後、Mainが直列のpath/hash/reason削除manifestで退役する。
controller/comparatorは削除せず、本taskもcapture/trace/binaryを作成・削除していない。
