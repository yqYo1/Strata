# Mainが直列に行う適用・検証案（全て未実行）

現在の公式bench Direct0 3 runsを閉じ、既存のexclusive measurement lockをMainが保持する。
別名のlockでGPU/build/test/cleanupを並列化しない。失敗/未完結果とowned processの回収を
記録してから本候補へ移る。Main以外の自動controllerは本提案から起動しない。

## 1. source reviewと適用

input-sha256.jsonの現ソースhashを比較し、driftがあれば差分をレビューする。
元提案を修正する代わりにv2へ記録する。CLI/API/既定値と既存launcherをreviewする。
Mainの通常Git手順で以下のdry-runと適用を行う（このagentはGitを実行していない）。

```sh
git apply --check bench/results/2026-10-15-pr6-recovery-audit/int8-prompt-head-split-v1/source.diff
git apply bench/results/2026-10-15-pr6-recovery-audit/int8-prompt-head-split-v1/source.diff
```

`files/tools/sycl/int8_prompt_head_split_probe.cpp`を同名の`tools/sycl/`へコピーし、
probe-target.cmakeの4行を現CMakeのmanual targetsへ追加する。
既存matrix-direct-probe-v1を適用済みにする必要はない。独立targetであり旧fixtureは保存する。

## 2. build確認

Mainの実SYCL build dirと記録済みconfigure recipeを使う。以下`$split_build_dir`は
Mainが実際の既存build dirに設定するtask-specific変数で、ここでは値を推測しない。
対象はCUDA/HIPではなくcmake/sycl_backend.cmakeのSYCL resolverであることを確認する。

```sh
cmake --build "$split_build_dir" --target sycl_int8_prompt_head_split_probe sycl_prompt_attention_test
```

CMakeにtargetを足すため再configureが必要なら、以前の全オプションを維持したrecipeを使う。
compiler/runtime/driver/hardware、実compile flagsとsource resolved path、AOT arch/GRF、
source diff/hash、linked binary SHA256をreceiptへ保存する。
`-fsycl-id-queries-range=size_t`と64-bit pointer/size_t、実prompt cppのmath flagsを確認し、
候補だけのfast-mathやresource knobを同時に加えない。
first buildは1200秒、text8MiBを初期上限案とする。期限やbudget停止は未完でありPASSでない。

コンパイラ受理/launch後にはnamed kernels C32/C64の実resourceを可能な範囲で収集する。
packed local宣言31,088/53,872 Bと実resourceを別欄にする。static SLM4096 B、device local容量、
kernelのlocal resource infoの意味が不明ならdynamic total/occupancyの証明には使わない。
unsupported形状/requiredWG/SLM/register/spill/driverエラーはbaselineとの差を保存する。

## 3. bounded component correctness

各native processを既存Main controllerのowned PID/start time/boot ID/有限deadline/ログ予算/
checked completion/fresh kernel fault receiptsで監督する。timeout後にbufferをfreeしたことを
完了証明にしない。probe内部は既存Runtime::wait/cleanup_fail_stop/owner leaseを使う。
初期予算はprocessごと900秒・text4MiB。診断用UR/ZE layerとclean timingは分離する。
自動reset、retry、サービス停止は追加しない。

```sh
"$split_build_dir/sycl_int8_prompt_head_split_probe" --help
"$split_build_dir/sycl_int8_prompt_head_split_probe" --host-only
"$split_build_dir/sycl_int8_prompt_head_split_probe" --validate32
"$split_build_dir/sycl_int8_prompt_head_split_probe" --validate
```

実binary pathがbuild rootでなければMainがrecorded target outputへ置換する。
--help/host-onlyはruntimeを作らない。--validate32が通ってからCH64を含む--validateへ進む。
候補fixtureには測定modeがなく、timingを示す値を出さない。

全28 correctness input casesを要求する。下表の件数をconsoleの実receiptと照合する。

| case群 | 個数 | nq/width等 |
| --- | ---: | --- |
| sparse_tail / dense_tail | 18 | widths0,1,31,32,33,63,64,65,2051。nq3 sparse/nq1 dense |
| all_missing / zero_scales | 2 | nq1/3、width65 |
| cap262144_logical262143 | 1 | nq3、active width65、full ids stride262144、小pool |
| large_query_tile0/1 | 2 | nq3,width65、2^20対2^-20、両KV head |
| subnormal_query / fp16_subnormal_scales / zero_query | 3 | nq3,width65 |
| negative_width / width_gt_cap | 2 | nq3、count -1/66、cap65 |
| 合計 | **28** | validate32は56calls、validateは84calls |

全候補armは同じQ/pool/table/IDs/steps/output allocationを使い、出力のみresetする。
入力/guard不変、全24*256*nq outputsのfinite/oracle、未書込sentinel、pair結果を保存する。
C32は旧Directとのbitwise必須。C64は旧Directとの許容差とFP64 oracleの両方。
各armの実bool falseはUNMEASURED_DECLINED、output不変を確認してfallbackは呼ばない。
probe exit2は少なくとも一つのdeclineを意味し、完走PASSへ読み替えない。
mode2/3等の既存gateでdeclineした場合はMode1個別結果と分け、全modeの認定をしない。
FNV64はfixtureのoperand fingerprintsであり、binary/source SHA256の代わりではない。

不正値9種、Direct off/legacy prefix4種×split2、他mode3種×split2のisolationは23 pairs/46calls。
これらは既存経路との出力不変を確認する。実routeを出力だけから識別できないため、
Mainのbounded kernel-name diagnosticで基準legacy、Mode1 split32/split64を別々に観測する。
期待kernel名はlaunch_mode1_head_split<32/64>を含む匿名lambda/関数名で、
実demangle表記はcompiler依存。Mode0/2/3やinvalid値で候補kernelが出ないことも確認する。
同じentrypoint boolだけでcandidate呼出を証明しない。

既存hookも再利用する。候補envをunsetした`sycl_prompt_attention_test`で既存pathsを確認し、
STRATA_SYCL_PROMPT_HEAD_SPLIT=32で同testを別processとして走らせれば既存Direct/staged
bitwiseとFP64 oracleの比較をC32へ適用できる。既存testはDirectを内部設定する。
同testに64を設定すると意図したchunk変更に既存bitwise gateが当たるので、C64のgateを
弱める変更はせず、専用fixtureの事前固定tolerance結果を使う。

## 4. component timingを追加するなら

本v1には未検証のbenchmark loopを増やしていない。Mainが旧matrix-direct-probe-v1の
厳密なmarker/checked-drain/ABBA計測loopを次版へ再利用する案とする。
既存probeをそのまま起動してhead-split速度と呼ばない。旧probeはDirect0/1を切替えるため、
head-split対legacy Directの条件と一致しない。

旧Direct vs split32、その後旧Direct vs split64を別々の同一binary比較として、
Directは常にexact1、NativeKVは0、head-splitだけunset/32かunset/64をABBAで切替える。
Mode1 dense-hot width2051、nq32/256を初期shapeにし、同一operandをreuse、両armを4calls
prewarm、40samples(20/arm)、4calls/sample、1whole-queue drain/sampleを初期条件にする。
同じkernelが実モデルで使う実nq/widthをMain profileから加え、hot syntheticの限界も記録する。
全個別wall/envelope/actual admissionを保存し、declineはtimed sampleへ入れない。
wallはsubmit+marker+drain、marker envelopeはsubmission gap込みでありexclusive kernel timeではない。
correctness/readback/oracle/trace/validationをtimed windowへ入れない。

## 5. モデルgateと採否

Direct自体のclean benchmark結果をまず閉じる。head-splitの比較でIndirect→Direct改善を
混ぜない。同一候補binary、同model/input/selection/chunk/KV/cache/residual/CPU/MTP設定で
legacy Direct、split32、split64を別processで比較し、32K以上実入力とagent session、
thinking有効、各3有効sample以上を要求する。元の公式bench.sh/client SHAを固定する。
標準例は既存手順の`./bench.sh --url http://127.0.0.1:18095 --long --think --runs 3`。
port/service実起動はMainの既存controllerを使い、同時serverを立てない。
標準32Kの実inputが32768未満ならその値を正確に記録し、厳密32768+は別追加で確認する。

CH64は同じ固定履歴のlogits/尤度/分布/教師強制top1、retrievalと状態/継続/RESTOREを確認する。
異なる自由生成履歴の後半を同位置logitsとして比較しない。生成ID完全一致だけを一般的な
品質基準にはしない。CH32でもcomponent bitwiseとモデルgateを省略しない。
必要なMTP強/弱・長時間生成は既存拡張benchのthinkingONで各個別結果を保持する。

物理full256K入力の最後の消費位置、出力余地がある近傍生成、範囲外拒否、実状態RESTOREは
別gateである。cap262144/ID262143 small fixture、32K/64K成功、context設定だけで代用しない。
未実施は未完のまま残し、速さがばらつきと重なる場合も既存実装を維持する。
source・commands・effective env・個別samples・correctness/fault・判断をMainの通常境界へ保存する。
採用しない場合も候補originと失敗/却下理由を保持し、元のprofile/benchmark statusを変えない。
