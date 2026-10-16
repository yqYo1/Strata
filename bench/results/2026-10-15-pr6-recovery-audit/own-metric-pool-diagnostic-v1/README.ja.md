# Metric pool 単独診断の source-only 提案

状態は **SOURCE_ONLY_UNEXECUTED**。この worker は source の読解と作成、ファイルの
hash 記録だけを行った。configure/build/compile、構文検査、test、native/GPU 実行、
process/device の検査は行っていない。実行可能ファイルの承認や収集成功を意味しない。
Main が全 source を review し、build と実行を別々に管理する。

既存の `own-metric-query-pool-failure-v1.json` と
`metric-query-pool-failure-v1.stderr` は変更していない。そこでは ComputeBasic の
EVENT、domain 3、83 metrics と activation が成功し、最初の PERFORMANCE/count 64
pool 作成が `0x78000004 / ZE_RESULT_ERROR_INVALID_ARGUMENT` で停止した。
元の native exit は 125、ownership は閉鎖済み、fresh faults は 0、numerical result
は得られていない。この提案でその結果を PASS に変えない。

## 実装と終了条件

`metric_pool.cpp` は既存の独立実装 `own-metric-query-tiny-v1/enumerate_full.cpp`
の bounded enumeration、JSON escaping、API entry/result、pre-exec admission の考え方を
使用した新規 helper。`result_names.inc` は `own-metric-enumeration-v1` から
**byte-identical** に同梱した。外部の相対 include に依存しない。
新規 project files の license は repository の `LICENSE`。XeStrata の source、
helper、test、archive は使用していない。公式 Intel source は以下の動作と定数の
確認だけに使い、その実装は取り込んでいない。source と参照元の SHA256 は
`source-contract.json` に記録した。

全 driver（最大 8）と各 root（最大 32）を一回ずつ列挙する。handle の null、重複、
count 変動を拒否する。root GPU が全体で一つであり、それが non-subdevice の
`8086:e20c / 0000:05:00.0` であることを確認する。GPU の未知 device flags は拒否。
PCI extension の advertisement と loader symbol の両方を確認する。
対象 device の全 group（最大 256）を調べ、未知 sampling bits を拒否し、名前が
ComputeBasic、EVENT のみ、domain 3、metricCount 83 の一致を一つだけ要求する。
driver version/UUID、root UUID、選択した実際の handle を記録する。
この property check は 83 個の metric schema の再検証ではない。

同じ context/device/group で activation を一回行い、zero-initialized descriptor の
`stype=METRIC_QUERY_POOL_DESC, pNext=null, type=PERFORMANCE, count=64` で pool 作成を
一回だけ呼ぶ。成功時は pool destroy → deactivate（count 0/null）→ context destroy。
pool 失敗時も、成功が確認できた activation と context だけを順に片付ける。
cleanup の各 API status は別に残し、一つの cleanup API が error を返しても後続の
適用可能な cleanup を呼ぶ。query handle、query begin、queue、command list、event、
kernel、明示的 GPU buffer、submission、retry、count 1、provider fallback はない。
ただし pool 作成自身が driver 内部で allocation を行う可能性はある。

各 Level Zero API は直前の entry と直後の result を一行ずつ flush する。pointer は
数値、descriptor の flags/type/count/pNext と出力 handle は呼出し前後の実値。
error 時の `data` は部分的なメモリ状態で、`output_valid=false` とする。
entry に result がなければ completion は unknown。signal/deadline 停止から
cleanup 成功を推定しない。host exception、成功なのに null、失敗なのに non-null
なども成功扱いにしない。helper の lifecycle は公開 API の結果の範囲であり、driver
内部で捨てられる下位 cleanup status まで保証しない。

`operation_status`、cleanup、各 hold の状態は別項目。operation/cleanup が失敗すれば
125 で、post hold に失敗しても元の failure を保持する。pre-init hold の失敗、または
operation/cleanup 成功後の post hold の失敗は 65。CLI 誤りは 64。
**0 は pool probe と cleanup と二つの明示 ack が全て完了した時だけ**。
0 でも `collection_qualified=false, numerical_result=false` であり、model run を解禁しない。

## `iu.config` の根拠と admission

MetricsLibrary tag `1.0.231` の commit は
`74eac4065c178a2d459b9fba3acbf38a2fef4dc3`。
通常の `ContextCreate_1_0` 経路は `LibraryConfiguration::InitializeLogger` →
`IuLogGetSettings` → Linux `IuOsLogGetLocalSettings` と進む（redirect 経路は別）。
最後は **`/proc/self/exe` の親** の `iu.config` を開く。CWD の設定ファイルではない。
`IU_CONFIG_FILE_LINE_LAYOUT` の quoted key と `=dword:`、`wcstoul(...,16)` に合わせて
次の 3 行だけを ASCII/LF で置く。

```text
"AssertEnable"=dword:00000000
"DebugEnable"=dword:0000001f
"DebugLevel"=dword:e7000007
```

`DebugEnable=0x1f` は INSTR/MD/MDAPI/IU/ML の 5 layer。
`DebugLevel = 0x80000000 | 0x40000000 | 0x20000000 | 0x07000000 | 1 | 2 | 4`
`= 0xe7000007`。CONSOLE_DUMP、EOL、CONSOLE_FLUSH、SHOW_ALL と
CRITICAL/ERROR/WARNING を指定し、FILE_DUMP (`0x10000000`) は含めない。
AssertEnable は 0。gen/API override は設定しない。
console log が stdout/stderr に混ざり得るため全出力が JSON とは仮定しない。
ログの有無だけから installed binary の build 設定や失敗原因を断定しない。

helper は初期 hold 前とその ack 後に、必須環境値と executable-relative の実ファイルを
再確認する。config の symlink、非 regular file、違う size/内容、read/close error、
library の 260-byte path limit を超える executable/config を zeInit 前に拒否する。
config read は 1 回、最大 88 bytes（期待値は 87 bytes）。Main は private directory の内容を実行中固定する。
helper は environment を変更しない。`MetricsLibraryApi/Gpu`、`LD_PRELOAD/LD_AUDIT`
は存在するだけで拒否（空文字も拒否）。

driver source は compute-runtime
`050536ff3b6f830199ec99cff8ffcac3d804c40a` の
`level_zero/tools/source/metrics/metric_oa_query_imp.cpp` を **git show で**確認した。
checkout root の旧 22.x 実装は根拠にしていない。該当経路では MetricsLibrary
initialization、QueryCreate、zero GPU report size などの失敗が broad invalid argument
に潰れる。今回の原因がどれかは source だけでは分からない。
既存調査の driver `26.31.39395.14`、mapped `libigdml.so.1.0.231` / package `1.0.231`
と Discovery `1.14.188` は検討対象の識別情報である。
package version は参照 source の手掛かりであって installed binary がその source と
一致する証明ではない。dependency 差異は candidate のまま扱う。

## Main 向けの未実行 command 案

以下はレビュー用の build/link/env 案。新しい private executable directory だけに
config を置き、既存 executable directory、source package、seal、runtime を変更しない。
`mktemp` の親 path は短くして library の path limit を守る。compiler は Main がレビュー
した host C++17 compiler を使う。ここで `/usr/bin/c++` を承認済みとはしていない。
LLVM install を使う場合も compiler/header dependency だけであり、XeStrata 実装を使わない。

```sh
P=/home/yayoi/orca/workspaces/Strata/sycl
S="$P/bench/results/2026-10-15-pr6-recovery-audit/own-metric-pool-diagnostic-v1"
H=/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-primary-unitrace-build-root-v1/build/level_zero
O=$(mktemp -d /tmp/strata-mpool-v1.XXXXXX)
cp "$S/iu.config" "$O/iu.config"
timeout --signal=TERM --kill-after=3s 120s /usr/bin/c++ \
  -std=c++17 -O0 -g -Wall -Wextra -Wpedantic -I"$H" \
  "$S/metric_pool.cpp" /usr/lib/x86_64-linux-gnu/libze_loader.so.1.32.0 \
  -ldl -o "$O/metric_pool"
sha256sum "$S/metric_pool.cpp" "$S/result_names.inc" "$O/metric_pool" "$O/iu.config"
```

Main の owned controller が以下の native argv を pipe stdin/stdout/stderr で直接起動する。
`timeout` に native ownership の記録を代用させない。controller は **wall 90 秒、
TERM→KILL grace 3 秒、stdout+stderr 合計 64 MiB** の有限 budget と、所有 PID/starttime
に対する終了確認を実装してから起動する。各 native API 自体には timeout 引数がない。

```sh
env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C \
  ZET_ENABLE_METRICS=1 ZE_FLAT_DEVICE_HIERARCHY=FLAT \
  ZE_ENABLE_VALIDATION_LAYER=1 \
  "$O/metric_pool" --main-owned-hold
```

Main の既存 owner に合わせた有限 handshake は次の通り。helper には新たな controller
framework や seal を含めていない。

1. build/test/GPU/cleanup を直列化する。実際の command、source/binary/config/runtime
   hash、UTC、boot ID、owned native PID と starttime、effective environment、fresh fault
   window を記録する。新しい dependency は別に review する。
2. `hold/phase=pre_init` を受けてから 20 秒以内に、その PID/starttime を再照合し、
   `/proc/PID/{environ,maps,exe}` と config を owner が確認する。環境全体をログに
   漏らさず、必要な allowlisted values だけ残す。その後 pipe に **1 byte `X` のみ**を
   書き flush する。先送り、`XX`、改行、stdin EOF は ack にならない。
3. pool attempt と cleanup の entry/result を原形で捕捉する。
   `hold/phase=post_probe` を受けてから 20 秒以内に同じ ownership と actual maps を
   再確認し、特に MetricsLibrary の実 path/hash を記録してから新しい `X` を送る。
   post hold は cleanup 後なので一度 load された library が unload 済みの場合もある。
   snapshot に無いことだけから「一度も load されなかった」と結論しない。
4. process_summary、実 exit、所有終了、fresh faults/driver dump の有無を照合する。
   deadline/log budget 超過や result 欠落は INCOMPLETE。成功でも counter sample の
   qualification は別途必要であり、count 1 や再試行はこの実行には含めない。

Main は閉鎖後に `docs/ARTIFACT_RETENTION.md` に従って必要な compact evidence を残す。
未解決機序の代表 failure window を残し、full trace が必要なら owner/consumer/
byte budget/次回 review を記録する。既存 receipt の status/hash は変更しない。
