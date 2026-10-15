# INT8 prompt head split v2 — exec時の環境を固定する提案

2026-10-15 JST。**UNAPPLIED / UNBUILT / UNTESTED / UNEXECUTED / GPU-UNQUALIFIED。**
書込は新規`int8-prompt-head-split-v2/`だけ。v1原本、製品、CMake、既存fixture、binary、
build/test/GPU/server/profiler/Git/cleanupには変更・実行していない。
controller/comparatorのhelp、host test、構文検査も実行していない。
実行したPythonは提案テキストの作成・転記とSHA256台帳生成だけである。

v1のprobeはruntime作成後にprocess環境を変更するため、SYCL/UR/ZE背景threadの環境参照と
競合する余地があった。Mainはv1 probeを実行不可と判断した。
v2は**1 native processにつきlegacy/32/64の1 armだけ**を動かす。
probeには環境を変更する関数を置かず、起動時のroute/Native KV設定を読み取り検証する。
Controllerは親Pythonの辞書に子process用envを作り、Popenのexec引数へ渡す。
親の`os.environ`や起動済みprocessの環境は書き換えない。

**適用前に、同じ`92acf71e`バイナリによるDirect0とDirect1両方のfull clean benchmarkを
閉じること。** thinking有効3 runs・agent sessionの実結果と、server/client/controllerの
owner退役を確認するまで、製品source/build/binaryを固定する。Direct0だけの終端では足りない。
`freeze-release.example.json`は未完の記入例であり、実行を解放するreceiptではない。
Mainが元benchmark receiptsの実hashと全owner情報に基づく新しいrelease receiptを作る。

kernelと`source.diff`はv1からbyte-identicalである。
Mode1 INT8 K/V + Direct exact1 + split exact32/64のみ候補、それ以外は既存分岐を保持する。
既存launcherは保存し、WG64/local rows8の8+4 split、共通12-head Q scale、CH32の演算順を保つ。
CH32のbitwiseは未証明の期待で、gateでは必須。CH64はchunk変更に伴う許容差と品質gateが必要。
packed local_accessorは31,088/53,872 B。動的SLM・spill・occupancyの実測ではない。
R605の「2 KV headsを1 WG」は誤りで、現行は1 query × 1 KV head/WG。
旧int c0のINT_MAX末尾加算は別課題として記録継続し、既存本文は変更しない。
その潜在overflowはcap262144では発生しない。新候補だけint64_t c0を使う。

`files/tools/sycl/int8_prompt_head_split_probe.cpp`は単一armの28決定的cases、または
isolation用1caseを走らせ、全出力+guardsと独立FP64参照を新規private captureへ保存する。
入力12配列それぞれのdtype/count/SHA256を記録し、全入力/guards不変を確認する。
同じbinary/source/fixtureと入力fingerprints、全FP64出力、CH32 bitwise、CH64 toleranceは
別processの`compare_captures.py`で照合する。`1e-5*(1+abs(ref))`を緩めない。
正常exit0とowner退役receipt、完全な28cases、Mainのfresh fault receiptが必須。
decline・不足・異常終了・timeout・予算超過は比較へ入れず、後続jobも停止する。

`serial_controller.py`と`capture_common.py`はMain向けの未実行controller案。
同じ既存exclusive lock、両benchmarkのfreeze-release、有限deadline/budget、専用新規path、
子processのPID/start ticks/boot ID/exit/retirement、全SHA receiptを管理する。
各native armの前後にMainがレビューしたread-only fault helperを必須とする。
そのhelperはMainの既存journal/driver診断に接続する外部adapterで、本提案では実装・実行しない。
helper無しではcontrollerはnative processを起動しない。kernel-name/SLM診断もMainが別途担当する。
helper interface、CLI、適用/build手順は`main-controller-plan.ja.md`と`capture-format.ja.md`に記録した。

23 isolation pairsも各設定を別processにする（46 native processes）。
内部で環境を切替える`sycl_prompt_attention_test`はv2計画から除外した。元testは変更しない。
全出力が同じでも実routeの証明にはしない。期待routeを保存し、実kernel名はMainが独立確認する。

raw output+FP64 refsの論理量はfull1 armで4,722,176 B、3 armsと23 isolation pairsで
17,563,904 B（metadata/log別、算術値）。各arm capture8 MiB、hard max16 MiB、log4 MiB、
controller全体64 MiBを初期上限とする。private raw capturesはGit外に保存する。
exact full arraysの現在のconsumerはoffline比較とMainの最初の差分調査。
比較と必要な診断を記録後、Mainが通常保持方針に沿ってpath/hash/reason manifest付きで退役する。
controllerはraw filesを削除・上書きしない。失敗の最後の必要fixtureを消してはいけない。

少量poolへ論理ID262143をmapするfixtureは、物理full256K/実入力消費/状態/RESTOREを証明しない。
モデル評価は最低32K実入力、thinking有効の公式bench、agent session、品質gateを別途要する。
既存profiler Directの44.066→29.805秒をhead-split性能やclean速度へ転用しない。
本v2にも測定結果、速度・品質・物理256K/RESTOREの認定、既定変更はない。

`input-sha256.json`が入力、`artifact-sha256.json`が本提案の出力を固定する。
保持consumerはMainのv2レビュー・直列適用/最初の検証、次の見直しは最初のcompile/host/device/
controller/comparator結果または却下時。本版も原本として残し、修正は次版へ記録する。
