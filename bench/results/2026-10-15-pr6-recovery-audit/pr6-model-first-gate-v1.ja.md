# PR#6復元: 実モデルの初回確認

2026-10-16、Arc B570、固定LLVM22/libsycl9/driver1.17.39395+14。
検証branch `verify/pr6-prefill-restore` の `3602e9c4e1a33ab7a4148d63c5f340a3d25682d4`。
本番engineの採用や長入力性能の合格判定ではない。

復元prefill専用targetは113工程を実際にcompile/linkして完了した。
その後の変更はMainの4つのstdin holdと固定値の環境受理だけで、歴史的engineの
数式・処理順・kernel・D3禁止を変えていない。変更TUだけを既存の実compile/link argvで
再buildし、source/header不変と7つのarchiveの全hashを固定した。
最終binaryは `56bd0fefdf117f1cd6baea6a32f9acad0c8d82ba438b6fd6b2c61cc70f3db88f`。

## 実行結果と限界

| 実行 | 実際の結果 |
| --- | --- |
| model-small-v1 | 入力作成で未導入のtokenizers packageを参照して失敗。GPU実行なし。project自身のTokenizerへ修正した。 |
| model-small-v2 | 実モデルは36層目まで進んだが、UR全API traceが事前64MiB上限に達しSIGXFSZ/管理cleanupで中断。未完了のまま保存。 |
| trace-stop-review-v1 | 別の出力rootで終了を独立確認。元の全PID/start不在、reap/最終session観測、fresh fault0、dumpなし。元の失敗を合格へ書き換えていない。 |
| model-small-v3 | traceを警告+parameter-validation+既存進捗へ変更し、8tokenを48層/PLE/最終headまで処理。全finite、PLE窓一致、4hold、exit0、所有終了、fresh fault0。 |
| model-small-v4 | 同じ8tokenでZES_ENABLE_SYSMAN=0もexec前に固定し、startup関連環境の一致も確認。同じ48層と終了gateを通過、fresh fault0。 |

入力はproject Tokenizerで通常文を符号化した先頭8token。n=8、context8192、chunk8192、
first0、GPU residual8、ring8、K8V8、layer-major2/compact1/MMQ0/attnXMX0。
layer-major2で一つのchunkも48層の専用処理を通す。全expertの固定読み込みコストがあるため、
この短入力の時間をPP比較に使わない。v3/v4のstartup/JIT/pagecache条件の差も速度改善と扱わない。

v3/v4の最終hidden stateと全logitsのSHAはそれぞれ一致した。
これは同じ8tokenの再実行が一致したという証拠で、現在の実装との数値比較や一般のモデル品質、
全state・physical262144・RESTORE・600tok/s再現を認定していない。
長入力性能は公式bench.shのworkload/計時に対応させて別processで評価する。

## 環境観測で分かった点

v3ではprocfsにあるexec時領域は指定環境と一致したが、harnessが最初のMain holdより前に
収集したenvironには、指定していなかったZES_ENABLE_SYSMAN=0が既に存在した。
どのlibraryのstartup処理が加えたかは未特定で、runtime初期化後に変わったと断定しない。
v4はこの値をexec前から渡し、実際のstartup関連環境と指定値が一致した。
/proc/PID/environだけではheapへ追加されたsetenv後の値を検出できないため、
procfs照合を「実行中の全環境が不変」という証明にしない。
この仕様は [Linux man-pages](https://man7.org/linux/man-pages/man5/proc_pid_environ.5.html) でも確認した。

各実行の元receipt・source/commands・環境/maps・個別結果・journal境界は同directoryの
own-pr6-model-small-v{1,2,3,4}.json、trace-stop-review、build/ELF receiptにある。
成功trace・重複logと入力は、必要な値と異常代表窓の保存を確認後に退役させる。
