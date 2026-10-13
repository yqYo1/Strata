<!-- SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# decode の CPU service 診断

`STRATA_DECODE_SERVICE_CPU=1` は、既存の native expert pool の CPU 時間を
request ごとに記録する。通常の単一 stage の decode、batch なし、MTP・suffix・
lookup なしを対象とする。raw verifier capacity と実際の NT は別であり、dispatch
で実際の NT=1 を確認する。対象外でも推論の選択は変えず、終了時に incomplete を
記録する。最大 64 layers、128 workers に限定する。

Linux の `CLOCK_THREAD_CPUTIME_ID` と host の steady clock を使う。
他の OS では CPU clock を unavailable とし、完全な結果とは扱わない。
flag が off のとき、新たな clock read、job ごとの割り当て、出力は行わない。
on のとき request 開始時に固定容量の state を割り当てる。

issuer（inline drain を実行する caller）と workers を分けて、native GU、
`native_quant_h`、native Down の invocation/range ごとの CPU nanoseconds と
wall nanoseconds、calls、clock read attempts を記録する。既存の Q2 GU/SwiGLU、
ActQ quantization、Q2 Down は独立した bucket とする。Q2 GU bucket は既存の
二つの row call と SwiGLU を含む一つの range を数える。claimed row tasks と
native call ranges は同じ数とは限らない。row の内側には clock を置かない。

issuer の per-layer dispatch 全体、および GU/Down の既存の
parked-before、publish-to-done、done-to-parked の境界にも同じ clock を置く。
worker ごとの単独 writer の固定 slot を使用し、既存の release/acquire publication
と worker-park completion の後で読む。新たな pool atomic、GPU event、copy、wait、
device branch は追加しない。request binding と終了処理の mutex は host 専用である。

これらの wall spans は重なっている。worker wall の和、phase wall の和、GPU ticks
を request wall に足してはならない。wall minus CPU は blocked/preempted を含む
residual であり、純粋な wait 時間ではない。既存の `decode_wall_ms` は commit wait
より前の request decode wall で、CPU diagnostic の formatting は含まない。
この診断だけでは GPU との overlap や critical path は確定しない。

一つの request close block は最大 65,536 bytes。job/row ごとのログは出さない。
clock read failure、逆行、counter overflow、task/quant count 不一致、layer coverage
不足、異常終了は incomplete とする。元の終了 status をそのまま保持する。
`cpu_jobs` は同一 expert にまとめた job 数、`cpu_activation_rows` は各 job の
`nt` の合計で、quantizer call 数は後者と比較する。dispatch NT と job の `nt` は別の値。
footer の `request_invalid`/`request_overflow` は request の flag のみを示し、
全体の成否は最後の footer の `incomplete` で判定する。
`length`/`stop` の close でも、各 layer の completed dispatch と clock receipt が
emitted token 数に一致しなければ完全とは扱わない。最後の close 行が判定を示す。
abort や強制終了で close 自体が出なかった場合も、完全な receipt とは扱わない。

`STRATA_DECODE_ROUTE_LEDGER=1` は owner、NT、call coverage の補助資料として
併用できる。ledger の logical bytes は実測 DRAM bytes、転送量、時間ではない。
ledger 自体は別の出力 budget を持つ。`STRATA_DECODE_TIMING` は既存の host 診断で
ある。`STRATA_VERIFY_PROFILE` は eligible DF branch を変えるので、この診断の
branch-preserving measurement では有効にしない。

CPU fixture は `tools/test_decode_service_budget.cpp`。clock accounting、aggregation、
overflow、reset、availability、normal/failed coverage、parallel worker の独立 slot、
off の clock read ゼロを確認する。assert を有効にする `-UNDEBUG` が必要である。
モデル上では同一 request の off/on output/logit parity と decode wall の変化を別に
確認する。clock read attempts は測定 overhead の記録であり、overhead の時間測定
そのものではない。
