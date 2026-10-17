# MoE窓のGPU経路用fixture（未実行）

Mainの新規MIT実装。`tests/sycl/moe_window.cpp`とmanual専用
`sycl_moe_window_test` targetを追加した。既定build/CTestには含めない。
基準engineと候補engineの数値・速度採用は変更していない。

productionのC++ route builder、`gather_rows16`、native16 `combine`をリンクする。
expert演算は元入力とexpert IDから決まる合成witnessに置き換え、GEMM/dequantを
再現したとは扱わない。元tokenのexpert IDsから直接計算するGPU oracleは
route offsets/jobs/slot-mapを参照しない。全gather要素と全最終出力を照合する設計。
shared scalar gateを0に固定し、sigmoid差でrouting検証が揺れないようにした。
bit一致を要求するのはこの合成witnessだけで、実モデル出力の完全一致要求ではない。

各GPU spanには前後256Bの実canaryを設け、checked readback後に全byteを確認する。
元IDs、route plan、host readback先、device bufferをleaseより先に宣言し、
非同期source/slot uploadから最終drainまで所有する。失敗drainではowner解放前に
fail-stopする。guard readの一時host bufferも失敗時にunwindしない。

実行予定の独立processケース：

- `smoke`（既定）: 3行、異なるtoken routing。
- `single`: 1行。
- `hot8193`: 同じ10 expertに全行を割り当て、各expertの8192+1 jobを検証。
- `hot16384`: 同じ10 expertに全行、8192+8192。ne==windowを許す。
- `balanced16384`: 全512 expertに分散。
- `tail16383`: 最大窓から1行少ない場合。
- `unwind`: uploadと最初のgatherをenqueue後に意図的にthrowし、
  host/plan ownerがliveなままchecked leaseのdrainに到達したことを確認する。
- `negative-source` / `negative-slot`: 範囲内のmap値を一か所だけ故意に壊し、
  oracleが差分を検出しなければfixtureをfailさせる。

各process内でsalt=1と37の別operand/routeを使う。ただしbuffer ownerは各回別であり、
同一arenaの再利用やengine実経路の寿命を直接証明するものではない。
小さいtrace付きsmokeを先行し、合格してからvalidationのみの大ケースへ進む。
全ケースを一括traceにしてログを肥大化させない。

`run_moe_window_fixture_syntax_v1.py`で既存Intel SYCL compile databaseから
実fixture TUをO0/fsyntax-only/Wall/Wextra/Werrorで検査しPASS。
nice19、wall45秒、process CPU20秒、RSS2GiB、artifact8MiB上限。
実所要11.797秒、peak観測session RSS334639104B、子process全終了/reap確認。
この時間を性能値として扱わない。CMake configure、codegen、link、
全GPUケース、実モデル、attention/PLE/GDN/QSA/HC/MTP、cancel後server再利用、
物理262144 context、保存/RESTORE、長入力速度は未検証。

foreign CPU llama-bench PID3431768/start52544839は実装前にもlive。
そのprocessや環境は変更せず、heavy build/GPU実行も行っていない。
未完了verify/moe-windowは実GPU検証のため保持し、完了後feature/syclへ統合して削除する。
