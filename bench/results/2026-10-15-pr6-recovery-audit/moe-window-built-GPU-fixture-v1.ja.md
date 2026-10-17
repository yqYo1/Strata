# MoE窓候補: 完全buildとGPU fixture

2026-10-17、candidate source53c759cbから新private Release buildを実行し、
strataとmanual sycl_moe_window_testのcodegen/linkがPASS。build中のsource hashを確認し、
全所有processの終了・fresh kernel faultsなし・devcoredumpなしを記録した。
commands/compile entries/compiler/runtime/source/binary identitiesは
moe-window-engine-build-v1.jsonに保存した。基準Pの旧binaryは変更していない。

run_moe_window_fixture_v1.pyで、別processのsmoke/single/unwind/
negative-source/negative-slot/negative-expert/negative-down/hot8193/hot16384/
balanced16384/tail16383を順に実行し全11case PASS。各caseはsalt1/37の2sample。
最初のsmokeにcombined UR tracing+parameter validation、他はvalidation/warningsを使用。
実allocationの前後canary、独立stable-sort oracle、gather/ID/Down/combine照合、
全所有process終了、fresh kernel faultsなし、dumpなしを確認した。

hot8193/hot16384は各10expert、8192+1/8192+8192の20jobを実行。
balanced16384/tail16383は512job。正しい入力では全counterが0。
negative-sourceは各saltで全2560列のgather差分を検出し、wrong-expertもexactIDで検出した。
negative-downのsalt1ではDown差分1に対してcombine差分0であった。
つまり最終BF16出力だけなら誤りを見逃す場合を実GPU上で確認し、
今回の独立pre-combine検査がそれを捉えた。salt37でもDown差分1を検出した。

これはproduction route/gather/combineの合成fixtureであり、real expert GEMM/dequant、
実FFN/attention/HC/MTP統合、real-model数値/品質、cancel後server reuse、
物理262144 context・保存/RESTORE・速度を資格付けするものではない。
engine候補は引き続きDraft PR32に保持し、次はoptimized QSAを明示した実modelの
window0/2比較へ進む。packed QSAは別の診断でqueue完了待ちが進まなかったため使用しない。
再起動せず、owned process停止後のruntime/native16 health両方がPASSした。

全case/counterはmoe-window-fixture-GPU-v1.jsonに保存。
成功したfull API traceはcompact証拠を確認した後のretention reviewで退役する。
