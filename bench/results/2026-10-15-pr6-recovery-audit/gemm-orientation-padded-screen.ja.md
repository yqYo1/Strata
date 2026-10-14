# GEMM記述比較の初回結果とstride訂正

2026-10-15 JST、Arc B570、oneMKL 2026.1、同じconfigured compiler/runtime。
standalone probeをbuildし、runtimeを作らないhost範囲検査が通った。
GPUでは8行数×GU/Down×beta0/0.25の32cases、各3armのCPU oracle96件を実行した。
row-major、column-major、現在のGemm wrapperのoutputは全casesで差0。
full outputのfinite、padding/両端guard、入力不変と独立double referenceも通った。
validation traceは953677 bytes、warning0。

別processの320samplesは正常終了したが、timing側でもvalidationと同じ
`ld=N+17`を使っていた。実MoE callerは`ld=N`なので、この結果を実モデルshapeの
記述比較や性能差の原因判定に使わない。元結果はsynthetic padded-stride measurementとして
PASSEDのまま保存し、書き換えない。実7format rotation/cold source/whole-model速度ではない。

v2はvalidationにactual `ld=N`とpadded `ld=N+17`の両方を含め、timingはactual `ld=N`を使う。
beta/数値tolerance/CPU oracle/guard/shape/queue/warmup/ABBAと個別sample保持は変えない。
新しいrun directoryとsource/recipe hashでbuildと検証を行う。
production GEMMと推論のdefaultは変更していない。

v1の両GPU jobはnative/owner/journal exit0、reaped/sessionempty、cleanup/errorsなし、
freshfault0、dumpなし、boot不変。これらの終了・数値結果をtiming条件の訂正と混同しない。
