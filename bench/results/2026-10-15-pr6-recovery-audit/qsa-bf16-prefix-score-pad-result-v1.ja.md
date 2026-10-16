# FP32 score行幅32→33の比較

Arc B57010GiB / Ryzen5600X、2026-10-17 JST。現行K配置、16threadでのp計算、
BF16 matrix operand、FP32累積・max・32値sum・fma順、barrier数は同じ。
scalar staging scoreだけ行幅33へ変え、local明示payloadは28160→28224bytes（+64）。
Q/K/P/Vのmatrix loadのleading dimensionは変えていない。bank構成やcodegenを仮定しない。

同一binary、8192query/width2051/physical32768、同じinput/allocations、毎call同じprefix pack込み。
独立3process各2warmup・10標本、順序base0/pad0/pad1/base1/base2/pad2。

|経路|各processの中央値 ms|3processの中央値 ms|
|---|---|---:|
|行幅32|390.857487 / 390.798745 / 390.869305|390.857487|
|行幅33|333.186993 / 332.806483 / 332.926927|332.926927|

latencyは14.8214%減少した。全出力・入力fingerprint一致、独立4query/24576output
BF16 oracle・全有限・input不変・guard・最終repeatのsigned-zeroを含むbit一致に合格。
保存frame同士の全byte比較はしていない。先行traceと11caseの構造/数値検証
（physical262144KVを含む）もPASS。全所有終了、新規fault0、dumpなし。

これはcomponent host時間で、物理SLM bank-conflict/VRAM traffic/XMX使用率は未測定。
`*33`の追加address計算やallocation/codegen差も実時間へ含む。モデルPPは未測定。
確率計算の分散（別comparisonで9.93%短縮）との効果は加算せず、組合せを別に測る。
採用先は長大prefillの選択経路で、default/general-quality/full-model256K/RESTOREは未認定。
