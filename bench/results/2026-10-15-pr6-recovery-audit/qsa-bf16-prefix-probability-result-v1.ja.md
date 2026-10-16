# 同じ数値更新順で確率計算を分散した比較

Arc B57010GiB / Ryzen5600X、2026-10-17 JST。現行K配置、BF16 Q/K/V/P、FP32
matrix累積・online状態を維持し、p exp2を16から128threadsへ分散した。max、exp2の引数、
32値を足す順番と最後のfmaは同じ。score SLMをunrounded pへ再利用し、tileごと2barrier追加。
同一binary、同じallocation/input、両経路の各callに同じprefix packを含めた。

8192 queries / width2051 / physical32768 / reverse page table。
独立3process各2warmup・10標本、順番base0/parallel0/parallel1/base1/base2/parallel2。

|経路|各processの中央値 ms|3processの中央値 ms|
|---|---|---:|
|現行|390.724850 / 390.811019 / 390.787401|390.787401|
|p分散|351.976080 / 351.923833 / 351.987750|351.976080|

latencyは9.9316%減少。追加barrierとshared score再利用の費用を含む。
これはcomponentのhost elapsed比較であり、モデルPP、物理traffic、XMX/SFU/SLM飽和率ではない。
全6processの入力・全出力fingerprintは一致。独立4query/24576outputのBF16 oracle、
全有限/input不変/guard、最終repeatのsigned-zeroを含むbit比較もPASS。
保存frame間の全byte比較はしていない。初回trace・11caseの構造/数値検証
（physical262144のKVケース含む）もPASS。所有終了、新規fault0、dumpなし。

production経路はまだ変更していない。モデル全体のclean中央値は445.4 tok/sのまま。
次にscoreのFP32行pitch32→33を独立に調べ、確率分散と組み合わせるかを判断してから
全engineへ統合し、公式bench.shの長入力を反復測定する。full-model256K/保存session
RESTOREは未認定。llama.cpp/FP32との完全一致を採用条件にせず、同モデルSGLang/vLLM契約を使う。
