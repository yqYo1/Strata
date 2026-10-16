# BF16 QSA score行幅33のモデル全体比較

同一バイナリ、公式bench.shのprefill-only拡張、thinking ON/生成上限24、実入力39262 token、context/resident65536、chunk8192、GPU残差32768行＋RAM。各arm3独立processを交互順に測定。

|経路|個別prefill tok/s|中央値|標本SD|
|---|---|---:|---:|
|基準score32|[445.7, 446.7, 441.2]|445.7|2.930|
|候補score33|[463.4, 464.1, 464.5]|464.1|0.557|

スループット+4.128%、prompt時間-3.960%。同じ入力body hash、全6回正常終了/ownership closed/新規GPU faultゼロ。部品測定14.82%の短縮をprefill全体の改善率として扱わない。

BF16 Q/K/V/P/出力とFP32の累積・max/normalizer、32セルの逐次更新順は同じ。all48合成129tokenのNLL/3箇所全語彙logit差はゼロ。精度の採用基準は同系モデルのSGLang/vLLM契約であり、llama.cppとの完全一致ではない。一般品質、full-model256K、保存/RESTORE、長いdecodeは未認定。prefill600目標はFalse。

次は独立した64K入力/残差GPU配置と、現在の入力順に対するGPU内K stagingの部品検証。全実行はMainが直列管理する。
