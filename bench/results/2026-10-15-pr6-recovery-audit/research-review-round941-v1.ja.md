# R941 調査レビュー

- R940は指定SHA `54d799f7…806a1d` と一致する1,763語の判断支援レポート。R928/R931、R935の一般的なQSA算術比較と一部重なるが、今回のMain診断が示したBF16確率境界の2点と、その後の段階別ゲートを扱う点が新しい。
- 旧receiptは10/24,576の不合格を維持する。合成fixtureの2セルでBF16 Pが境界をまたぎ、同じGPU exp2引数に対するhost exp2ではcast差がない。係数再生は256次元の差を説明するが、新しい合格、pack固有の差、モデルPP/RESTORE/物理262,144文脈、品質、速度や採用の証明ではない。
- 算術配置は正確に区別されている。SGLangはQをscale×log2(e)倍してQ dtypeへ丸めてからQK、vLLM v0.30.0はBF16 QK後にFP32 scoreをscaleする。vLLMは代替契約として扱う。vLLMはsingle/splitの両経路ともgate前にattention出力をBF16へ丸める。実行時の分岐、JIT lowering、split数、実引数はsource記述から確定しない。
- 段階ゲートは構造・範囲・同じ入力/IDでの比較、BF16 midpointを許す条件、normalizer/PVまでの誤差伝播、別枠のtask/性能検証を含み、旧FP64/llama.cpp値との完全一致や同じtoken列を要求しない。`gamma_(D-1)`は測定済み許容値ではなく、演算・下/上溢れ条件を確認して使う初期モデル。
- 「scoreが等しい」はreceiptの表示精度で等しいという範囲に限定する。上記の実行時・数値モデル限界を保ったうえで本文31–43、45–47行をMainが参照可能。新しい調査は候補のJIT/分岐/段階receiptと同契約比較をMainが閉じ、具体的な残課題が出るまでschedulableではない。
- launchはgpt-6-lunaでturn開始を観測、releaseはterminal終了・transcript保存。別個のworker_done receiptは提示物に無いため、内容のSHAとレビューを独立に確認した。
