# BF16 QSA dense16/VNNI配置のモデル全体比較

同一バイナリ、公式bench.shのprefill-only拡張、thinking ON/生成上限24、実入力39262 token、context/resident65536、chunk8192、GPU残差32768行＋RAM。各arm3独立processを交互順に測定。

|経路|個別prefill tok/s|中央値|標本SD|
|---|---|---:|---:|
|基準score33 row-major|[459.1, 463.5, 450.8]|459.1|6.449|
|候補score33 dense16/VNNI|[459.3, 469.4, 402.0]|459.3|36.350|

観測された中央値の差はスループット+0.044%、prompt時間-0.044%。同じ入力body hash、全6回正常終了/ownership closed/新規GPU faultゼロ。部品測定6.68%の短縮をprefill全体の改善率として扱わない。

モデル全体での速度採用は保留。最後の各armの実行時間帯は、ユーザー依頼のOrca/OpenCode機能テストと重なった。重複はnative processの開始・終了とworkerのdispatch・完了時刻から確認しており、prefill内の厳密な重複時間や速度低下の原因は未特定。cohortの元receiptにある「No parallel test」はこの例外を含まないため、concurrency addendumで訂正する。全6測定を残し、遅いサンプルだけの差し替えや原因の断定をしない。候補のばらつきと微小な中央値の差からも優位とは言えない。

BF16 Q/K/V/P/出力とFP32の累積・max/normalizer、32セルの逐次更新順は同じ。all48合成129tokenのNLL/3箇所全語彙logitの比較は別のmodel-smoke receiptを参照。精度の採用基準は同系モデルのSGLang/vLLM契約であり、llama.cppとの完全一致ではない。一般品質、full-model256K、保存/RESTORE、長いdecodeは未認定。prefill600目標はFalse。

次は現在のモデルprofileと実際のMoE形状分布を取得し、別の独立した比較で速度を再確認する。64K入力/残差GPU配置も未測定。全実行はMainが直列管理し、機能テストも速度測定と重ねない。
