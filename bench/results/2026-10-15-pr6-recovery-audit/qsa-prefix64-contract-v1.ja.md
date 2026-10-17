# QSA64セル候補の検証契約

Mainが既存Strataの独立実装をパラメータ化した候補。XeStrata由来コードは使用しない。
元の32セル経路を残し、候補は単一query/KVhead、16 padded head、WG128/SG16、
QK8/PV8 subgroup、score65、SLM47936バイトを使用する。mask、ページ境界、入力/出力の
allocation領域と寿命、in-order queue、checked cleanup契約は維持する。

64セルごとのtile最大値を使ってBF16確率へ丸めるため、32セル版とのbit一致は要求しない。
GPUとは独立したCPUの密なdot/PV評価を使い、64セルのonline max/normalizerと確率丸めを
明示的に計算する。Q/K/V/P/outputはBF16、局所accumulatorとonline stateはFP32。
数値基準は従来と同じ1 BF16 ULP + 2e-6とし、失敗はそのまま記録する。これは
SGLang/vLLM型のstorage/accumulation境界に基づくcomponent検証で、実モデル品質は別に必要。

最初にパラメータ化後の32セル実装で既存smokeを実行し、次に64セルsmoke、componentへ
進む。componentはselector幅1/17/31/32/33/63/64/65/127/128/129/2051、最終partial、
未map/不正page、INT32両端の不正cell、NaN/Inf、空selection、alias/domain/extent拒否、
完全なpack内容、全出力guard、入力不変、反復の決定性を確認する。
物理262144セルのcomponentはmodelで262144を消費する検証やRESTOREの代替ではない。

小さい初回GPU実行はUR tracing+parameter validationを有効にし、実行環境、ELF、
process identityをruntime前/初期化後/cleanup後に確認する。GPU再設定やリセットは行わない。
成功後のみ同一binary・同一input・pack込みで32/64のkernel時間を比較する。host待ち時間を
hardwarebusyと読み替えない。公式bench.shの長文モデル比較はcomponent候補の選定後に行う。
