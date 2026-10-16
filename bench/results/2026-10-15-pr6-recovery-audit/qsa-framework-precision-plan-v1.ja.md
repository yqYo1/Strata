# SGLang基準のQSA算術候補

2026-10-16。MainはR931の短いreviewを読み、許可されたR928の13–17/21–29行のみ取得。
SGLang本体の該当sourceも独立に確認した。R928は旧精度基準でhi/lo保持を要求しており、
その採用条件をR931-C1/C5で訂正する。新しいユーザー指示を実行中workerへsteerしていない。

## 対応する実行分岐

公式SGLang877be2974aa151a984e52822c028276c2b1e9303の
`qwen_sparse_attn_backend.py::forward_extend`。通常CUDA extendでは
`_try_fused_kv_attention`はdecode/target-verify以外で戻り、speculative paged経路も
通常extendには入らない。no-prefix時は `sparse_gqa_fwd_interface_triton`、prefix時は
Q dtypeへのKV pack後 `sparse_gqa_fwd_interface_triton_ck` を呼ぶ。
非CUDA用 `qsa_sparse_attention_reference` の全FP32 operand計算を、CUDA prefillの
数値仕様として使わない。これはsourceで辿った分岐であり、SGLang CUDAを本機で
実行した記録ではない。設定BF16を候補の明示的な参照契約とする。

`qsa/sparse_attn.py` の両prefill kernelは、Qのscale/log2変換後にinput dtypeへ丸め、
Q/Kのmatrix積、FP32 online max・normalizer・V累積、input dtypeへ丸めたP/Vのmatrix積、
output dtypeへ保存という構成。したがってFP32累積を維持するためにQ/Pのhi/lo二重積は不要。
indexerのTileLang有無やscore dtypeは別の経路・契約として扱う。

## Mainが実装する候補

候補は、既存typed BF16 Q/Oに対する**BF16 matrix operandの直接経路**。
K8/V8 cacheのコードとper-64 FP16 scaleを、選択した32-cell tileの中だけで
FP32乗算してBF16へ丸める。weight/KVファイルや選択セルを変えず、全prefixのKV packも
追加しない。QはBF16、PもBF16、matrix累積・online max・normalizer・持越しaccumulatorは
FP32。既存のFP16 hi/lo pathと、BF16 Q/Oだがscale-group別にFP16 matrix積を行う
typed pathは比較用に保持する。

これは既存typed pathを名前だけ変えて再測定する提案ではない。既存9-run比較では
FP32中央値372.5、vLLM/BF16 hidden394.9、typed-QSA384.8tok/sでtyped-QSAを不採用。
この既存結果は `own-vllm-clean-summary-v1.json` にあり、R928の「未測定」を一般化しない。
今回はKV tileをBF16 operandへdecodeする位置、per-group後段scale処理とP tileの個数を
変える。必須のscale適用を省略するものではない。

候補のpayload SLMはQ8192＋KV16384＋P1024＋scores2048＋max/sum/alpha192＋
rows256＝28096Bを基本にする。実際のflags/scale scratch等を加えて記録する。
既存typedの32336B、hi/loの44496Bより小さくできる見通しだが、occupancy増や速度は
未実測。query×KV-headのtile形状と12実head＋4paddingは維持し、棄却済みheads8を繰り返さない。
BF16 matrix tuple/subgroup/SLM capabilityを確認してから小さいnative gateを行う。

検証は同じQ/quantized-KV/scales/selected IDsに対するBF16 operand＋FP32累積の
参照を新しく定義し、scale/丸め/softmax/missing-page/空行/非有限/境界/lifetimeを調べる。
旧FP32 oracleへのbit一致は要求しない。候補を本番へ入れる前に小規模debug、component、
モデルのstate/数値検証、反復した公式bench.shの32K以上PPを行う。現在は設計段階で、
実装・性能・品質・262144/RESTOREの資格付け完了とは扱わない。

## selector問題は分離して解決する

SGLangの512 blockを全展開＋causal tailと、Strataのweighted-cell selectionには
2052–2054等で幅の差がある。これは単なる並べ方の差ではない。2048–2055の全tail残数、
境界tie、262144でset/cardinalityを確認する。top-kの出力順はSGLangの契約ではない。
attentionの精度候補は同一selected IDsで評価できるため、selector変更を混ぜない。
本番のselectorを変更する前には、モデル仕様とprefill/decode双方のstate整合を別途確認する。

## 同日実装と最初の測定

`qsa_bf16_attention.cpp` と独立host oracleを実装。小規模2case/12 kernel提出と
境界10case/28 kernel提出は正常終了、新fault無し、所有終了。境界には実体262144-cell
KV poolを含むが、モデル全体の256K context/RESTORE認定とは区別する。
明示SLM payloadはflagsを含め **28168B**。この値だけではoccupancyを断定しない。

`qsa-bf16-matched-bench-summary-v1.json` は同一Q/K8V8/scales/IDと確保量で
8192query/2051selected cellsを各3process×10回測定。process中央値の中央値は
FP32 **632.1835525ms**、旧typed **679.6906295ms**、新BF16 **627.3464365ms**。
新BF16は旧typed比7.701%短縮、FP32比0.765%短縮。
単体改善は小さく、全体PP向上・採用を意味しない。新経路を明示的な手動CMake
ターゲットから検証できるようにするが、production caller/defaultは切り替えていない。

次はタイル内の同期コストを切り分ける。現実装はK scale異常、live-row、V scale異常の
workgroup collectiveを各32-cell tileで行う。これらは最終診断にのみ必要なので、
局所bitを最後に集約する候補で、算術・選択・BF16 roundingを変えず同期頻度を減らせるかを
調べる。速度は未測定で、source上の頻度だけから支配的資源とは断定しない。
