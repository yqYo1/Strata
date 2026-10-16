# 長大prefillのggml依存と置換単位

2026-10-16、feature/sycl。vLLM・SGLangのCUDA実装と処理構成を比べるという
ユーザー指示を受け、現在の実際のSYCL経路を再確認した。CUDA/HIP backendを
実装する依頼ではなく、B570用prefillの不要な処理と良い算法を見つけるための比較。

| 処理 | 現在の実装 | ggmlとの関係／比較する点 |
| --- | --- | --- |
| GGUF・IQ weight解釈 | artifact reader、gguf_decode、iq.cpp等 | format/codebook/逆量子化の意味を保持。format対応と実行粒度は別に判断する。 |
| expert行列積 | prefill.cppからSYCL Gemm。BF16候補はvllm_xe2のnative16 backend、FP32経路はoneMKL等 | 常にggmlのgraph executorを通す構成ではない。dequant、行gather、GU、SwiGLU、downの分割を比較する。 |
| QSA長入力attention | prompt_attention.cppのjoint_matrix＋選択セルgather＋online softmax | 独自kernel。実測約34秒を占めるため、ggml libraryを外すだけではこの処理は変わらない。 |
| GDN | prefillのprojection、gates、conv、gdn_recurrence、output projection | ggml由来の算術をSYCLに移した箇所と独自のprefill用処理がある。CUDAのchunked処理と依存順・state更新を照合する。 |
| HC・PLE | native16 HCと、PLE等に残るFP32境界 | 次段normとのfusion、再変換、繰り返すmaterializationを照合する。モデル固有の意味を省略しない。 |
| CPU native experts | ggml-cpu、ggml-baseにlink | build依存の存在はlong-prefillでCPU推論している証明ではない。decode用経路を保ちながら長大prefillを分離する。 |

根拠となる現在のcall siteは `src/prefill/prefill.cpp` のGDN2305付近、QSA2349/2604付近、
expert dequant/GU/down3235付近、combine3355付近。providerは
`src/prefill/sycl/gemm.cpp`、`gemm16.cpp`、`vllm_xe2/native16.cpp`、buildの分岐は
`cmake/sycl_backend.cmake` と `CMakeLists.txt`。行番号だけで版を固定せず、
各調査specのSHA256と最終reportのimmutable source linkを使う。

SGLangのローカル公式checkout877be2974aa151a984e52822c028276c2b1e9303には
`models/qwen4_exp.py` と `layers/attention/qsa/` が実在する。vLLMのローカル
b31e9326a7d9394aab8c767f8ebe225c65594b60ではqwen4exp名のmodelは見つからず、
Qwen3.5/Next等の対応部品を調べる。この観測を最新upstream全体の不在証明にはしない。

Orcaのgpt-6-lunaを3系統、R928 QSA/indexer、R929 GDN/HC/PLE、R930 MoEで起動。
過去indexを共有し、現在のsourceとprofileをpinしたread-only調査。実装・GPU実行は禁止、
実行中のsteeringは行わない。reportの新規性・範囲レビューはnative Codex Luna-maxに委任。
Mainは測定と依存調査を並行する。6.1-sol workerは新規起動していない。

採用判断は「ggmlを全廃する」こと自体を目標にせず、長入力で時間を使う経路を
vLLM/SGLangの処理構成に基づいて書き直せるかで行う。GGUF/IQとdecodeの互換性、
262144境界、数値品質、GPUの寿命・同期を保つ。新backendの数値・小規模・fault検証を
先に行い、公式bench.shの反復した長入力PPで採用を決める。優位性を示した部品は
長大prefillで置換し、比較用FP32経路を保持する。

## ユーザーによる精度基準の明確化

同日の追加指示により、計算の基準は本番運用で使われるSGLang/vLLMの該当modelの
実dispatchを優先する。storage、累積、reduction、丸めの位置、norm定数、routingと
persistent stateまで一致する検証済みの数値仕様は受け入れる。llama.cppやFP32 oracle
の方が高精度である場合でも、それに揃えるための追加演算や一致を採用条件にはしない。
FP32は実装バグの切り分け用に保持する。従来criteriaによる失敗receiptは変更しない。
R928–930の実行中promptには介入せず、Mainの統合判断・新たな検証criteriaに適用する。
