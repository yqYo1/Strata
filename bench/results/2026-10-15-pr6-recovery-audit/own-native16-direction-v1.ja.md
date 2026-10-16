# 16bitのカーネル間入出力への変更

2026-10-16のユーザー指示に従い、保存用RだけのFP16化から、長大prefillの
hidden stateを16bitで直接生産・消費する実装へ優先順位を変更する。
現行MMQ=0経路は独自SYCLとoneMKLだが、FP32のカーネル間バッファが多い。
llama.cpp全体の設計に原因を断定しない。

参照はghq管理のSGLang、vLLM、vllm-xpu-kernels、Intel llm-scaler。
vllm-xpu-kernelsのMoE dtype policyは16bitのA/B/OutputとFP32の
Accumulator/ComputeEpilogueを分離している。具体的なライセンスはファイルごとに
確認し、XeStrata由来の実装・テストは使用しない。

MainはHC残差・正規化・活性化・MoEの接続を実装する。Orcaの別の6.1-solは
16bit入出力・FP32積和のGEMMをソース案として実装する。Luna2件は別々に
SGLang/vLLMの型契約、Intel実装の利用可能なAPIを調査する。実行中の研究者へ
介入せず、Mainのみがパッチをレビューし、ビルド・テスト・GPU測定を管理する。
R781のFP32精度を保持するattention案は返却済み、未採用・未ビルド・未測定。

旧FP32基準a/bは各78443行の有限NLLを取得し、正常終了・新規GPU障害0。
平均NLLは4.823238516902749と4.823259333754754。2行だけ異なり、
最大絶対差0.8948270054323606。この孤立した差の原因は未解決であり、
16bit化の許容誤差とはみなさない。生NLLとrecipe/closed/faults/metaを各基準で
保持し、新しい比較器の入力とする。固定単語列なので一般的な言語・タスク品質の
証明ではなく、速度測定にも使用しない。

旧storage-only計画のfp16-matched / fp16-all-gpuは**未実行**。
今回の指示で先送りする。新コード比較では意図的にsource/binaryが変わるため、
新リビジョン内のFP32対照も用意する。旧比較器の同一source条件や結果を書き換えない。

まだ16bit経路の数値品質、速度、物理262144コンテキスト、RESTOREは未検証。
short/decodeのFP32経路を保持する。600tok/sの反復達成、次いで1000tok/sの
長大prefill目標も継続する。
