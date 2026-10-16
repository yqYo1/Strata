# vLLM Xe2 比較ツールの Main レビュー

R816 の controller、CMake、DSO evidence wrapper、paired_gemm 全文と manifest の実行条件・依存ピンを確認した。R810 は別途全文レビュー済み。実装ワーカーは実行していない。Main が単一 measurement lock で全実行を管理する。

既存 v2 の Runtime を benchmark に重複リンクしない。fixture の Runtime は別実行ファイルのみ。入力・出力・guard と host copy destination の寿命を checked drain まで保持する。比較は同じ入力・同じ queue の5経路、3独立プロセス/geometry/format、各3 timed submits。全出力を独立 dyadic rank2 oracle で検証し、readback/oracle は時間外。モデル品質や実効XMXをこの比較だけで認定しない。

最初の build-v1 は過去の build-only receipt に存在しない GPU fault フィールドを要求し、preflight FAIL。owned command は0件、コンパイル/GPU未実行。元の結果は own-vllm-qualification-preflight-failure-v1.json に保存。Main は該当する正確な hash-pinned v2 build-only record のみ別スキーマとして検証するよう修正した。68-case GPU receipt と最新GPU実行記録、今回の fresh fault window は従来通り必須。失敗をPASSに変更していない。修正後 manifest SHA7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58。新 private root の build-v2 を実行中。

R810 AOT compile の実際の flag acceptance、数値・denormal・chain・全geometry、ISA、性能、モデル品質、物理256Kはこのレビュー時点では未検証。旧68caseの基準は変更しない。
