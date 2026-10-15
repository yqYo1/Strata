# oneDNN BF16 比較実装の Main ソースレビュー

対象は R804 の独立した `OneDnnFaithfulBf16`。実装 CPP、公開ヘッダー、CMake、全 fixture、README、installed/source evidence、manifest を Main が通読した。現時点のビルド・ライブラリロード・GPU実行・速度・モデル品質はすべて UNRUN。製品の Gemm ABI・既存68ケース・prefill dispatch は変更していない。

- BF16/BF16→BF16 の公開 matmul、strict FP32 accumulation、faithful destination conversion を別 API として扱う。既存 exact-RNE API のテストを緩めない。混合 FP16 入力/BF16 出力はこの候補の対象外。
- 呼び出し元の Intel GPU/device/context/in-order queue を明示的に渡す。device0 の選択、別 queue、CPU数値フォールバック、環境変更、unsupported 時のリトライはない。実際の engine/stream/descriptor/implementation を取得して判定する。
- extent/overflow/alignment/overlap/registered-device/context を enqueue 前に検証。32 entry/512 pending owner、128MiB の共有 scratch と descriptor が報告する primitive-owned bytes の上限。完了を確認してから eviction/resize/release、partial enqueue 例外では owner 保持・poison、破棄時の不確実な drain は fail-stop。
- strict attrs や GPU engine は実際の XMX・CPU演算不使用・内部FP32一時領域ゼロを証明しない。選択された primitive と実行を別途確認する。公開合計 scratch の内訳は未確定。
- インストール済み oneDNN 3.11.4 の `DNNL_WITH_LEVEL_ZERO` が未定義であることだけで排除しない。実際の caller queue で初期化を試し、APIの失敗は失敗として記録する。libsycl.so.9 ABI/ロード依存の一致も未検証。
- fixture は dyadic FP64 oracle、境界/stride/入力不変、有限値・cancellation・midpoint・subnormal・NaN/Inf・16bit chain、cache/queue/close を確認する設計。全て未実行。主 API の faithful 契約と意図した denormal 契約が実ハードウェアで成立するかはテスト結果で判断する。

追加する実行側の対策は、例外時に fixture の host vector/USM を解放するより先にチェック付き drain を実行する guard。既存 Allocation::release も Runtime drain を行うため、これは新しいGPU障害原因が見つかったという報告ではない。元の source proposal を変更せず、私有ビルド用変換の差分/hashを記録する。lifecycle/chain は GEMM owner の方が Buffer より後に生成されている点も確認した。

Main の自然文32K 3arm品質測定が完全終了するまで、この候補のビルドやテストを開始しない。R809 が実行しない controller のソースのみを準備中。Main がその全文を確認し、小さな traced smoke を先に実行する。性能採用、長文品質採用、物理262144/RESTORE 検証は別の未完了ゲート。
