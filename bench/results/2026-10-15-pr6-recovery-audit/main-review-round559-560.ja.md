# PR #6復旧調査の初回レビュー

2026-10-15 JST。thinking有効のbench拡張検証が正常終了した後の読み取り調査。
調査用Luna 2件を読んだ。旧実装のGPU再実行、候補の実装・採用はまだ行っていない。

## 主エージェントで照合した事実

現在の`cmake/sycl_backend.cmake`は`src/prefill/prefill.cpp`を直接コンパイルする。
このrootソースとheaderに`run_layer_major`と`STRATA_PREFILL_LAYER_MAJOR`はない。
PR #6の実装は`f263d87d44aa671d056c33e6c6b6b0635fc25b8c`の
`sycl/src/prefill/prefill.cpp`にあり、現在のビルド対象ではない。
Git履歴に残っていることと、現在の実行に統合されていることは区別する。

`d0c5dda0dc2e8d82458c4b0ff31c4cef44c40b32:sycl/src/prefill/prefill.cpp`
の`run_layer_major`を直接読んだ。旧ルートはfull single-GPU compact FP16、
全K/V residentを要求し、各層の全expertを一度ロードして全prompt chunkを処理する。
現在のstreamed KVにそのまま使うことはできない。
ローカルRestoreはcompute/copy/stagerの完了を待ってポインタとcallbackを戻すが、
完了を確認できないエラー時の所有権保持は、現在のruntime契約に沿って再点検する必要がある。
現在のdraft upload寿命修正を旧ソースで置き換えない。これらは復旧前のソース上の懸念で、
過去のGPU障害の原因がPR #6だと確定したものではない。

保存済み`../2026-10-14-performance-regression-audit/historical-layer-major/summary.json`
のlabelと値を照合した。32,768入力、8,192chunk、65,538context、K8V8全量resident、
timed MTPなしの3組で、baseline中央値453.2935、layer-major中央値645.848 t/s。
これは当forkの実測で、現在の256K設定・MTPありの測定との直接比較には使わない。
この比較はcompactとlayer-majorを同時に切り替えており、効果を片方だけに帰属させない。

## 調査レポートの訂正・限定

R560の表のpair 2は、実行順を列の意味と取り違えて速度の列が逆になっている。
正しくはbaseline **453.2935**、layer-major **646.4646** t/s。
baseline receipt SHA256は`fd03dc328243042cd566e0fa68e62748a3f683685ae45bb513a2cd8de8013456`、
layer-majorは`e5a608704cfc63a91015c84bcde55b6614207deb5397d837ba710d8a64e5ac57`。
元レポートのreceipt hashの列は正しく、速度の列だけを訂正する。
元レポートは残し、本レビューと元JSONを正しい対応の参照にする。中央値の記載は正しい。

R560の118.5 t/sは過去の別条件のroot測定であり、今回閉じたsuiteの最新値ではない。
最新値は`../2026-10-14-long-generation-benchmark/upload-fix-suite-final.ja.md`に保存した。
公開Codexログの新たな照合、streamed KVを含む全呼び出し契約の詳細調査は継続中。
両レポートの提案を実測済みの改善や256K安全性の証明として扱わない。

## 次の実作業

旧PR #6と後続の当fork実装を現在のrootの部品に対応付ける。公開セッション記録と設定を照合し、
まずcompact workspace、QSA layout、処理順について失われた選択肢と採用条件を特定する。
寿命・同期・範囲・streamed KVの契約を確認してから候補を実装し、主エージェント管理で
CPU検証、build、診断付き小規模GPU確認、32K以上/agent/長いthinking生成の比較を順に行う。
prefillとdecodeは別々に採否を決める。実入力262144とRESTOREは引き続き未検証。
