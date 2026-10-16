## 判定

v2はソース上の安全性修正を確認できたため、段階的な資格確認へ進める案として受理。コンパイル、実行、性能、障害、推論採用の実績はない。v1の停止記録とレビューはそのまま保持する。

新入口は、明示されたin-order queueに対し、state/h/gate/beta/z/gamma/y16の全7領域が同じcontext/deviceの生存中Device Allocation内に収まることを確認してから、packed hのQ/K/Vオフセット計算とenqueueを行う。FP32の再帰、state、reduction、barrierとFP16丸めは維持し、削るのは不要なFP32出力storeだけ。選択子は既定OFFでnative16時だけ有効、短いprefillとdecodeは既存呼び出しを通る。

fixtureはstate/FP16出力のbit一致、反復・分割実行、guard、7領域の不足、host pointer、queue domain等を扱う。一方、enqueue数の検証はない。profileとCPU作業の終了後、traced smokeでT=1は候補2回、T=17は4回、拒否ケースは0回の候補launchであることを数え、fault確認を行う。通過後にcomponentへ進む。32-bitも対象ならspan乗算に明示的なoverflow検査を追加する。

manifest、patch、fixture、4つのbase source、3つの安全性sourceのSHA256は記載値と一致。提案後の4 source hashはmanifest記載値で、patchを再適用しての独立再計算はしていない。研究indexへの最小追記案はJSON内に記録した。新規調査は不要。
