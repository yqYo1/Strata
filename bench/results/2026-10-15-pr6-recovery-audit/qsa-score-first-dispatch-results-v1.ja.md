# QSA初回投入の独立診断

小さい専用fixtureで、従来scorerを先に動かして温める前に、選択したscorerの
最初のsubmitとwaitを別々に測った。context32768 / queries256、各arm1プロセス。
UR tracingとparameter validationを併用し、実行前から `SYCL_CACHE_TRACE=2` を設定した。
これはログ有効の診断であり、定常時の性能測定とは分ける。

| scorer | 初回score submit ms | 初回score submit+wait ms | 初回top-k submit+wait ms |
|---|---:|---:|---:|
| 従来 | 87.92 | 92.08 | 303.09 |
| query8 | 87.01 | 91.87 | 306.60 |
| 4ヘッド並列query8 | 99.72 | 103.53 | 313.37 |

最初のプログラム・カーネルのcache insertionと実際のカーネル名を、
明示した開始・終了マーカーの間で確認できた。全スコア・IDは従来と全ビット一致し、
通常終了、所有権解放、新規GPU障害なし。

このfixtureでは初回scoreの差は約12msに留まり、実モデルの初回差約9.6秒の
原因を説明していない。リンクされたimage構成もモデルbinaryと異なるので、
モデル全体のJITコストが小さいとも断定できない。AOTで解消するという結論は保留する。
次は統合した実エンジンの比較と、モデル側のcache・phase情報を確認する。

環境変数の説明には[Intel公式資料](https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_cache_trace-options)を使用した。
これは可変branchの現行資料であり、インストール済みv711のソース同定ではない。
実際のruntimeのハッシュと観測出力は[各armの証拠をまとめた記録](qsa-score-first-dispatch-results-v1.json)に保存した。
