# 資源切り分けの検証コードとPR6の小さな実機確認

資源別の数値と解釈は `hardware-resource-followup-v5` を参照する。
GPU内の演算器・VRAM/cache・SLM・同期のどれが不足しているかはまだ未確定である。

R914の案をnative Luna-max R916でレビュー後、Orca Sol R919が独立のcomponent診断を
納品し、Mainが全425行・CMake・比較器をレビューした。Mainは未計測の比較ごとに出力を
poisonする補正を加え、`tests/sycl/qsa_resource_control.cpp` とmanual opt-in targetへ
統合した。既存kernel/fixtureは変更していない。まだbuild・GPU実行・採用はない。

- 同じK/V値・計算量・全allocationを保ち、query間で同じ約2.17MBを再利用する場合と
  query別に複製した約277/554MBを使う場合を比べる。cache/internal-memoryへの感度を
  調べるもので、物理VRAM trafficや帯域飽和率を直接測ったとは言わない。
- 同じ行・同じID・同じin-order queueで、128/256queryを一括または1/8/32queryずつ
  投入する。enqueue/wait/全体のhost wallと発行thread CPUを分ける。投入とgrid schedulingの
  影響を含み、GPU-only時間とは呼ばない。

最初は小さいFP64 oracle・全出力一致・guard・入力不変・fault/owned終了で資格付けし、
cleanな別processを各arm3回以上で比較する。model PPの代用にはしない。
却下済みhead8+4/KMapping2は再実施しない。知識収集R918はレビューR920に委任してあり、
Mainは本文を読まず、実機telemetryの対応を確認できたという主張もしない。

別ブランチのPR6復元では、旧DPCTの型解決・memory helper kernel名のコンパイル互換を
原本を変えないprivate headerで処理した。namespace宣言だけの追加が同じprovider TUを
通すcompile-only対照もPASS。新しいmodel-free binaryはbuild/link/ELF確認後、Mainが
実環境とmapsを3箇所で照合して、別process4ケースを実行した。

`own-pr6-provider-contract-v1.json` はPASS/complete。128Bコピーと32セル整数kernel、
default/owned queue、受理済みcopy後の模擬HOST例外、owned破棄後default動作を確認した。
12回の明示hold、全exit0・所有終了、新規fault0、dumpなし。event pending観測は0で、
queue pendingの観測がある。模擬HOST例外を実provider障害や非空async87deliveryの
適格化へ拡大しない。model/旧CLI実行はなく、600再現・品質/state/physical262144/RESTOREは
未認定。source commit ae8f85f1のbuild v3は59/113で別のunsigned64 shuffle-up型解決エラーによりexit1。
所有終了・GPU実行なし。別Sol R921に限定した修正を委任し、Mainは現行の資源比較の検証へ進む。

方法は `docs/RESOURCE_PROFILING.md`。大きい既存timelineの現在consumer/budget/reviewは
変更していない。成功API traceと閉じたbuild重複出力は、要点を照合後に保持規則で退役する。
