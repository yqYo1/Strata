# 上流benchmarkへ長い生成とagent結果の評価を追加

2026-10-14、Arc B570 / Ryzen5600X / 128GiB。ベースclientはNiko1221/Strata
7695859a、依存3clientsは無変更。mainがbench.sh/strata_bench.pyと新しいworkload helperを
実装し、GPU不要のHTTP/usage/agent-history検査4件を通した。推論sourceは変更していない。

標準suiteは従来どおりagent6turnと32K promptを含む。追加でthinking込みの実生成
1024..1152 tokens、実入力32768以上のregular-code / constraint-reasoningを測る。
MTP強/弱は仮説で、3回以上・各256以上offered・受理率各60%以上/40%以下の実測が必要。
生成の数学・code品質やMTPそのものの速度効果はこのspeed suiteから認定しない。

[手順](../../../docs/BENCHMARKS.md)、[上流source pin](upstream-source.json)、
[並行Luna調査](research-index.json)。調査はFINALまで介入せずMainが実装を並行し、
全test/GPUを直列管理した。プロジェクト専用スキルの計測条件も更新した。

```sh
python3 bench/results/2026-10-14-long-generation-benchmark/check_extended.py screen
python3 bench/results/2026-10-14-long-generation-benchmark/check_extended.py suite
```

前者は2候補各1回の長さ・受理率screeningであり分類確認ではない。後者は標準全suite・
32K3回・agent3sessions・追加2workloads各3回。source/binary/numerical gateを確認し、
exclusive lock、owned process、期限/RSS/log budget、正常終了とfresh fault検査を使う。
各fresh rootは一度だけ作成し、閉じた結果を上書き・自動再試行しない。

現在68da0262 native、MMQ0/R8、compact1、chunk4096、context262144、MTP4/splitOFF。
初回screeningとfull suiteはこの記述時点では未実行。旧636の正式標準suiteに含まれる
agent結果も保存済みで、これまで報告が32K値へ偏っていたため今後は必ず並記する。

## v1実測

各1回は正常終了、全owned child回収、fresh fault0、dumpなし、boot不変。
regular-code実33215入力/1152生成、7.7t/s、739/1239（59.64%）。
constraint-reasoning実33322入力/1152生成、7.1t/s、717/1311（54.69%）。
両方content0文字で全量thinkingだった。**MTP強/弱は未確認**で元結果を保持する。
v2はthinkingOFFのcodeと多様な日本語散文でworkloadを分け、同じgreedy/受理率基準で
測る。数学的品質・MTP速度効果・推論採用には数えない。[全個別値](screen-v1.json)。

ユーザー修正により**v2のthinkingOFF案は未実行で撤回**した。現行v3はthinkingONを
維持し、max output8192、実生成1024以上、本文512文字以上を確認する。1Kは上限ではない。
context262144には約33K入力と8K生成の余地があり、満杯になるまでの検証とは分ける。
全量thinkingで終了した場合は未完了として保存し、有限の出力/context予算を再検討する。
GPU nativeは同じ68da0262、greedy、MTP4。新しいsampling kernel/精度/queue経路は使わない。

v3のCPU検査5件とskill validationを通し、source76e4bf7eをcommit/push後に
`check_extended.py screen`を開始した。出力rootは
`~/.local/state/strata-sycl/post-reboot-tuning-20261007/main-primary-long-generation-screen-root-v3`。
この境界では実モデル確認は進行中で、MTP分類・本文出力・正常終了/fault検査は未確定。
Owner期限9300s、raw text64MiB、RSS88GiB、job1GiB、reset/service変更なし。
閉じた後に全個別値と元receiptを抽出し、各3回のfull suiteを同じownerで直列に評価する。
その検証完了後に[next-work](next-work.json)のPR #6・Codex元ログ・安全性点検へ戻る。
