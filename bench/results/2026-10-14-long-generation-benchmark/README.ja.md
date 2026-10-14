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
