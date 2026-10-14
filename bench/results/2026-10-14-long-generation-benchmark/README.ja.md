# 上流benchmarkへ長い生成とagent結果の評価を追加

2026-10-14、Arc B570 / Ryzen5600X / 128GiB。ベースclientはNiko1221/Strata
7695859a、依存3clientsは無変更。mainがbench.sh/strata_bench.pyと新しいworkload helperを
実装し、GPU不要のHTTP/usage/agent-history検査4件を通した。推論sourceは変更していない。

標準suiteは従来どおりagent6turnと32K promptを含む。当初v1はthinking込みの実生成
1024..1152 tokensを設定した。現行v4は1024..16384、実入力32768以上の
regular-code / varied-proseを測る。
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

## v3終了結果とCPU上限修正

v3は失敗として閉じた。regular-codeは実33214入力/3025生成、PP172.9t/s、TG9.3t/s、
受理2173/2559=84.916%、本文5578文字、思考3063文字、正常stop/usageあり。
1回の候補結果でありMTP強弱の分類や速度効果を認定しない。
varied-proseは思考27138文字、本文0、正常終了/usage/engine clocks/countersがなく失敗。
client fallback4.9t/sとstream pieces7206は性能/実token数に採用しない。
[全個別値](screen-v3.json)、[終了診断](screen-v3-stop-diagnostic.json)、元receipt gzipを保存した。

nativeは2320秒程度でexit-24(SIGXCPU)、serverは終了0。owned children全回収/session空、
fresh fault0、dumpなし、boot不変。原因はMainがRLIMIT_CPUをwallと同じ9300にしたことで、
この上限は複数threadのCPU時間をprocess内で合計する。GPU障害とは分類しない。
元FAILED/INCOMPLETEのstatusを変えず、今後はwall×CPU affinity countを有限のprocess上限にする。
壁時計、RSS、log、artifact期限は維持し、nativeが実際に継承した/proc/PID/limitsも保存する。
これはsession全体のCPU合計上限ではない。

v4は同じ課題、thinkingON/xhighを明示、出力上限16384、最小実生成1024、本文512文字を使う。
実際のtokenizer template12827f24のlines58..69で既定xhighを確認した。
1Kを上限にせず、context262144と約33K入力には思考と本文の余地がある。
v3と生成枠/明示requestが異なるため同一workloadのengine改善とは比較しない。
agentの--thinkも各turn8192へ広げ、実際のanswer/reasoningを履歴へ戻し、完走6turnだけ集計する。

Solの[元提案](evidence-proposal-v1/report.md)をMainが閉鎖確認後に適用・レビューし、
並行pending map、開始時body hash、post-suite failureのterminal記録、journal検証を追加した。
元提案は未適用だった時点のartifactとして変更せず残す。推論source/binary68daは変更していない。
新recipeはroot末尾-evidence-v2で旧rootを上書きしない。追加LunaR529/R530の原文とMain判断は
research-indexへ保存。GPUに介入せずMainは並行実装/検査を続けた。

MainのCPU検査は既存5件+新11件=16件PASS。外部SIGKILLで後続要求を止めても先の測定と
pending hashが残ること、8並行要求のchainとpending、改ざん拒否、suite後/出力時failureの
complete=false、実際のthinking/answer履歴、部分sessionの除外、no-MTP speedの区別を確認。
Python/shell syntaxもPASS。この境界ではv4実モデルscreen/full suiteは未実行。
PR#6/Codex元ログの性能・安全性回復はbenchmark検証境界の後へ続ける。


v4 screenはsource605b80f8で開始した。rootは
`~/.local/state/strata-sycl/post-reboot-tuning-20261007/main-primary-long-generation-screen-root-v4-evidence-v2`。
Mainは実native PID334431/start32864521の/proc/limitsからCPU soft111600/hard111601秒を確認。
long-generation/regular-codeの実送信開始rowはthinking=true/max_tokens16384、入力実33214。
[起動時観測](launch-v4.json)はlive確認のみで終了/fault/性能合格ではない。
元の標準word-count校正2要求は1生成・thinkingOFFで、速度結果には数えない。
全測定workloadとagent条件は--thinkでON。終了後にjournal chain、正常native/server exit、
全個別結果、fresh faultを検査し、3回の標準32K/agent/追加生成suiteへ進む。
active jobのclient/engine/recipe sourceを変えず、同時test/build/cleanupはしない。


v4最初のregular-code要求は正常stop/DONE/usageで実33214入力/3346生成、PP156.1t/s
(212804.9ms)、TG5.3t/s(637080ms)、受理2382/2895=82.280%、本文5499文字、思考4055文字。
[元durable row](screen-v4-first-request.json)をそのまま保持。現在owned screenはまだactiveで、
日本語要求の終了、native正常終了/fault、強弱各3回は未確認。v3高側の9.3t/sと大きく異なるが、
各1回・出力履歴/request/生成枠が異なるためengine退行の因果判定をしていない。

[1回のthread配置観測](thread-placement-v4.json)では主threadがcore0、CPU pool5threadが
core1..5（物理core6つ、SMT sibling6..11）で、後発3threadが全てcore0だった。
Stager::initの既定worker3本は親affinityを継承し、SessionLoopScratch.initはserveのprompt
初期化より前に主をcore0へ固定する。後発threadの役割はruntime名/stackで未確定だが、
ソースと配置が一致する具体的なprefill候補。累積CPU ticksをdecode専用serviceや原因証明にしない。
R533は実起動順/配置、R534は既存Stagerの並行event/source/buffer契約を調査し、別Solは
原allowed maskを渡す未適用案を準備する。全実行はMainで直列管理し、live sourceは固定。
