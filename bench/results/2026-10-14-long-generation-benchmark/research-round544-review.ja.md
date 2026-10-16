# R544のMain対応

2026-10-15 JST。thinkingON、long生成cap16384、agent cap8192、context262144は維持する。
usageのcompletion countはthinkingを含む。本文の長さはcontentだけで確認する。256K contextの
設定は256K実入力を検証した意味ではなく、現在のlong生成の実inputは約33K。

source reviewで指摘されたstandard chat/promptのterminal gateを直した。`[DONE]`、stop/length、
正の整数usageを揃えたresponseだけを通常の速度集計へ入れる。usageがないSSE piece countは
失敗sampleの診断としてsource label付きで残すが、client token/sの計算には使わない。
malformed SSE JSONも無視せず失敗にする。long生成には既に厳しいusage/body/engine-clock gateがある。

agentはcap到達のlengthをtruncatedと明示し、本文が出てもcomplete turn/sessionに含めない。
normal stopのbodyを次のturnへ再送する。旧suiteの完了した18agent responseは全てstopだったため、
この修正でそれらの元statusを変更しない。意味的な正しさやtask qualityまで認定するものではない。

long生成の1/2runは実行確認用のpilotとしてexit0を許す現在の設計を維持した。分類はfalseのまま、
CLIでもpilotと明記する。3run以上では全sampleがeligibleかつ各runのstrong>=60%/weak<=40%を
満たさなければexit1。Mainのfull suite controllerは3runとqualifiedを必須にする。
pilot exit0を強弱分類や正式性能比較の合格と解釈しない。

`--clients`のburst collectorは別の既存best-effort経路であり、今回の依頼のagent/32K/long生成には
使っていない。R544のterminal/usage制限を残し、burstを今回の検証結果に含めない。

old helpの8Kと生成reportのagent240の固定表記もactual capへ修正した。
CPUのfake SSE、usageなし/terminal欠落、capで切れたagentを含む19testは
[CPU receipt](benchmark-terminal-cpu-validation-v1.json)で全pass。
MTPのimmutable-source/early-exit lease guardは別のactual helper fixtureとconfigured compilerでpass、
native buildも完了した。[build receipt](draft-upload-build-v1.json)。
現時点で新nativeのGPU比較、3run long分類、実262144入力、RESTORE、品質/速度改善は未認定。
