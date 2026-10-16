# Main: native16処理内訳

同一6354f0b4 binary、officialbench actual39262/39261batched、C8192/R_GPU32768/context=resident65536、thinkingON/24output。FP32→vLLMBF16→typedQSAの3独立診断processを直列実行、全48層/5chunks、全て正常回収/fault0/dumpなし。clean速度とは別cohort。

| 診断区間 (秒) | FP32 | BF16 | BF16+QSA |
| --- | --- | --- | --- |
| wrapper |104.194|98.491|100.987|
| MoE |22.1048|20.2019|20.1739|
| QSA全体 |39.7081|39.6936|42.2598|
| attention本体 |33.9643|33.9610|36.2963|
| QSA projections |0.919847|0.919497|1.20636|
| upload |8.38302|8.39997|8.39453|
| GDN |9.04151|9.24649|9.24634|

これらはhost submission gapを含む非排他的queue区間であり、合計して100%のGPU稼働率やAmdahl上限にしない。attentionが最大の測定区間で、typedQSA版の追加約2.3秒と投影約0.3秒がwholepath低下を主に説明する。QSA追加によるhidden16bit化が自動的に高速化するとはしない。

CPUのCLOCK_THREAD_CPUTIME_IDではFP32のmain99.096秒のうちexplicitwait77.685秒、wait外21.411秒。BF16は93.405/74.282/19.124秒、QSAは95.819/76.626/19.193秒。1core飽和の大半は完了polling。wait外にもruntime内の未計測waitを含む可能性があり、全部をCPU算術と呼ばない。BF16のsourcecopy4.813秒、uploadsubmit4.929秒、Gemm投入2.400秒、hostgroup0.085秒。routeexchange区間53.131秒のうちexplicitwait外wallは0.00459秒で、53秒全体をroute転送やCPU振分のコストと呼ばない。

BF16では各方向のresidual host転送要求量が12,499,804,160から6,249,902,080Bへ半減し、重みupload50,292,326,400Bは同じ。CPU groupingをGPUに移す価値は同期依存経路と合わせて評価する必要がある。0.085秒だけを見て同期の影響が無いとも、routeexchange53秒だけを見てCPU転送が支配的とも結論しない。

次は自然文四arm数値screen。typedattention内のldexp/range処理、SLMとmessage配置は実際の原因候補であって未証明。R844のexact normal exponent fastpathは別のsource-onlyproposalとして依頼し、元FP32/typedOFF経路と範囲・NaN/Inf/subnormal契約を保持する。適用/速度採用はMainがテスト後に決める。
