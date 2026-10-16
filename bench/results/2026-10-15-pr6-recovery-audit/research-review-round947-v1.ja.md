R945はdecision supportとして採択（1,595語、予算内）。R=2 adjacent-query unionとN64 stagingは限定比較案。SGLangの固定commitでQのscale後BF16丸めとFP32オンライン状態、vLLMのpost-dot scalingとの差、Intel資料のSLM/register occupancy caveatを照合した。vLLM issueの87–94% overlapと修正後1.50x kernel/2.9% TTFTは作者報告のCUDA/GB10値で、Arcへ外挿しない。

提供された39,262-token全prefill中央値393.4/406.1/445.4 tok/s、attention+pack 21.5123秒などはタイル性能を帰属しない。queue区間は非排他でPMU/XMX/物理帯域の結果はない。Mainのcoalesced-K試行はhost preflightで止まり、compile/GPU実行なし。selector重複とcomponent timingで候補を識別し、それでも有限の未解決文献問が残るまで次の調査は非適格。

R946はknowledge-collection割当だったが、agent_readinessのagent-update-promptで失敗。worker turnもreportもなく品質評価なし。releaseで該当terminalのclose/captureを確認し、研究reportとして索引しない。R945 settlementにはworker_doneがある一方、acknowledgedはnull。Mainが実際のlifecycle ACK状態を処理する。
