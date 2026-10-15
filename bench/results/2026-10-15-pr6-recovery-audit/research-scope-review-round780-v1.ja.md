# R780 scope review

2026-10-16。R777〜R779のsettled/releasedを確認し、report・spec・receiptのhashを固定した。レビューは範囲、深さ、重複、根拠限界と次scopeのeligibilityに限定し、文献・主張の独立検証や実測はしていない。

- R777は1,558語で指定範囲内。現在のQSA kernel帰属方法を選ぶ直接の判断材料になるため、Mainには「Method comparison」と「Recommendation and minimum evidence」だけを参照してもらう。実行前に現在binary、loaded libraries、unitraceのbuild/version、kernel symbolとQSA callsiteの対応を固定する。時間区間からoccupancyやCPU負荷は推定しない。
- R778は1,347語で比較は十分だが、7件の主張に対して文中の正確なURLはIntel oneMKLとcuSPARSEの2件だけだった。Bell/Garland、SELL-C-sigma、Ginkgo、Yang et al.、CSR5の出典は再利用前にリンクを補う。Arc性能への適用根拠はない。
- R779は1,477語で指定budget内。R759/R763と近いgraph・SLM題材を含むが、transaction sector、page translation、依存遅延を分けて扱う点で独立している。実際のresident gatherやArc固有のpage/TLB性能は確認していない。

R778/R779の原文は今のMain判断に不要。次のGQA再設計調査は、KV共有候補のimmutable sourceをMainが確認し、既報と異なる問いが残るまで保留する。resource/counter調査もR777方式の閉じたtraceで対象kernelと具体的な未解決差が定まるまで保留する。すぐ開始できる幅広い非LLM調査は、degree skew下のgraph atomic fan-inとreduction順序の比較。colondelimiterの解決済み設定失敗は新証拠なしに再調査しない。
