# per-layer profile fillの修正候補

[報告](implementation-xe-per-layer-profile-fill-v1.txt)と[patch](per-layer-profile-fill.patch)をMainで全文確認した。1レイヤーの満杯で全fillを打ち切る処理を、有限の全rank走査とレイヤー別quotaへのfillに変更する候補である。globalのprefix順序は維持し、重複profileは明示的に拒否する。

[索引](report-registry-v121.json)には共有した過去reportと今回のsource identity、実際のagent状態を残した。実装agentは完了し、Lunaはusage limitによりactive0。報告の末尾欠落によるEOSの説明は仮説として扱う。Mainの独立レビュー・統合・build/lint/ASan・実occupancy・モデル品質・性能は未検証。
