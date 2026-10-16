# Main review: decode host phase observer v2

2026-10-15。FINAL受領後、v1との差分、new header、2つのCPU fixtureをsource reviewした。
未適用・未build・未test・未GPU実行。両Direct公式full benchの終了を適用条件とする。

v1で指摘した未所有slotのdereferenceは除去された。生存する旧borrowが失ったslotを
detach時に検出し、寿命を知る自分のownerをinvalidとする。新borrowは不明なpointeeを触らない。
別のASan用fixtureにはfreed heap address、live collision、元return/error、後続fresh requestがある。
これらのsanitizer実行結果はまだない。

window bandのcount/sum/histがglobal run_totalを分割すること、ordinal別の期待count、
4phaseそれぞれの全layer count/sumがglobalと一致することがchecked arithmeticで追加された。
独立破損fixtureも追加されている。v1から既存GPU/queue/callback操作の順序は変えていない。
原推論のerror/statusやnative pool fail-stopを診断のために変更しない設計を維持する。

Mainの次の実行候補としてsource reviewを通した。これは実行合格や計測値の採用ではない。
両Direct arm終了後、CPU normal/ASan+UBSan、actual SYCL build、small OFF/ON、
同じ34K128通常MTP4の順でMainが直列に確認する。OFFは追加出力なし、ONはjoinedと
request identity/仕事量/全保存則を独立consumerで照合し、host nsを記録する。
worker serviceやGPU durationからの差引きをGPU idleと呼ばない。
