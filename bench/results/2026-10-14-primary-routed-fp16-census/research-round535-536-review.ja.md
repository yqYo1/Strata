# R535/R536のMain確認と訂正

2026-10-14。原文のbytes/hashを保持し、誤りを別記する。source/archived sourceと
閉じたv4 screen receiptを読み、計算し直した。新しいGPU測定やソース適用はしていない。

R535の183862579200 bytes / 34560 inputは、**5320097.7778 bytes/token =
5.320097778 MB/token**。本文の「5.319 GB per batched input token」は単位が誤りで、
要約の5.319も丸めを訂正する。1000t/sへの比例投影は5.320097778 GB/s、既測Host-USM
6.447 GB/sの82.5205%。64Kへの比例投影は348.657927964 GB。requested source-copy payloadと
別のbulk転送の比であり、実PCIe通信量、成功転送量、critical path、絶対上限の認定ではない。
cacheの容量比だけではroute偏りによるhit効果を否定できず、削減可能量はcoverageの測定が必要。

R536の「対象はno-MTP/actualNT1」は今回の実測条件と異なる。v4のengine-start/recordは
**`--spec 4 --mtp /home/yayoi/.local/share/strata-sycl/mtp/rt --no-spec-split --suffix-draft 0`**。
実際にMTP pack/headが読み込まれ、要求ごとのoffered/acceptedが記録された。spec容量と
actual dispatch widthは別だが、この設定で常にNT1とは断定できない。

Mainはarchive commit `8e85169a4002e08671e09d84139390e110f2cdc5` の
`include/strata/core/decode_route_ledger.hpp` lines11..18を読み、gateが明示的に
`!mtp`と`actual_nt == 1`を要求することを確認した。
`1cd442981faf137e4370fffdc9d73ab91cce8c7a:docs/DECODE_SERVICE_CPU.md`も同じ限定を持つ。
よって「少しjoinすれば今回のmodeに互換」という結論は採用しない。既存案はreferenceとして
再利用できるが、MTP対応にはactual dispatch T、CPU job nt、window数・検証rows・rollbackと
commitのcoverage、format/owner table、failure/closeの検証を設計し直す必要がある。
単にeligibilityを緩めたり`layers × emitted`をMTPのcoverageに流用したりしない。

current root `expert_pool_dispatch_multi`はn_tokを実dispatch幅として検証し、CPU jobを
expert IDでまとめる。job ntとdispatch Tは別の分母。実emit数からobserved call数を埋めない。
追加観測はdefault-off、host-only、bounded、既存queue/copy/wait/graph経路を保持する。
PR#6回復を優先し、benchmark検証が閉じるまではこの案の適用やGPU確認はしない。

Solの`stager-error-guard-proposal-v1`も原文のまま保存する。31か所のAPI result確認と
失敗時native abortは、sticky errorにより実waitへ到達しない既存リスクに対する未適用案。
fixtureは未コンパイル・未実行。異常exitはDMA完了やGPU復帰ではない。正常時のqueue/event順序、
fixture extraction、既存release/issuer各経路をMainがレビューし、user指定のbenchmark→PR#6境界後に
必要なsource adaptationと直列検証を行う。affinityを広げる性能実験はそれより後。
