32K入力・同一9c0b291dソース/バイナリで、ring96/4096幅は215.56 tok/s、ring16/6144幅は266.35 tok/sだった。診断付きの個別試行であり、最適設定や性能採用の結論ではない。decodeはそれぞれ8.10/9.15 tok/sの初回16tokenで、独立した反復とRESTOREは未実施。

expert blobの成功H2D要求量は399,981,158,400から295,956,300,800 byteへ26.0%減少した。6144幅では5つのfull-nonresident chunkと2047幅のrouted tailを、全48layer・288行と型/extentまで照合した。要求byte数は物理PCIe帯域やexclusive GPU serviceではない。リングだけを16から96へ広げた試行には速度改善が見えず、チャンク拡大と転送回数削減を次の候補とする。

6144幅の追加initバッファ要求をソースで最大886.0625 MiBと見積もり、17allocに計68 MiBの計画上の丸め余裕を置いた。これはdriver page granularityの実測ではない。起動後の空きは実際に744 MiBで、512 MiBの開始条件を満たした。既存のdecode graph capture時には空きがさらに減る。8192幅へ無条件に拡大しない。

最初のv1は出力比較の位置を誤りKeyErrorでGPU起動前にFAILED。v2では推論16tokenとQUITは正常だが、検証の固定8chunk条件によりFAILED。元のstatus/hashを保持し、正しい6chunk期待のCPU再検証で転送行と16token/LP/top5の完全一致を確認した。新しいv3の実測はnormal QUIT、正しいruntime、全転送行、基準出力、前後のfresh GPU health、新faultなしまでPASS。

グループgatherは既定で有効で、整列した単体gatherも1kernelである。以前の複数copyという説明は非整列時のfallbackに限る。実際のMMQ/FP16選択と未使用expert転送量は未測定なので、そのcensusとbyte parity probeを別実装エージェントから受領し、全文とhashを確認した。census observerのCPU10正常・39拒否ケースはMainでPASS。census sourceとgather probeのbuild/GPUは未検証で、次にMainが順次実施する。

本家は箱出し設定だけの測定ではないが、最適設定探索は未完了。[追跡TODO](../2026-10-11-upstream-parameter-audit/TODO.md)を維持する。実256K、RESTORE、広い能力評価、最新本家の独立設定調整は未完了。
