# 資源別測定の続きとカウンター取得の実結果

現行の長大prefillの観測結果を、帯域・演算・CPU発行の制約に分けて扱う。
元の時間・転送量・全個別sampleは `hardware-resource-profile-v1.json`、
計装条件による差は `resource-instrumentation-context-v1.json` に残した。
今回、新しい速度改善を採用したり、計装速度をclean速度へ混ぜたりしていない。

| 資源 | 現在の実測と判断 | 残る確認 |
| --- | --- | --- |
| PCIe経路 | 長入力Attentionの観測区間ではcopy占有9.11%、M2D要求payload/service約6.05GB/s。通常H2Dの別componentは約6.02GB/s。転送中はその参照に近いが、PCIeだけで全体が支配される説明は支持されない。 | 同じallocation・copyサイズでの容量比較、依存を守った重なり。要求payloadは物理wire bytesではない。 |
| GPU内部 | 同区間のcompute command占有77.31%、長入力QSA60回で33.963秒。まずこの経路を調べる価値が大きい。 | ALU/XMX、VRAM/cache、SLM、同期、依存のどれが足りないか。computeというcommand種別だけで演算飽和と決めない。 |
| CPUの1コア | 発行threadの98.420CPU秒の70.8%は明示GPU待ち。実際のexpertコピーは50.293GB/4.852wall秒で約10.37GB/s、groupingは0.106CPU秒。1コア占有だけでCPU演算律速とは判定しない。 | 待ち以外の28.723CPU秒、12万回超のGEMM投入、発行とデバイスの依存。コピーの値は論理payloadでありRAM物理trafficではない。 |
| RAM/VRAM | RAMの過去copy/streaming値と、同stackのD2D copy測定を操作ごとの参照として保持。 | 現行kernelの実DRAM/cache traffic、レイテンシ、再利用。copyの帯域をそのままAttentionの帯域にしない。 |

単なる機能時間の短縮ではなく、対象資源の上限と依存経路から改善可能量を見積もる。
例えばcopyが約9%を占める観測区間で、転送だけをゼロにする仮定は、その区間内の
約9%を上限とする参考計算になる。これを全prefillの実現可能な高速化率とは呼ばない。
計装で約9秒がbarrier前に見える件も、clean処理から回収できる9秒とは判断しない。

## 小さなカウンターテストを実行した結果

公式Intel PTIの完全なsourceを固定して別のprivate treeへコピーし、Mainが検査した
API結果記録patchだけを適用した。collectorをfresh buildし、実際のcompiler入力452件、
3つのpatch header依存、fresh object/link、生成headerのビルド中snapshotを確認した。
公式Release処理が生成headerを消すため、後からの想像ではなくコンパイル中のコピーと
Ninjaのmtimeを照合した。private treeのGit commit macroは実際に空であり、sourceの
同定には完全baseline/patch inventoryを使用する。元のcollector/checkoutは変更していない。

controllerのpin表現差を修正したv4も含め、offline31ケースがPASS。最初の実行v1は
監視側の4MiB hard file limitが内側の62MiB設定を拒み、native exec前に失敗した。
元の失敗statusを保持し、別のv2実行で親limitを64MiBへ合わせた。native環境と数値条件は
同じ既存BF16 tinyのまま、モデル測定へは進めていない。

v2では、`zeContextCreate`、24groupsの取得、ComputeBasic EVENT_BASEDの83列の確認と
`zetContextActivateMetricGroups` がSUCCESS。その次の最初の `zetMetricQueryPoolCreate`
(PERFORMANCE、count64) が **0x78000004 / ZE_RESULT_ERROR_INVALID_ARGUMENT** を返した。
auditはexit125で停止し、所有プロセスは終了、新規GPU fault0、dumpなし。
GPUハングではない。数値oracle完了もGPU内部counterの取得もできていない。

`own-metric-query-tiny-v4-v2.json` が元receiptのbyte同一コピー、
`own-metric-query-pool-failure-v1.json` と `metric-query-pool-failure-v1.stderr` が
API順序・戻り値の証拠である。列挙・activation成功をcollection成功へ昇格していない。
またINVALID_ARGUMENTだけから権限不足、context違い、Strata kernelのバグを確定しない。
既存公式driver sourceは下位のquery作成/初期化失敗にもこの広いerrorを返すため、
実際のruntime/library版と照合する読み取り専用調査をOrca Luna R905へ依頼した。

Mainはその間、証拠をcommitし、既存時間・payloadと資源容量の対応を整理する。
PR#6の復元は別branchでSolに、無効化したdecode doorbellを通らないprefill専用API
harnessのsource実装を依頼した。ビルド・テスト・GPU実行は引き続きMainが直列管理する。

これらの実測結果が示していない資源飽和を断定せず、counterを使える場合は主要componentで
取得し、使えない場合も同条件の容量測定と制御したcomponent比較で競合する仮説を検証する。
