# 実メトリック定義の取得と保持整理

Mainが全列挙用C++ source、既存controllerとの差分、実際のhost依存を確認し、
通常C++ (`-fno-sycl -Werror`) としてbuild後、別の所有プロセスで実行した。
`own-metric-enumeration-query-v2.json` はENUMERATION_PASS、native0、同一boot、
所有PID/start消滅、新規xe fault0。queue、command、metric activation/query/streamerを
作るsourceではない。権限・sysctl・service・GPU状態を変更していない。

24groupsの全1,414metric propertiesを取得し、API entry/resultの対応と成功を照合した。
`own-metric-full-catalog-evidence-v2.json` は全定義、名称・型・単位・driver説明を保持する。
ComputeBasic EVENT_BASED group3/domain3の83列も完全に得た。GPU_MEMORY_BYTE_READ/
WRITE/RATE、XVE演算/active/stall/occupancy、SLM競合等の定義が存在する。
これは**定義取得であり、カウンター収集成功や資源律速の判定ではない**。

公式収集器のGetData/decode経路にAPI結果の欠落があることをSolがsourceで指摘した。
`own-metric-query-tiny-v1` はsourceのみで、strict collection資格はHOLD。実行していない。
正確なAPI状態を記録する別source提案をOrca gpt-6.1-solへ依頼し、実際に存在する83列の
解釈・異常flag・分母の調査をOrca gpt-6-lunaへ依頼した。Mainが全ての実行を直列管理する。

成功captureの重複raw、解決済みbudget停止raw、旧summary timeoutの重複個別表など
4ファイル、論理565,999,224bytesを測定lock下で削除した。元FAIL/FAIL/PASS receiptは
byte同一のまま。timeoutの不完全summary原文、compact集計とQSA個別sample、
現行の53,623,555bytesのcanonical全個別表を保持した。削除path/hash/reasonとファイル
allocation・実空き容量の区別は `resource-trace-retirement-v1.json` に記録した。
metric source作業が参照する旧列挙の入力やbinary、現行比較fixtureには触れていない。

前の計算資源別測定はPR#11としてforkのmainにも反映済み (247b182a)。
今回も演算最適化や性能改善の採用はなく、現時点の未確定点はGPU内部の資源律速と
host投入gapの因果、clean cohort、旧PR6復元経路の残存安全性問題である。
