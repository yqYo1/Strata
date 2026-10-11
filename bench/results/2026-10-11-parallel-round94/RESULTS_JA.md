# 読み取り調査とdense GEMMテストの準備

R366は物理262144セルの専用probeと保存復元のソース条件、R368はモデルなしの最小GPU検証候補を確認した。元レポートとSHA-256を保持した。[registry v117](report-registry-v117.json)は前のv116へつながる増分索引で、完了報告は485件。実行結果とは区別する。

R368のdevice_selftestの説明を訂正する。poison付きDeviceArenaは64MiBのdevice memsetをsubmitしてchecked waitするため、照会だけのテストではない。最初のRuntime検証では `STRATA_EVENT_MARK=barrier` を明示し、markerの自動比較実行を避ける。kernel-benchはdense GEMMの数値正しさを証明しない。

別のgpt-6.1-solがpublic BF16/FP16 Gemmの小さな数値比較テストを作成した。独立double oracle、全出力、入力・出力guards、BF16 beta=1、変更入力による反復を含む。ステージングのみで親ソースは未編集、build/lint/test/GPUは未実行。DP4aの量子化誤差上界と型別oneMKL成功の観測限界を明示しており、モデル精度の証明には使わない。

再開時に実際のagent状態を確認した。R366・R368・Solは完了し、次のLuna調査R369はサービスのusage limitで停止した。完了報告として数えず、指定されたモデルを無断で変更しない。mainは閉じた必須lintの新規指摘を修正し、次の直列検証を準備している。完了済みSolへ別のoneMath初期試行の寿命・checked completion修正を依頼した。immutable snapshotのみを使うstaging実装で、実行はmainが管理する。

本家の設定監査は[TODO](../2026-10-11-upstream-parameter-audit/TODO.md)に継続する。過去測定は設定を指定していたが、本家単体の設定探索は証明できていない。候補表と最適化済みの測定は別に扱う。
