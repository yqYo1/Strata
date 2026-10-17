# 39K実コマンド計測の結果と未解決点

2026-10-17、検証用エンジン`8e653297`、同じ39,262入力トークン、context 65,536、chunk 8,192、GPU residual 32,768行、MoE window2、最適化済みrow-major QSAで、イベント収集OFF、ONを別プロセスで実行した。両方とも入力再利用0、thinking有効、24トークンの完了確認、native exit 0、所有プロセス終了、新規GPU障害なし。dequant 27,679、job 28,196、window 96という各カウンタも一致した。reasoningのSHA256も同じだったが、24トークンは品質評価ではない。

OFF側の診断は成功。ON側の推論は完了したが、プロファイルの判定は**不合格のまま保存**した。4層の先頭windowで532イベントを保持し、508レコードを出力した。転送8件とカーネル16件が、時刻順序・正の処理時間・分解能・問い合わせ時間の複合判定で除外された。除外された生の値をv1は保存していないため、どの条件に該当したかは復元できない。転送だけの問題だとは断定しない。

508件の出力値はそれぞれの時刻順序と52 ns以上の処理時間を満たし、グループの時間合計と一致する。一部グループでは選択数より出力数が少ないため、全演算・window・モデルの使用率やCPU律速の根拠には使わない。補助記録は`command-service-model-diagnostic-on-partial-v1.json`。元の不合格receiptの状態は変えていない。

ログ・パラメータ検査つきの診断値はOFF 461.3、ON 488.9 tok/sだった。各1回で、同時刻に読み取り専用Luna調査の起動・作業もあった。これはclean性能比較でも収集オーバーヘッドの定量評価でもない。既存のclean最適化済みQSAの3回値490.6/490.1/491.4、中央値490.6 tok/sを置き換えない。

## 独立した小規模再現

普通のhost配列から512 KiBを2回転送し、3,075個のカーネルを投入、141イベントだけ保持する独立テストを、同じin-order/profiling Level Zero compute queueとURトレース・パラメータ検査で実行した。全出力・ガード・入力不変性・141件の時刻判定・正常終了・所有プロセス終了・新規障害なしを確認した。このテストでは現象は再現しなかった。モデル側とはトレースとカーネル等が異なり、原因が解消したことを示さない。

原本は`event-service-stress-v1.json`、ソースは`tests/sycl/event_service_stress.cpp`、実行手順は`run_event_service_stress_v1.py`。

## 次の測定

判定条件を維持したまま、32件までの除外値と失敗bit、source rowを記録するv2をMainが実装した。fixtureの例外時のイベント保持順も修正し、明示的なpre-drain例外モードを追加した。旧モデル測定が終了した後にソースを変更し、別ディレクトリで新規ビルドした。小規模正常・例外テストを閉じてから、39KのON側を1回実行して生の失敗理由を取得する。captureの成功とtimestampの適格性は別の状態として記録する。

時刻契約は[公式SYCL event reference](https://github.khronos.org/SYCL_Reference/iface/event.html)のprofiling descriptorsを参照。submit/start/endの順序条件を、原因不明の不一致に合わせて緩めない。次の性能比較候補には、既存の未実行64K・全GPU residual配置案も残す。大きいcontextでの実メモリ割当と速度は未確認であり、物理262144-contextやRESTOREの採用条件も維持する。
