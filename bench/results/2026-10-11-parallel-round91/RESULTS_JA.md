# 更新版の同期と本家の設定調整 — 並行調査91

XeStrata xe0.1.40.2.2 / e32b8b05、本家 v0.1.42 / 61b3fb5d を対象にしたソース調査。リモート main の再確認でも同じ revision だった。GPU、モデル、テストの新規実行はこのレポートの根拠に含めない。

R353 は CPU分担0・単一GPUでの主KV/MTP KVを、同じqueue内の書込み・転送とsnapshot前後の完了待ちに限定して点検した。別のMTP出力ポーリングは未解消として分けた。R355 は通常の spec4 からそこに到達すること、連続 session_replay の mapped入力再書込みに待ちが必要なことを確認した。メインで replay の入力書換え前に完了待ちを追加し、直接token経路の --stream-token も未認定のホスト同時アクセスでは完了待ちを行うようにした。MTPの修正v4も返答を受領した。mapped出力は全queueの完了後に読むようにし、E4入力の所有期間と失敗時のdrain/再使用拒否を修正した。返答前に介入していない。callerの寿命・例外処理と末尾の推測window境界はR359/R360が独立して点検中。

Sol v3 と R356 は、実際の262144セルを全て処理する専用診断、最終セルの明示的なstorage probe、RAMの事前確認、全KVとdraft ringおよびrunning stateのpoison/restore比較を実装・静的点検した。新規実行はまだなく、数値の独立参照・能力評価・通常生成での採用を証明しない。末尾を使い切った後には生成しない。実際のGPU実行による境界確認は保留。

R354 は本家のlong prefillにもdevice-written host-USM sequenceのホストpollが残り、--no-pool等では全経路が安全にならないことを確認した。現在の未修正pristineを実行する性能結果はない。

R357/R358 と [設定調整TODO](../2026-10-11-upstream-parameter-audit/TODO.md) は、本家の過去の明示設定と実効設定を整理した。保持しているv0.1.40.2/v0.1.41長入力の記録は単一レシピの反復で、独立した本家単体の設定探索を証明しない。v0.1.41は要求8Kに対し実効6K、要求128に対し実効144キャッシュスロット。別のtuned比較にはpristineがない。最新本家で対応する設定を確認し、安全に実行できる状態になってからprefill/decodeを個別に選択し、32K以上・decode少なくとも3回の独立した検証を行うTODOを残した。

ソース製Intel LLVMのビルドv3は進行中で、active receiptやログをarchive・削除していない。free/contrib-llvm全ソースビルド、host ASan/UBSanと実helperテスト、新runtimeのcapability/library identity queryのレシピは準備のみ。MTPv4と実callerの独立レビュー、およびツールチェーンの正常終了を待つ。ビルド、テスト、GPU、cleanupを別々に同時実行しない。

[report-registry-v114.json](report-registry-v114.json) は新しい8件だけを登録し、過去465件はSHA付きv113参照で辿れる。過去全レポートと履歴の再コピーを避けた。[prepared-source/manifest.json](prepared-source/manifest.json) は完了した変更だけを再構成するpatchと未trackedソースを記録する。完了したMTPv4も含め、e32に全overlayを再構成できる。実行資格はまだ得ていない。ソースは未ビルド・未採用。新規ログ削除は0件。
