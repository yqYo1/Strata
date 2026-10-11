# 新ランタイムでのモデル開始前の診断

free候補`ea395598c8e967c632a955dc43b77a9291286d49`を使用した。モデルなしの6件の確認は直前に通過済み。今回の2件はどちらもFAILEDの元receiptをそのまま残す。

v1は`systemctl --user show`で「Failed to connect to bus: No medium found」とexit 1。controllerの明示環境にXDG_RUNTIME_DIRがなかった。GPU・モデルは未実行。全16ファイル・85,990,152,459 bytesのSHA-256取得は、独立したCPU childとして85.33秒で正常終了した。v2は明示的な`/run/user/1000`を与え、全file identityが同じことを確認して、その成功したhash phaseだけを再利用した。全体の失敗をPASSに変更していない。

v2は直前の64KiB転送・16384整数値確認を通過し、同じsource SYCL/UR/UMFの8 library mappingをモデルprocessでも確認した。モデル・262144 contextのsessionを初期化し、GPUの校正kernelも終了したが、serveの受付でexit 2になった。`--spec 4 --prefill 4096`とprofileを渡しているのに、必要なserve条件がないという診断で、READY・GEN・prefill・decodeは実行していない。model ownerにはprotocol EOF errorが記録されるが、direct childはreap済み、sessionは空、cleanup signalはなく、live identityも残っていない。終了後のkernel gateはPASS、新しいxe faultは0。GPUハングとは記録しない。

原因候補は、concurrent host handoffを拒否するwhole-token graphのguardと、serve側のgraph-hit metadata要求の不整合。Mainは別のgpt-6.1-solへ小さい修正と全callerの確認を依頼した。元の危険なhandoff・doorbell・chainを有効にして通す変更は認めず、既存の同期するper-layer verifier/MTPを使う。原因の確定と修正の検証は報告受領後に行う。Mainだけがbuild/lint/test/GPUを直列管理する。Luna調査はサービスusage limitが継続しており、別モデルへ代替していない。

[元receiptと保持manifest](retention-review.json)、[モデル入力の完全なhash](model-input-identity.json)に個別status・実行条件を残す。数値品質、モデル推論、性能、262144セルの全域・保存復元は未検証。[本家の設定TODO](../2026-10-11-upstream-parameter-audit/TODO.md)も独立した探索・再検証が未完了。

fd798512で結果をcommit・pushした後、25件の重複raw log、論理サイズ2,189,680 bytesを削除した。[deletion-manifest.json](deletion-manifest.json)にpath/hash/理由を保存した。元のmodel stderr 3361 bytesは実装agentとMainの現行比較入力なので残す。受付エラーを記録する源receiptのFAILED・owner errorは変えていない。

今回の開始時点では、expert cacheとgraphsの準備後の空きVRAMが3936 MiBだった。これはserveのprefill buffer準備前で、実効INFOや16Kのadmissionの証拠ではない。[capacity-followup.json](capacity-followup.json)に、旧候補の1565 MiBという予算を新候補へ流用せず、受付修正後に大きなチャンクの実効予算を確認する項目を残した。転送回数を減らせる可能性を先に測り、まだ速度向上を主張しない。

その後、[実装担当の完了報告とMainレビュー](../2026-10-11-parallel-round97/RESULTS_JA.md)を保存した。source上の原因はserve用residency tableの準備がTokenGraphのconcurrent handoff許可に結び付いていたことだった。Mainが小さい修正を適用し、release・host ASan/UBSanの両モードを通過した。lintと実モデルの確認は未完了で、元の受付失敗statusは変更しない。
