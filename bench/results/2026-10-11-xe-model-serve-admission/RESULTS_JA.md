# 新ランタイムでのモデル開始前の診断

free候補`ea395598c8e967c632a955dc43b77a9291286d49`を使用した。モデルなしの6件の確認は直前に通過済み。今回の2件はどちらもFAILEDの元receiptをそのまま残す。

v1は`systemctl --user show`で「Failed to connect to bus: No medium found」とexit 1。controllerの明示環境にXDG_RUNTIME_DIRがなかった。GPU・モデルは未実行。全16ファイル・85,990,152,459 bytesのSHA-256取得は、独立したCPU childとして85.33秒で正常終了した。v2は明示的な`/run/user/1000`を与え、全file identityが同じことを確認して、その成功したhash phaseだけを再利用した。全体の失敗をPASSに変更していない。

v2は直前の64KiB転送・16384整数値確認を通過し、同じsource SYCL/UR/UMFの8 library mappingをモデルprocessでも確認した。モデル・262144 contextのsessionを初期化し、GPUの校正kernelも終了したが、serveの受付でexit 2になった。`--spec 4 --prefill 4096`とprofileを渡しているのに、必要なserve条件がないという診断で、READY・GEN・prefill・decodeは実行していない。model ownerにはprotocol EOF errorが記録されるが、direct childはreap済み、sessionは空、cleanup signalはなく、live identityも残っていない。終了後のkernel gateはPASS、新しいxe faultは0。GPUハングとは記録しない。

原因候補は、concurrent host handoffを拒否するwhole-token graphのguardと、serve側のgraph-hit metadata要求の不整合。Mainは別のgpt-6.1-solへ小さい修正と全callerの確認を依頼した。元の危険なhandoff・doorbell・chainを有効にして通す変更は認めず、既存の同期するper-layer verifier/MTPを使う。原因の確定と修正の検証は報告受領後に行う。Mainだけがbuild/lint/test/GPUを直列管理する。Luna調査はサービスusage limitが継続しており、別モデルへ代替していない。

[元receiptと保持manifest](retention-review.json)、[モデル入力の完全なhash](model-input-identity.json)に個別status・実行条件を残す。数値品質、モデル推論、性能、262144セルの全域・保存復元は未検証。[本家の設定TODO](../2026-10-11-upstream-parameter-audit/TODO.md)も独立した探索・再検証が未完了。
