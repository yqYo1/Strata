# R851: R850 レビュー

R850 はdecision-support reportとして適切な範囲を扱っています。報告は1,813語で、指定の1,500–2,000語・一次URL最大7件の予算内です。自然32K FP32-bのFAIL/incompleteを維持し、原因を断定せず、前の別キュー回帰と切り分けています。

SYCL 2020 R12の一次仕様では、`handler::memcpy` の両端に通常のhost pointerを使えます。したがって `std::vector` の行先自体は規格違反ではありません。ただし、この許可はIntel runtimeのpageable転送がstagingや暗黙同期をしないこと、あるいは停止原因でないことを証明しません。in-order queue、同じqueue内の投入順、`wait_and_throw` の完了・非同期エラー配送の説明は適切です。ソースも同じmain queue上でGR/head、D2H、checked waitを行い、wait後まで残差行とscratchの所有者を保持しています。scratch aliasはprefill内の`mixed`/`block_out`をembedding scratchに重ねますが、session block arenaとは別です。具体的なcallbackのUAF、誤った同一queue依存、範囲外行、直接のcache/scratch alias欠陥は見つかりませんでした。

MainはR850の次だけを取得してください。

- 15–22行: 同一queue、通常host行の寿命、guard、bank所有権、alias境界。
- 23–25行: 全行NLLの投影・転送規模と性能診断に使えない理由。
- 29–33行: 原因未確定の仮説とCPU/source-onlyから始める切り分け。
- 37–41行: sparse selected-logitとobserver案。以下の修正を反映して読む。

主な修正点は三つです。位置677の1回だけselected-logitを計算して成功しても、dense callbackの反復loop、累積状態、FP64 reducer、stdio経路は否定できません。observer案のphase一覧には、比較に必要な`before-sync`がありません。前後の単調時計時刻と、submit/wait/reduce/appendごとのスレッドCPU時間を記録し、親が読む状態には一貫したsnapshot手順を設けてください。`after-sync`はそのwaitが観測期限までに戻った証拠であり、それ以前の長い待ちを除外しません。さらにYama1でstackを取る案は、実際のtracerが許可された親子関係にあり、通常のdumpability/credential条件も満たす場合に限って安全です。Linux kernelのYama文書を出典に加え、ptrace設定は変更しないでください。CPU mockはguardの制御分岐を確認するだけで、SYCL runtimeやdeviceの寿命を証明しません。

R849本文は読まず、ハッシュとassignment metadataだけを保持しました。次の研究は今は`schedulable_now=false`です。将来、安全なphase観測が実際にpageable D2Hまたはそのwaitへ時間を局在させた場合に限り、インストール済みIntel stackの版を固定して、通常host pointer転送のstaging・blocking・完了契約を一次資料で比較します。現receiptにはそのphase情報がありません。

対象R850: `/home/yayoi/orca/workspaces/Strata/sycl/bench/results/2026-10-15-pr6-recovery-audit/research-round850-v1.txt`
SHA-256: `2ce0af8a73afa1913d1018be1b520db94729b28c00a19918f3bab9d7dab16c70`
レビューJSON: `/home/yayoi/orca/workspaces/Strata/sycl/bench/results/2026-10-15-pr6-recovery-audit/research-scope-review-round851-v1.json`
