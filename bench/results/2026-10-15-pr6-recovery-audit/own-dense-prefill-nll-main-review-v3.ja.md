# 独立NLL案の表記修正とMain source review

2026-10-16 JST。v3は実commit f27189d6の333 source fileをhash固定したsource-only提案。
root LICENSEはMIT、SHA256 `949b9b0aada7476006469e58ee4ffb0e0f68c2d9137e90a356757b61a09d37f0`。
旧v2のLGPL metadataは実装担当がown project skillの二行を誤って慣例として使ったもの。
外部実装やtemplateを使った根拠とはせず、担当の説明・参照したown production fileとそのhash、
元v2のdiff/manifest hashを`provenance.json`に保存した。current-tree既知由来診断の撤去記録も維持する。

Mainはv2の791行の全patch、現在のprefill_logits writerとDraftUploadGuardを読んだ。
v3の6 proposed file、prepare、全patchをv2と照合し、5 SPDX license identifierのMITへの修正以外は
byte一致を確認した。copyright metadataは保持した。decode側の分離修正4 fileはNLL patchの対象外。
表記変更で出自を消したとは扱わない。これは現在のproject codeを設計参照した自作であり、
project全体から独立したclean-room実装という説明はしない。

source reviewでは、exec前のopt-inとroute拒否、callbackの順番、borrowed残差、既存GR/head scratch、
partial head/readback submissionのchecked drain、例外時owner解放前のfail-stop、全n_input-1行の
target一致と有限loss、安定FP64 reduction、排他作成と全prefill/restore成功後のcompletion公開を確認した。
host fixtureは別方式FP64 log-add-exp、極値、262144表現、順序・coverage・I/O failureを含む。
実API呼び出し・numerical/task品質・故障時寿命の実証はbuild/host/smallGPU/modelで別途行う。

この経路は78452回の既存single-row headと約77.9GBのlogit readbackを追加する品質診断であり、
速度cohortには絶対に含めない。clean selector比較のbinaryを途中で替えないため、現productには
まだ適用していない。build/host/GPU/FP32 repeat/FP16 matched品質/allGPU admissionは全て未実施。
比較器は品質合格を自動で出さず、FP32 repeatabilityと事前の許容値・task評価を別途要求する。
physical262144を使い切るengine/RESTORE検証も未実施。表現上限のhost fixtureで代用しない。

旧v2のfull source mirrorと誤表記の提案は、v3との正規化byte一致・元hash・説明を確認して退役した。
元v2は未適用・未build・未testだった。v3のcompact patch/manifest/provenance/prepareをGitに保存し、
現在のfull packageをMainの次の品質gate consumer向けに16MiB予算で一時保持する。
見直しはscoped selector clean比較後のNLL統合・却下時。退役manifestは
`closed-prompt-jit-and-nll-retention-v1.json`。元測定のFAIL/PASSを変更しない。
