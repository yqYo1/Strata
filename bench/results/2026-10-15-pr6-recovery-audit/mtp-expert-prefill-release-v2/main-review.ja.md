# Main review and current-root MTP expert-release preparation

2026-10-15。未適用・未ビルド・未実行の候補。元のSol source.diffは
f086fe2e17df5a6347004e93903821e5b687f951e5a7b5dac2ac0ba9289052c7。
`rebase-v1.json`は最初の3-way include conflictを未完了のまま保存した。
`rebase-v2.json`は現在rootのrec lease、selected-head、decode observerを保つ別の準備結果。
`rebase-v3.json`はさらに復元元をprivate immutable RAMへ保持する現候補で、v2の結果は不変。
生成スクリプトの最初のroot指定ミスは製品変更前に失敗し、v2へ経緯を記録した。

実sourceを追い、KV-only record_forward(T,-1)とPrefill::draft_kvがrouted expertsを
読まないことを確認。退役対象は707,788,800 B=675 MiBのimmutable expert allocationと、
それを参照するnormal/coupledのround/step graphsのみ。KV/dense/workspace、
Verifier final_Rをborrowするwindow_Rは維持する。全graphとcs_をchecked retireし、
free成功後に初めてpointer/state/accountingを変更する。初回ロード時に675 MiBの元bytesを
private RAM vectorへ読み、長さとSHA256を記録して保持する。以後の復元はその不変な
RAM ownerを64 MiB以下のviewに分け、各H2Dをchecked完了後に新しいdevice pointerを
publishする。再度のファイルIO、SHA scan、host slab allocationは不要。初回RAM admissionは
675 MiB+1 MiBと8 GiBの余裕を数える。正常経路はMath変更なし。

Mainは次を追加した。

- capture_round/capture_stepがC++例外を投げた場合、新しいexpert recapture modeでは
  caller ownersのunwind前にfail-stopする。default-offの既存例外動作はrethrowで維持。
- before-detach、after-detach、after-prompt-before-restore、after-restoreの4境界で
  実free/total bytesを記録する。曖昧なruntime結果を成功扱いせずowner生存中に終了。
- original include conflictはlimitsとmemoryのincludeを一つずつ保持する解決だけに限定した。
- 初回から675 MiBの元weightsをRAMへ保持し、復元時のファイルIOと全量SHA scanを省く。
  private ownerにwriterを公開しない。初回SHAはsource identity用で、毎promptのhot pathへ入れない。

実測long FP32 planのGPU prefix4096行という結果から、当時のplan budgetは
160 MiB以上320 MiB未満だったと限定できる。これはdisplay2343 MiBからの推定より
実際のallocation時点に近い。cache1300 MiB+headroom512 MiBを足したfreeは
1972 MiB以上2132 MiB未満に相当する（FP32 auto plannerの条件下）。
675 MiBが全部戻る仮定なら34K FP16全store675.82 MiBへ算術余地があるが、
MTP675+decode cache約260 MiBだけで34K FP32全storeまで確実に収まるとは言えない。
R574/Mainの2343 MiB表示に基づく表は条件付き算術であり実admissionではない。
新しい4境界・layer-major plan・実確保結果で判断する。

`selected-prefill-long-v1.json`は同じ34,603入力/34,602 batchedの19位置を比較し、
新binaryのFP32 end logitsが元referenceと一致、両dtypeの全採取logits finite、fault0。
FP16のKL min/median/maxは約9.1e-12/0.000994/0.04358 nats、
選択位置の平均target NLL差は-0.08222 nats、top1は19/19同じだった。
相関する19位置を独立task数やfull PPLとみなさない。top1一致も品質採用条件ではない。
この結果により次のcapacity/performance候補へ進むがFP16既定化はしない。

Mainの次gateは実buildとSHA known-answer CPU fixture、その後matched small/longの
正常detach/restore、old end-logit/selected-row regression、normal/coupled recapture、
同じserve instanceの反復とcold/restored decode cost。GPU故障の意図的注入はしない。
R576/R577はMainの長文検証中にFINAL受領済みで、profiler/transfer提案を別途レビューする。
