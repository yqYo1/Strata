# Main diagnostics integration and CPU boundary

2026-10-15。目標1000 prefill / 70 decode token/sは未達。最後の正式bench.shは
32K prefill中央値187.2、長文decode strong8.4/weak4.8 token/sで、本境界から
新しい速度改善を主張しない。

MainはFINAL受領後、各base/source hashを照合し、3個のincremental patchを
現在rootへ適用した。source resolver/CMakeは変更せず、full-file overlayも使っていない。
`diagnostics-application-v1.json`は適用したpatchと実sourceを記録する。

- selected prefill logitsは通常OFF。local BlockBuffersコピーと実native head分岐を使い、
  既存MTP callbackを一度forwardする。最大64位置、一行host row、既存d_logitsを再利用。
  head/readbackのchecked wait前にownerを破棄しない。metadataが正常公開されていない
  partial captureは完了扱いしない。CPU fixtureは範囲/順序/上限、raw/meta/no-clobber、
  失敗prefixの非完了を検証しPASS。MTP内部rec問題は別の実修正を同時に統合した。
- MTP E-4 rec leaseは、H2D前arm、既存最終sync成功後retire。異常return/exceptionでは
  recより先にchecked guardが走る。完了不明ならowner生存のままabort。正常時に
  追加GPU waitを入れない。既存DraftUploadGuard CPU fixtureを実行してPASS。
- joined decode observerは2flags exact1のみON、prompt FNV/dense histogramなし。
  actual T/W/R/D/A/C/E、layer callbacks、CPU job/rows、issuer/worker serviceと
  pool task/phase終了をjoinする。defaultOFFでは追加clock/scan/allocationなし。
  単一writer slotと既存done/park publicationを保持するsource reviewを行ったが、
  CPU fixtureだけで実worker happens-beforeやGPU decode criticalpathを検証したとはしない。
  CPU fixtureとsizeof実行PASS。実sizeof request_state70848、native_service49664、
  slot384、clock_scope48 B。出力cap98304 B、combinedcap262144 Bを満たす。

`current-diagnostics-build-v1.json`は実strata buildと全CPU fixtureのcommand/exit/source/
compiler/binary hashを保存する。新binaryはc2e46f2f…。
GPU conversionと5条件モデル比較が通った旧binary26f81551…の結果を、
この新binaryへ移し替えてPASSとしない。次はsmall FP32/FP16 regressionと
自然な34K Thinking入力の複数位置teacher-forced NLL/KL、続いて32K以上contextの
有限decode計測をMainが直列実行する。NLL位置sampleは全token PPLとは区別する。

MTP expert release候補はFINAL受領、未適用/未ビルド/未測定。
その675MiB解放は全normal/coupled graphのchecked退役と元weights復元が必要。
元snapshot005c47b6に対して現在mtp.cpp/generate.cppが進んでいるため、
現在のlease/observer/head機能を保持したincremental統合を行う。
Direct component候補も未適用。採用決定は部品単位、prefill/decode別で行う。

終了済み75ファイル、29,354,339 logical B / 30,538,752 allocated Bを
`closed-artifact-retention-v3.json`の通り削除した。元PASS/FAIL状態とhashは不変。
FP32 natural0、小規模FP32 controlとFP16-autoだけを次のregression consumerとして保持。
FP16候補の7個のfull review copyはreachable005c47b6+source.diffから
全hashを再現してから退役し、重複ソースをGitへ増やさなかった。
