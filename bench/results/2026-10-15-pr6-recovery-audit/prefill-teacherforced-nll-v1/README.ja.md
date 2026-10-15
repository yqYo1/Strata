# initial chunked batched Prefill 全teacher-forced NLL — SOURCE ONLY

**未build・未test・未GPU。統合していないsource proposal。** 既存source、receipt、他proposal、稼働process、Gitには変更なし。`make_proposal.py` はsourceの読取とこのdirへのdiff/pins書込だけを実施したpackaging source。実測中のためcompiler、py_compile、native、testを一切実行していない。数学/host fixtureの合格もモデル品質合格ではない。

`source.diff` を `source-pins.json` の厳密base SHA256と一致するtreeに適用し、全proposed SHA256を確認して再構成する。Git commit SHAは今回Git禁止のため取得していない。baseとは現在worktreeのexact bytesであり、branch HEADの同一性は主担当が後で照合する。base/proposed全copyは保持しない。パッチ対象は新header、generate.cpp、SYCL sequence.cpp、verify APIだけ。fixture/consumerはこのdirにあるsourceを使う。

`STRATA_PREFILL_NLL=/absolute/private/new-output` のみで有効。空値もエラー。通常OFFでは新allocation、clock、queue jobはない。SYCL build限定、native head必須、standalone/single GPUのfresh full batched inputのみ。serve、peer指定、stage、remote、selected-logits同時指定、short `--prefill-until` をGPU preflightより前に拒否。inputは2..262144 tokens、predictionは必ずn_input−1行、position pのgoldはtoken[p+1]。全input tokenをnative vocab判明後・diagnostic upload/head前に範囲検査する。非SYCL buildの実装/validationは対象外。

既存selected-prefill callbackの `lm_head_mix` → `NativeHead::run` を同じmain in-order queueで実行する。量子化、GR normalization、KV/state、MTP/RESTORE、残差dtype/placementは変更しない。callbackのRは読み取りのみ。既存d_logits一行を借用し、head→NLL→次headを順にenqueueする。max-subtracted logsumexpをFP32で計算し、`(max−gold_logit)+log(sum(exp(logit−max)))` を出力する。probabilityをfloatに丸めてからlogしない。NaN/Inf logits、OOB targetはNaN出力で、writerがchunk全体を検査しcompleteを拒否する。極端なfinite差がFP32 NLL範囲を超えた場合もnonfiniteとして失敗する。

gold/NLL GPU ownerは各最大262143×4 bytes。run前に確保し、各chunk最大T個だけhostへ戻し一度checked waitする。per-token waitは追加しない。host NLLは最大同数で保持する。MTP callback連結と`run_prompt_with_mtp_experts`のrestoreは保持。run/restore成功後にownerのchecked drain/freeを完了し、それからfinishする。drain/free不明は既存`cleanup_fail_stop`でownerをunwindせずprocess終了。callback例外/途中失敗もDraftUploadGuardとownerが先にdrainする。大きなRAM/VRAM解放値や実free量は主張しない。

rawはnative-endianでint32 header4個 `{0x314c4c4e, n_input, vocab, written}`、全input int32配列、続いてposition順に `{gold:int32, NLL:float32}`。positionは暗黙で一度ずつ連続。最大raw 3,145,736 bytes、metadata上限4096 bytesで合計4MiB未満。metadataの`.partial`残存hardlinkも小さく、full logitsは保存しない。input identityは全token配列そのもの、consumerはそのSHA256を出す。head/vocab、expected/written/covered、final callback FP32はmetadataに記録。residual storage dtype/placementの実値はこのcallbackでは見えないので`external-recipe-required`と明記する。外側の実recipeはmodel/source/binary/input/env/chunk/dtype/固定GPU-prefix/host-tail/MTP/driver/fault/retirementを必ずpinする。

rawと`.meta.partial`は`wbx`でexclusive createする。write/flush/close/coverage/budget失敗はFAIL。完了markerはraw/meta両方close成功後の`.meta` exclusive hardlinkで公開し、競合する旧fileを上書きしない。overwrite可能なrenameは使わない。hardlink非対応FSはFAIL。partial単独、raw countだけ、footerのある`.partial`もcompleteではない。processの後続decode失敗等は外側receiptで別途FAILとして保持する。これはcrash時のdurability/fsync保証ではない。

fixture SOURCE（すべて未実行）:

- `fixture_writer.cpp`: 正常二chunk/byte layout、2..262144境界、target範囲、coverage順序/重複/欠落、NaN/Inf/negative NLL、exclusive create、publication競合、最大output budget。private新dirを引数に渡す。assertを有効にする。
- `fixture_io_failure.cpp`: ELF linkerの`--wrap=fwrite,fflush,fclose,fprintf`によるbegin/record/count/footer/両flush/両close failure。実I/O操作を全てproduction側writerで通し、complete非公開をassertする。wrapperはfixtureだけでありproductionへのfault envや算術変更はない。上部コメントの個別`--wrap` flagsを使用する。短いwriteの全byte境界網羅やOS crash耐性を証明しない。
- `fixture_nll_sycl.cpp`: existing Runtime/Allocation＋proposed sequence.cppへlink。全10行出力をCPU-double logsumexp oracleと比較。ordinary、large offset、very tiny probability、tail vocab259、extreme finite、NaN/±Inf、−1/vocab target。null/negative/zero-vocab/misalignment/overlap/null explicit queue/非in-order queueをhost側で拒否。全outputにassert、未実行。
- `compare_nll.py`: complete metadataと全record/gold/input identityを厳密確認してpaired mean NLL/ΔNLL/relative-PPL変化を出す。quality/physical256K/RESTORE qualifiedは常にfalse。baseline repeatabilityとtask/retrieval/physical-context別gateを置き換えない。source checksumはmanifest参照。

主担当の次のreviewは、現測定終了後のexact-base確認、patch適用、host fixture/linker fault fixture、SYCL compileとkernel全出力fixture、default-OFF差分確認、その後native短promptから全token paired runへ順次進めること。GPU fixtureリンクの実commandと全compiler/runtime/driver pinsは未取得。fixtureを追加CMake targetへ統合していないので主担当が実際のconfigured resolver/libraryを使用する。physical256K、FP16品質、BF16実装、RESTORE品質、速度改善は一切主張しない。旧auto24576 FAILやselected receiptは変更していない。
