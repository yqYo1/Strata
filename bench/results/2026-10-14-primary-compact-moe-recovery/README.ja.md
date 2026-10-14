# 現在のrootへFP16 MoE scratchを復元する

2026-10-14、Arc B570 / Ryzen 5600X / 128 GiB。
旧1ebのFP16経路を参考に、rootのXs/GU/Hhをexpertごとに再利用するopt-inを実装した。
`STRATA_PREFILL_MOE_COMPACT_FP16=1`の完全一致、全MoEがFP16 fallback、対象形状の場合のみ有効。
Dmとroutingは全T*K行を維持し、同じin-order queueでgather→GU→SwiGLU→Downを行う。
float演算やCPU/decodeの選択は変更していない。既定OFF、未採用。

同時に、large-ring threaded skipでissuerの非atomic used_of読取を待たずにconsumerが
書き換えていたraceを修正した。group flush→既消費prefix publication→wait_issued acquire→
skip event/map→新prefix publicationとした。default ring8はこの経路を使わない。
private copy queueを追加していない。progress診断にbranchとskip/group counts、scratch実効と
accounted/shared bytesを追加し、設定しただけで有効扱いしない。

現在のconfigured buildを再利用し、変更前native d3861dd5を名前付きcontrol consumer付きで
保存した。rootのresolver、precise math、size_t range、znver3等の旗を保持。
新native6361e706、buildと既存host range contractは正常終了した。
GPU kernelsの別移植やcompiler/runtime変更は行っていない。

初回のactual129 batched tokens、chunk4096、context262144、synthetic fixed token IDsのCLIは
native正常0、sampled residualと全first logitsを出力し、fault0・dumpなし・同じbootだった。
ただしMainの検証器が、CLIにないサーバー用終了メッセージを要求して停止し、ON arm未実行。
**元pair receiptはFAILEDのまま**。build成功やOFFの正常終了をpair/品質/性能の合格にしない。
[元状態](initial-result.json)、[byte-identical receipts](receipts-index.json)。

CLIの終了はcompat releaseのwait_allとAllocationのchecked_cleanupによる完了確認を使う。
Allocation completion failureはfail-stopし正常0にならない。少なくとも正常native終了、
owner closed、API/cleanup errorsなし、sampled residual/full logitsのfinite、実効mode、
fresh faultsを確認し、存在しない終了メッセージを要求する検証器を修正する。
必要な確認を外して失敗を成功へ書き換えない。

Luna R508/509をMainの実装・build準備と並行し、結果FINALまで介入していない。
原文はresearch/。R508の『旧/現copy両方alias』は誤りで旧private/current aliasを維持する。
R509はcurrent aliasでのhost orderingをreviewした。別queueでのskip payloadにdevice readerはなく、
copy→次copyは一つのin-order copy queueで順序付くため、追加copied waitが常に必要という
提案は未確定。別queue採用・GPU障害原因の確定には数えない。

次は同じ新binaryのFP16 off/onを別processで実行し、sampled residualと全first logitsを比較。
tail、ring16/group skip、relayoutとactual262144入力/0出力、実RESTOREは別の必要な検査。
標準速度は無変更本家bench.sh、32K以上、3回/armで測る。診断の短い入力のt/sを性能に使わない。
read-only R510ではnormal fused GRを維持できるHC scratch復元を調べている。

元build/first diagnosticの再現recipe:

```sh
python3 bench/results/2026-10-14-primary-compact-moe-recovery/build_candidate.py
python3 bench/results/2026-10-14-primary-compact-moe-recovery/check_model_pair.py compact4096
```

fresh outputが必要で、閉じたrunを上書きしない。成功raw全文はcompact evidenceへ抽出して
commit後にretireする。現在のresidual/logit capturesはoff/on comparator consumer、16 MiB/pair、
次reviewはpair比較完了。変更前nativeは128 MiB以内、consumerは同条件regression control、
比較確定後にretireする。新buildの予算4 GiB、旧1eb fixture/buildは変更していない。

## 実行済みの数値・skip確認

元のv1失敗を保持したまま、検証器を修正し、fresh v2でOFF/ONを両方再実行した。
全armでnative/owner正常0、API/checked-cleanup errorなし、fault0、dumpなし、同じboot。
129 batched inputと1 output、context262144の合成token診断であり、言語品質・長文速度の測定ではない。

| 確認 | 実chunk | 結果 |
| --- | --- | --- |
| FP16 compact OFF/ON、capacity4096 | 129 | residual30720要素、first logits248320要素、finite・bit一致 |
| FP16 compact OFF/ON、capacity64 | 64/64/1 | 同じ比較がfinite・bit一致、末尾1行を処理 |
| MMQ inline/threaded、ring16、stream-min16 | 64/64/1 | 同じ比較がfinite・bit一致、修正対象を実行 |

threadedの2大chunkで実seq各24448、skip19764/20747、open group直後のskip3449/2815を観測。
inline/threadedで同じcountsだった。event-enabled、group-gatherを含み、R8の未実行経路を
通った証拠になった。source HB proofと機能確認であり、TSanや全contextの合格ではない。
小さいstream-min overrideによる大量転送・診断時間は標準速度に使わない。

capacity4096のaccounted bytesは2024214528→1662831872、361382656 bytes
（344.641 MiB）減った。shared regionは968510976→607128320。
MoE payload405 MiBの縮小より小さいのは、他phaseとのmaxが必要量を決めるため。
capacity64では他phaseが最大でaccounted/sharedは変わらない。allocator accountingであり、
driver内部cacheを含むruntime peak VRAMの差とは区別する。

OFF/ON間のlogit範囲とargmaxも一致。chunk4096のargmax409、chunk64のargmax1385であり、
**異なるchunk構成の出力一致を主張していない**。現在の各chunk referenceに対するlayout-only比較。
初回生成1tokenなのでCLIのMTP表示には0draft rounds由来の`-nan ms/round`があったが、
保存した全logit/residualはfinite。モデルのNaNとは分類しない。
[全個別確認](numerical-results.json)、[元receipt](receipts-index.json)。

R510と別Solの[fused HC scratch案](fused-hc-proposal/report.md)は未適用・未build・未実行。
normal fused GRを維持し、carried xn16/grs/inj等を残したままlo/lo16/gatedをphase regionへ移す。
mixedを別配置にするが、R510の『later MoEだからmixed/bo aliasがunsafe』は証明されていない。
既存unfused限定のHC案を無理に有効にして、遅い演算経路へ変えることはしていない。

正式比較のrecipeは`check_stock.py 0` / `check_stock.py 1`。
同じ新nativeのFP16経路、MoE compactだけOFF/ON、無変更Niko bench.sh long3、quiet、
actual32049入力/24出力を使う。先の175.3 t/sの旧binary結果を新OFFの代わりにしない。
性能・品質・actual262144入力/0出力・RESTORE・borrowed relayoutの採用条件は未完了。

## 新しいOFF基準の正式測定

2026-10-14、native6361e706、同じFP16経路・compact=0の本家bench.sh一式を完走した。
long項目の実32049入力・24出力・reuse0、context262144、chunk4096、MTP4/split OFF。
本家の32K項目名と実token数を区別する。diagnostic/profiler/validationは速度測定に入れていない。

| run | prefill t/s | decode t/s | offered / accepted |
| --- | ---: | ---: | --- |
| 0 | 186.8 | 6.0 | 30 / 14 |
| 1 | 188.3 | 4.3 | 42 / 11 |
| 2 | 187.5 | 5.4 | 36 / 13 |

PP中央値187.5、pooled187.526、decode中央値5.4、pooled5.134 t/s。
native、server/client、ownerの正常終了、全child回収、source/binary一致、同じboot、
fresh fault0、dumpなしを確認した。[個別値と終了状態](stock-evidence/compact0/summary.json)。
元receipt/client結果/native identityはbyte-identical gzipで保持し、元状態を変更していない。
新binaryのON比較はこの境界では未実行。旧d386の175.3を新OFFの代用にしていない。
この差をthreaded修正、compact、単一kernelの効果に帰属させない。R8は修正対象を通らない。
24出力のdecodeは4.3–6.0で、採用や旧16.635の回復を確定する結果ではない。

Mainは元公開発言364件の本文hash・時刻を元session offsetで再確認した。
古い1ebの実compile databaseも確認し、generate/verify/expert_sourceはSYCL overlay、
native CPU/poolはroot TUと記録した。[実選択の証拠](selected-old-decode-source-proof.json)。
R514v1のold root参照をv2で訂正し、過去の速い自分のforkの結果を取り消していない。

Luna R511/512、続くR513/514、R515/516を異なるscopeで反復し、FINALまで介入していない。
Mainの測定監視・source照合と並行した。完了後のsource誤り訂正も原文とregistryへ残した。
[原文・訂正](research-index.json)。GEMM orientation、QSA fallback transpose、profilerの
支持・効果はいずれも未測定。既存PfTimerはqueue/host gapを含む区間で、exclusive GEMM時間ではない。

別Solの[full-input v2案](full-input-proposal-v2/report.md)は、実262144入力/0出力を既存の
262143 batched + 最後の実入力のnative T1 verifier/commitで処理する。設定しただけのMTPと
実bindを分け、最後のno-successor draftを処理したとは数えない。v1からv2の修正はsource reviewで、
未実行の不具合が起きたと主張しない。両案とも製品sourceへ未適用・未build・未test。
[exact ne/転送要求量のv2案](routed-fp16-observation-proposal-v2/report.md)も未適用であり、
成功した実モデル観測に代用しない。Mainが次のsource適用・build/test/GPUを直列に管理する。

四つの新しいunified patchはgit apply --checkを通った。Gitのouter-file whitespace検査は
patch内の単一spaceのblank contextを警告したため、字句とhashを保持した。
[形式の確認](proposal-format-check.json)は適用可能性だけで、build・モデル実行の合格ではない。

## 同じ新binaryのOFF/ON比較完了

ONも無変更本家bench.shの全項目とlong3を正常終了した。actual32049入力/24出力/reuse0、
4096 chunk、262144 context、MTP4/split OFF。source、native、client、model、CLI引数が一致し、
計算設定の差はMoE compactだけ。別processをOFF→ONで一つずつ起動し、各process内の3 requestsを
測ったものなので、独立processによるpaired反復とは呼ばない。

| run | ON prefill t/s | ON decode t/s | offered / accepted |
| --- | ---: | ---: | --- |
| 0 | 194.7 | 3.7 | 45 / 10 |
| 1 | 194.2 | 3.1 | 45 / 11 |
| 2 | 191.7 | 5.7 | 30 / 14 |

PP中央値187.5→194.2（観測差+3.57%）、pooled187.526→193.554。
decode中央値5.4→3.7、pooled5.134→3.871で悪化方向、draft acceptanceも異なる。
24出力の3回から安定した退行率や『無害なばらつき』を確定しない。
prefillの差でdecodeの採用を正当化せず、独立した長いdecode反復を必要とする。
両armとも正常native/server/client/owner終了、全child回収、source/binary一致、fault0、dumpなし、
同じboot。これは実行合格であり、品質・full-input・RESTORE・既定採用の合格ではない。
[ONの全個別結果](stock-evidence/compact1/summary.json)、[比較と採否](stock-comparison.json)。

旧同じstock clientの343.8 t/s、旧調整済み449.7 t/sを回復していない。
次は実モデルのroute形状・転送要求量・phase区間を取得し、測定済みhardware性能と照合する。
既存phase区間に含まれるhost/queue待ちをexclusive kernel時間とせず、critical pathに基づいて
旧compact、attention、queue、chunk、spec splitの復元順序を決める。現時点で支配的原因は未確定。

Luna R517/518と別Solの[grouped gather案](compact-fp16-gather-batch-proposal/report.md)は、
測定監視と並行して完了し、FINALまで介入していない。いずれも未採用。
gather案はXsだけgroup-localにし、GU/Hhはexpertごとのrow0再利用、Downはglobal CSR位置を維持する。
Mainのsource reviewとgit apply --checkは通ったが、未適用・未build・未実行。
R517の過剰なqueue順序条件、K10で不可能なtest route、R518のroot/overlay・OFF/ON・24出力の
誤記は原文を変えず[registry](research-index.json)に訂正した。GDN/WYやstate精度変更は候補に留める。

evidence commit `b73d15a9` のbytes/hashを検証後、閉じたON成功ログ19件と完了researchの重複2件を
退去した。合計271,890 logical bytesで、実空き容量差は測っていない。
[削除manifest](stock1-deletion-manifest.jsonl)、[退去集計](stock1-retirement.json)。
元の結果状態・全個別値・必要gate fixturesは維持し、source/model/buildや稼働runに触れていない。
ONの元rawは退去済みなので、close_stock_evidence.pyを同じrunへ再実行しない。
