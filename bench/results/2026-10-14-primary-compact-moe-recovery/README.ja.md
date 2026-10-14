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
