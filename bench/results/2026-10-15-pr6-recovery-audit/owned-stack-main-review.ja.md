# Owned stack capture: Main review and CPU execution

2026-10-15。モデル/GPU再現実行はまだ行っていない。正式Direct0失敗とDirect1未実行は変更しない。

Mainはproposalの7ファイルとpin済みdirect_owner.pyを読み、直接所有する未回収native leader、
pidfd、boot/start/parent/SIDの照合、別SIDの所有GDB、30秒/256KiB制限、全taskのTracerPid確認、
一度だけのcaptureと失敗後のrequest禁止を確認した。Yama scope1のプロセス単位の許可は
[Linux公式資料](https://www.kernel.org/doc/html/latest/admin-guide/LSM/Yama.html)とも一致する。
root、sysctl変更、PR_SET_PTRACER_ANY、GPU reset、他プロセスへのattachは使用しない。

`run_owned_stack_host_build_v1.py`をMainが作成し、共有LOCK_NBの下でCPUだけを実行した。
最初のbuildはGCC13のWerrorで、失敗メッセージ出力のwrite戻り値を無視したため失敗した。
元receipt/exit1を保存し、ptracer_parent.cの戻り値確認だけ修正した。元proposalのhashは
source-manifest.jsonに残し、実行時の修正版hashはbuild/preflight receiptに保存した。
再buildは新root-v2で実施し、shim/fixtureのcompileとPython構文確認が通った。

実際のGDB接続ではmainのread/fgets待機とworkerのnanosleepの2threadを取得した。
GDBは正常detach/exit0、全taskのTracerPid=0、fixtureのheartbeatは3から70へ進んだ。
fixtureの元終了コード7を保持し、native/GDBのsessionは空、全direct child回収済み。
build失敗、成功、実attach/detach/resumeの個別receiptは
owned-stack-host-preflight-v1.jsonに収録した。GPU/API初期化は行っていない。

モデル用にはMain側で正確な32089-token入力3件とGEN/READY/DONE protocolを接続する。
stdoutだけの無出力時間は正常prefillでも約162秒になる。STRATA_TRACEのstderr上の
layer/chunk進捗も監視してquiet timerを更新する。繰り返されるwarningだけで停止検知を
無効にしない。CPUの明示capture成功は自動停止検知、モデルの再現、GPUの健全性、性能の
証明には使わない。これらは次の独立診断で検証する。

R615のsource auditは新しい確定バグを見つけていない。layer27 markerはlayer入口であり、
pool完了だけでmain threadの待ち場所は分からない。queue/library境界をstackに対応づける。
R616のpacked-XMX案は今後の最適化候補で、現状の停止原因とは結び付けない。
