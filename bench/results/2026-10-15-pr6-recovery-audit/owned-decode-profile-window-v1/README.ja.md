# 一つのnormal decode windowの収集案 v1

2026-10-15 JST。gpt-6.1-sol。proposalのみ。書込はこの新規directory内だけ。製品source、build、test、GPU、provider実行、Git、service、cleanupは行っていない。project専用strata-upstream-sync skillとartifact retentionに従い、Main actual GPU中のsourceを固定した。禁止receipt、private Codex、envscanは読んでいない。

**固定済みproviderでは、安全性を説明できるin-process gateを確認できなかったため、source.diffを作らない。** 同じbinaryで実modelを全traceして後で切り出す代替もしない。次のCPU実証とprovider契約の解決までcontrollerはrun禁止とする。

実binaryのSHA256はreceiptと一致した。CMakeCacheはBUILD_WITH_ITT=0、実ELF dynamic exportに`__itt_pause`、`__itt_resume`、`__itt_api_init`を含む`__itt_*`が一つも無い。pinned tracer.hはitt_collector.hをBUILD_WITH_ITTで囲む。READMEのITT対応という一般記述からこのbinaryで使えるとは判断できない。`--chrome-itt-logging`も同じcompile条件のCLIである。製品側にもITT依存が無く、lib追加、dlsymによるprivate C++ data書換、別provider buildを提案に混ぜない。[pinned tracer.h](https://github.com/intel/pti-gpu/blob/6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033/tools/unitrace/src/tracer.h#L31)、[pinned CLI](https://github.com/intel/pti-gpu/blob/6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033/tools/unitrace/src/unitrace.cc#L303)。

`PTI_ENABLE_COLLECTION=1/0`は公式のin-application方式であり、起動後に一切効かないというR588の含意は採らない。ただしpinned UniControllerは毎回`environ`のpointer arrayと文字列を自前で逆走査する。これは単にglibcのgetenv改善に依存すれば解決する契約ではない。SYCL/UR/driver/provider/workerが存在する中でアプリthreadからsetenv/unsetenvすると、配列更新、pointer publication、文字列寿命、他のenv reader/writerとの同期を説明する必要がある。自前mutexはprovider側の走査を保護しない。起動時の事前setenv、値の長さ固定、putenvで一byteだけ変更、release fenceだけでもC++/Cの同期契約を証明しない。このため今回はenv gateを製品hookに実装しない。[pinned control source](https://github.com/intel/pti-gpu/blob/6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033/tools/unitrace/src/unicontrol.h#L437)。

同sourceのIttPause/IttResumeも内部でSetEnvするため、仮に別binaryでITT exportが見つかっても、ITTであるだけでこの懸念を解消したとは扱わない。named sessionのCLI pause/resumeは公式方式だが、sourceの通知から別processが反応するだけではwindow開始前のon/完了直後のoffを保証しない。ack待ち、SIGSTOP、追加queue waitで穴を塞ぐ案は今回の「既存境界だけ、新しいwait無し」と両立しない。[pinned runtime control docs](https://github.com/intel/pti-gpu/blob/6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033/tools/unitrace/README.md#L807)。

provider receiptにsource path fieldはなくbuild_pathだけがあった。そこからCMakeCacheの記録source pathを得たが、そのworktreeは既に存在しなかった。ghq rootの現sourceを一度読んだところITT制御のcompile条件に差があり、pinned sourceの証拠には使わなかった。以降はexact commit URLの一次sourceをread-onlyで読んだ。Git操作、worktree復元、dependency download/buildはしていない。exportは実binaryのnm/readelf静的読取で確認し、dlopen/--help等のprovider実行もしていない。

添付のsource mapping、CPU fixture案、Main controller recipeは、現在のblockerと、解消後の必要条件を具体化する。88 GiB RSS / 1800 s wallを実model初回案とし、64 MiB trace、64 MiB text、単一file64 MiB、job256 MiBを開始前に固定する。8 GiB/300 sはsynthetic GEMM用であり実32K+ modelには流用しない。RAW host markerとGPU実eventは最初は独立列。GPU clock join、flow/PID/device/queue coverageは実収集なしにはqualifiedにしない。KV転送量やH2D wire bytesは推定しない。

成果物: pins.json、provider-exports.txt、source-mapping.ja.md、cpu-fixture-plan.ja.md、controller-recipe.json、controller-recipe.ja.md。source.diff無しの理由は上記の固定provider契約。全qualificationはfalse。次のreviewは「固定binaryの安全な境界制御を実証できるか」のCPU契約確認後。依存やprovider変更が必要なら別proposalとしてMainが判断する。
