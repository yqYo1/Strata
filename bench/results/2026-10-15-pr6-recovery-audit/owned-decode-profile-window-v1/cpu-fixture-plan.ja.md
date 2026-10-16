# Main用CPU fixture案（未実行）

この案は製品build/testを実行しない。次のMainの直列境界でのみ実装/実行を検討する。現在のproviderにthread-safeな制御APIが無いという結論を、fake providerのPASSで覆さない。

最初の実証は静的なCPU読取のみ: exact binary hashとCMakeCache、ELF dynamic symbol table、pinned sourceのcompile guard/control実装を照合する。本proposalのpins/exportsはこの段階だけ完了。`dlopen(libunitrace_tool.so)`はconstructorがGPU/runtime初期化し得るためCPU-only symbol確認に使わない。current ghq rootでなくexact pinned sourceを使う。

将来のscope helper用fixtureはSYCL/ITT/driverへlinkせず、fake control（call logだけ）、fake source-boundary outcome、fake monotonic RAW clockを注入する。これはselectorと失敗時の証拠契約を検証するCPU試験である。

| case | 必須結果 |
| --- | --- |
| unset/defaultOFF、invalid selector、missing provider contract | control call0、RAW sample0、file出力0 |
| prompt側runが第8回、別request/別Verifier/別TID、nested scope | on0、retarget無し |
| request1 decode attempts1..7、8がT4、cache済み | 8だけ1回on、両drain成功後1回off、request/run成功後complete |
| 8がT1、8のT4 capture/commit未準備 | on0、missed理由、9以降もon0 |
| initial/next launch、event wait、copy drain、finalcs、finalcopy失敗 | incomplete、失敗status保持、追加wait0、追加callback0 |
| pool例外、caller例外、native異常exit | incomplete、owner outcome別記、架空のsuccess/off0 |
| off後のsampling/out/readout失敗 | capture_drainedとrun_successを分離、request失敗を保持 |
| RAW buffer overflow、clock失敗、write partial、flush失敗 | capture incomplete、required evidence欠落を明示 |
| fake multithread callers | owner1個だけclaim、他threadにwindowを移さない |
| repeated requests/window再利用 | single-shot後rearm0、Verifier::windowsをrequest ordinalに流用しない |

記録のsite sequenceをsource mappingのexpected順序と照合する。poolの明示call spanとb..ms_pool加算区間を別にし、launchがaより前、copy drainがms_pool加算より後であることをfixtureで確認する。GPU時間はfake host clockから生成しない。scope helperが呼ぶcontrolはmockであり、actual providerの並行安全性、graph replay completeness、clock joinは未検証のまま残す。

環境制御の次の検討では、providerがraw environを走査する全threadと、runtime/driver/applicationのenv mutationを含む寿命/同期契約が必要。アプリだけのmutexやCPU stress数百万回PASS、TSANのno-reportはその契約の代わりにならない。ITTを有効にしただけでもpinned IttResume/Pause内のSetEnvは残る。既存固定binaryで契約を説明できない場合はここで停止し、providerの明示的thread-safe public controlまたは境界ackを含む別proposalをMainが判断する。今回のbuild/download禁止、新wait禁止を黙って緩めない。

controllerのCPU-only fixtureは所有process/byte limit logicをstub入出力で確認する案: foreign PID/startticks拒否、deadline/RSS/text/trace/job/file capごとのincomplete、native exit0でもruntime-exit receipt欠落はfail、boot変化/fault/dumpでfail、重複flowやqueue identity欠落でqualification false、原run statusをcleanupで書換えない。provider/actual model/GPUを呼ばない。
