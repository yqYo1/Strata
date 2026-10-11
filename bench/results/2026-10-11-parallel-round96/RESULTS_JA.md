# parity helperのホスト側故障境界

別のgpt-6.1-solが、実際のproduction checked_transfer templateを直接使う小さなhost harnessを作成した。1024 bytesのcaller vector、正常callbackの全byte/checksum、std::exceptionと未知例外、caller destructor・atexit・after-call・caller catchの観測を含む。stderrを明示的にbufferし、失敗時のproduction fflushも確認対象にする。mainが全source/reportを読んだ。Runtime stubやtemplateの複製はなく、mainが通常host C++のrelease・ASan/UBSan両版で計8 caseを実行し、終了statusと全stderrが通過した。実GPUのsubmission/wait/asyncエラーやoneMath private queueの故障証明ではない。

[registry v119](report-registry-v119.json)はv118に1件を追加した増分索引で、完了報告は488件。実装agentのsource報告とmainの実行結果を分ける。Lunaはサービスusage limitのため新規調査を実行できておらず、モデルの無断代替や未完了報告の加算はない。mainは修正後lint v4、host callback確認、最新source全体のhost ASan/UBSan free・contrib-llvm v6を閉じ、すべて通過した。候補ea395598をcommit・pushし、その後の整数GPU確認とモデルなしの6件のGPU parityも通過した。

[本家設定のTODO](../2026-10-11-upstream-parameter-audit/TODO.md)は引き続き独立設定探索・32K以上の再検証が未完了。この報告で新しい性能やGPU資格を主張しない。
