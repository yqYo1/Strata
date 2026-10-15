# 外部 named session による normal MTP4 decode 診断案 v2

2026-10-15 JST。proposalのみ。製品source、Git、build/test、GPU、provider実行、service、cleanupは変更・実行していない。Mainのactive GPU runとsource freezeを維持した。読取対象はresearch-index.json、R588、R591、owned-decode-profile-window-v1の全資料、artifact retention、実configured SYCL source、固定providerの公開exact revision source、a29d8257…のdirect_owner.pyである。project専用strata-upstream-sync skillを使用した。

v1/R591の「新しいhost wait無し」「exclusive traceの厳密保証が無ければ実装不可」は以前のproposalが置いた制約で、ユーザー要件ではない。この案は別の診断processで有限host handshakeを追加し、既存providerをそのまま使う。clean性能測定は別processとする。provider fork、ITT、setenv/unsetenv、非export symbolの書換え、追加device queue、追加device wait、driver resetは不要である。

## 最初に実施する内容

初回はChrome timelineを出さず、`--device-timing`のkernel/command集計だけを採る。名前、実呼数、合計GPU service ns、平均/min/maxを保存し、合計service順に並べる。queue gap、同時実行の重なり、GPU全体idle、host callback serviceはこの集計からは観測できない。合計kernel serviceは重複実行を含み得る和で、whole-wallやGPU busy unionではない。hostのGPU-boundary timerとの引算で説明しない。

実CLI候補（placeholderはMainが開始前receiptに実値を固定する。今は実行しない）:

```text
<pinned-unitrace> --device-timing --session <fresh-alphanumeric-ID>
  --start-paused --result-dir <new-private-job/provider>
  /usr/bin/python3 <pinned-main_controller.py> --runtime-child <new-private-job>
  <diagnostic-native-binary> <exact-normal-MTP4-native-argv...>
```

`--device-timing`はL0 buildで有効。`--session`、`--start-paused`、別processの`--resume <session>`と`--pause <session>`が公式CLIである。`--result-dir`は`--output`、`--output-dir-path`と併用できないため、v1のflat-directory案を混ぜない。追加の`--chrome-*`、`--call-logging`、`--device-timeline`、`--kernel-submission`、PMU/metric query、kernel名filterは初回に付けない。`--follow-child-process`は既定1を維持し、runtime wrapperのnative childも収集対象にする。[exact CLI](https://github.com/intel/pti-gpu/blob/6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033/tools/unitrace/src/unitrace.cc)、[official README](https://github.com/intel/pti-gpu/blob/6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033/tools/unitrace/README.md)。

固定providerはBUILD_WITH_ITT=0のためITTを使わない。named sessionがある場合、`UniController::IsCollectionEnabled`は共有session stateからreturnし、conditional environ走査に進まない。外部CLIのstate変更は対象アプリのenvironmentを変更しない。CLI自身がexec前の設定にSetEnvを使うことと、稼働中modelのenvironmentを変更することは別である。Mainは既存のfrozen runtime environmentを変えず、非秘密のopt-in channel path一個だけをexec時に加える。`PTI_ENABLE_COLLECTION`のruntime toggleも、UNITRACE_SystemTimeによるRAW join仮定も用いない。[exact control](https://github.com/intel/pti-gpu/blob/6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033/tools/unitrace/src/unicontrol.h)。

## source.diff の境界

source.diffは未適用で、追加headerと実SYCLのgenerate.cpp/verify.cppの小さいcallsite差分である。`STRATA_EXTERNAL_SESSION_PROFILE_V2`は標準buildで未定義。Mainがisolated diagnostic childのstrataとstrata_engine両targetへ明示defineした場合だけhookをcompileする。さらにexec時の`STRATA_EXTERNAL_SESSION_PROFILE_DIR`がunsetならclock/channel操作をしない。標準buildの追加callsiteはpreprocessorで除去される。通常source、binary、queue順序は今変更していない。

ordinary serve decode callerだけがrequestとdecode attemptを数える。promptのver.run回数をdecode ordinalに流用しない。Mainの新規native processに実modelのGEN一件だけを送り、input34603、max_new128、greedy/no penalties、normal non-coupled MTP4、single GPU、suffix/oracle/corrupt/split無しを固定する。初回T1を含むdecode attempt8だけを選び、第8回のTが4でない、T4 segments/commit graphが既に準備されていない、whole-layer verifierでない場合はMISSEDとし、後続へずらさない。request counterはdecode入口に達したrequestを数えるため、Mainはその前の別GENを一件も送らない契約を守る。

markerの`decode=8`とは別に`window=ver.windows+1`を記録する。prompt処理を含む実verifier window番号を捏造しない。Workは`work=normal_mtp4`、実T/pos/layers、Verifier identity、cs_/copy_ opaque identity、request/decode番号、native PID/TIDで記録する。source clockはCLOCK_MONOTONIC_RAW ns。source markerのACTIVE/PAUSED/FINISHED、controllerの受信、CLI開始/終了、ack送信RAWを独立に保存する。

正常順序は次のとおり。

```text
既存capture/staging成功（warm条件はcapture前の状態で検査）
START marker → Main: unitrace --resume session 完了確認 → ACK START
ACTIVE marker
既存 initial graph launch → 全既存segment/event/pool/copy順序
既存 final cs_ drain成功 → 既存 final copy_ drain成功
DRAINED marker → Main: unitrace --pause session 完了確認 → ACK DRAINED
PAUSED marker
既存sampling/readout成功 → FINISHED marker → 既存run正常return
```

resumeは最初の対象submissionより前に完了し、pauseは既存両queueの正常drainより後に完了する。外部反応だけのv1と異なり、sourceがackを待つため、その間に対象windowが通過しない。追加waitはこの二つの有限host handshakeだけであり、queue/event/hostcallbackは追加しない。前windowの非同期commitを新drainで隠さない。START handshake中に前のcommitが完了することもあり、そのスケジュール変化も診断侵入である。carry-in、無関係workの混入を完全に排除したとは主張しない。

二つの5秒wait、profiler自身、CLI起動時間は診断wallを変える。既存ver.run、decode caller、whole request wallにwaitが入るため、このrunのtok/sやboundary marginalをclean性能として採用しない。二つのwaitを後から引けばclean性能に戻るという補正もしない。kernel serviceのobserved内訳としてだけ使う。

early return/例外はFAILED、対象missはMISSED。destructorは非blocking markerのみで、pause/新device drainを行わない。Mainは捕捉不完全として既存Ownerで有限終了する。両drain成功、pause成功、ver.run成功、request成功、native exit成功、provider completenessを別々のbitで残す。後段readout失敗をcapture成功で上書きしない。

## channel と Main controller

`main_controller.py`のcontrollerは既存Main runnerへimportする候補である。単独実行はMain用の明示`--runtime-child` modeだけに限定し、native argvを継承stdin/stdout/stderrと変更しないinjected environmentで起動する。environmentを列挙・記録せず、native PID/startticks/binary hash/RAW開始/実exit/reapをexclusiveな16KiB以下のruntime-start/exit.json各一件へ残す。既存runtime_child.pyはexecutable一個しか受け取れず、Strata argvの直接代替にはならないため、ここに複数argv用の小さい候補を含めた。新wrapperは未検証なのでMainのCPU fixtureと新source hash固定を必須にする。

`prepare_channels`で新規absolute directory0700、markers/acksの専用FIFO各0600をO_EXCL相当の新規作成で用意する。productはchannelを作成せず、O_NOFOLLOW/openat/fstatでowner/mode/type/link数を検証する。両端はO_RDWR|O_NONBLOCKで開き、相手消滅時のopen待ち/SIGPIPEを防いで、5秒timeoutへ収束させる。既存stdinはGEN/QUIT専用のままである。

markerは最大16件、1件512bytes未満、合計8192bytes。ackは二件で各128bytes未満。PID/window/phaseが一致するackだけを受け付ける。重複、partial record、過大行、誤phase、owner不一致、書込失敗、欠落はincomplete。FIFOは同一UIDの完全な悪意へのsecurity boundaryとはしない。専用private jobと通常の所有契約を前提とする。

Mainは既存Owner.runのinteraction内でController.install(io)を呼び、元のpollを保ったままFIFOを監督する。`identify(pid)`をMainが実装し、native wrapperのPID/startticks/binary hash、既にpidfdを取得したowned session内の実体、期待TIDを独立検証する。markerから未知PIDをadoptしない。`persist(records)`は監督済みjob内の有限receiptだけを書く。

capture制御CLIは同じa29d8257… direct_owner.Ownerの**別instance**でresume/pause各一回だけ起動する。model ownerの元pollをCLI interaction中にも呼び、modelのRSS/log/deadline監視を止めない。CLI通常deadline2.5秒、Owner wall3秒、source ack deadline5秒。CLI exit0だけでなく`[INFO] Session ... is resumed/paused`の実state-write確認を要する。CLI failureのOwner終了処理が5秒を越す場合、sourceはtimeoutしてincompleteになり、Mainも既存有限terminationへ進む。失敗後にackを送らない。

実際に読んだOwnerは`/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/xe-llvm-v711-source-v5/direct_owner.py`、sha256=`a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f`。leaderはsession anchorとしてfinal scanまでunreaped、signalはpin済pidfdだけ、numeric killpgや未知ownerのadoptをしない。候補はOwnerをコピー/修正しない。

exact unitrace.ccのLinux fork/waitpid経路はKernelMetricsまたはChromeKmdLoggingがある場合で、今回のdevice-timing/sessionだけの経路は`execvp`でruntime wrapperへ置換される。Ownerのunreaped leader PID/SIDはそのままwrapperのsession anchorになる。wrapperはnativeのpidfdを開き、death確認後・reap前に同じnative identityを再検証して実exitを保存する。Mainが新wrapperをCPU検証し、そのsource/hashとnative spawn/exit/reap receiptを新receiptに固定する。現在のproposalにその新実行receiptは無い。

named sessionの共有メモリはpauseだけでunlinkされると仮定しない。Mainは32桁の新規random hexをsession IDにし、開始前に対応する`/dev/shm/uctrl<ID>`が存在しないこと、開始後に同じpathがUID一致/0600/1024bytesであることを確認してinodeを保存する。既存sessionは再使用しない。model Ownerのfinal scan/reap/session空が確定した後、同じinodeのsessionが残っていればMainが同じcontrol Ownerで`unitrace --stop session`を一回だけ実行してsessionを閉じる。既に消滅していればその観測を記録し、stopしたとは記録しない。candidateは`model_closed()`がtrueのときだけstopを許す。ここでは既にreapしたmodelの元pollを呼ばず、job budget checkだけを渡す。capture中のstop/flush介入を避け、stop失敗もclosure failureとして残す。Mainはumask0077でjob/provider/control/channelを作り、全出力とshared sessionを同一UIDのprivate診断として扱う。このproposal作業ではstopもcleanupも実行していない。

## 容量と時間

RSS88GiB、start-to-reap1800秒、provider aggregate64MiB、raw text64MiB、job256MiBを開始前に固定する。GPU/modelとcontrol CLIのRSSを合算する必要がある。例えばmodel Ownerを87.5GiB、CLI Ownerを0.5GiBに割り当てて合計88GiB以下とする。全processは同じjobのfile/log予算に課金する。

Mainのnative Owner.runはwall1770秒、text_cap64MiB、total_cap256MiB、file_cap16MiBを使い、既存OwnerのTERM/KILL/final scan/reapと最後のsession-close CLIへ30秒の余裕を持つ。CPU-secondsはwall秒と取り違えず、既存runner同様に`1800*len(os.sched_getaffinity(0))`の有限guardを与える。Owner自身はstart-to-reap1800の独立timerではないため、Mainは最初のspawnからcontrolを含む最終reapまで実wallを記録し、1800秒未満かを別gateで判定する。異常closure/未reapは失敗として残す。

初回provider出力は新規`provider/`配下の`<launcher>.<pid>/run_config.json`、`<native>.<pid>/device_timing.txt`を期待する。runtime wrapperもlibraryを継承するため、そのprocess名/PIDの空または非空device_timing.txtが現れる場合をMainの事前allowlistに含める。wrapper行はnative行と混ぜない。実identityに対応しない追加process/report、Chrome/CSV/raw timeline等はunexpectedとしてincomplete。候補Budgetはbasenameをこの二種類に限定し、max8 regular provider files、per-file16MiBを検査する。Mainはさらに実process名/PIDに対応するrelative path allowlistを固定して最終照合する。

Budgetは通常20msごとにprovider48MiB/text48MiB/job192MiBのsoft thresholdを監視し、通常の上限までreserveを残して停止する。既存OwnerのRSS scanは250ms、per-file RLIMIT_FSIZEはchildが継承するhard limitである。しかし**aggregate64MiB、text64MiB、job256MiBはfilesystem quotaではない**。monitor間隔、process scheduling、CLI failure終了処理、終了時flushのovershoot bytesに事前の厳密上界は無い。8files×16MiBも64MiB以下の保証にはならない。初回は低出力量の集計を選んでこの限界を受け入れ、上限到達、観測gap、未知file、I/O failure、途中終了を必ずincompleteにする。final observed byte counts/peak/file一覧と各hash、monitor最大gapを記録し、hard aggregate capを保証したとは報告しない。timelineの厳密hardcapだけのためにprovider forkを新作しない。

## pause と pending events の exact-source 判定

今回再取得したexact URL bytesのSHA256とline rangeはpins.jsonに残した。pinned `ze_collector.h` の成功した `OnExitEventHostSynchronize`/`OnExitEventQueryStatus` は `ProcessCommandsSubmittedOnSignaledEvent`を呼び、成功した`OnExitCommandListHostSynchronize`/`OnExitCommandQueueSynchronize`は`ProcessAllCommandsSubmitted`を呼ぶ。これらのpending command回収、`ProcessCommandSubmitted`、`LogCommandCompleted`からtiming statsへの加算はcollection enabledを再判定しない。従って、**既にstaged/submittedとして登録された通常kernel/commandのtimingをpause後の回収が単純に捨てる実装ではない**。destructorのFlushData→Reportで後から回収されるpending timingも同じ経路である。[exact collector](https://github.com/intel/pti-gpu/blob/6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033/tools/unitrace/src/levelzero/ze_collector.h)、[exact tracer](https://github.com/intel/pti-gpu/blob/6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033/tools/unitrace/src/tracer.h)。

ただしmetric queryのreadbackは回収時の`IsCollectionEnabled(true)`に依存し、pause後にmetric dataを保存しない経路がある。この案でmetric queryを付けない理由でもある。queue/list/graph submitのOnEnterとOnExitはenabled gateを持つので、途中toggleはstagingを失い得る。source handshakeは対象windowの全submissionと既存successful drainsの間にtoggleしないが、他runtime thread/callbackやcarry-inの不存在を証明しない。

さらにpaused中のcaptureには`ShouldCaptureAppend`のgraph-capture例外、graph instantiate/replay用callbacksがあるが、実SYCL non-native command_graphがこの実provider/runtimeで全kernel/copyを登録することは未観測である。opaque eventやdriver内部work、hostcallbackが全て見える保証も無い。normal kernel集計が非空になっても全window coverageの証明ではない。必要なcallback/API経路、実replay、実native PID/device/laneの対応が確認できない場合は `collection_complete=false`、`attribution=incomplete_observed_service`のままにする。名前/呼数/observed serviceを得る目的には使えるが、boundary全時間の原因確定へ昇格させない。

## Main の直列検証と次の判定

active GPUが閉じ、既存measurement lockをMainが所有した後にだけ進む。Mainがsource.diffをisolated childへ適用し、target両方のcompile define、configured resolver、source hashes、診断binary hashを新receiptへ固定する。測定armのsource/base binary pinsを旧baselineと混同しない。

CPU fixtureはcpu_handshake_fixture.cpp。header以外にSYCL/ITT/driverへlinkしない。MainのCPU build例は `c++ -std=c++17 -DSTRATA_EXTERNAL_SESSION_PROFILE_V2 -I<applied-child>/include cpu_handshake_fixture.cpp -o <owned-cpu-fixture>`。Mainが既存Ownerで各caseを直列実行する。Controllerのcontrolはfake関数に差し替え、resume/pause呼出記録だけ返す。**実unitraceをfixtureから呼ばない**。正常caseはSTART/ACTIVE/DRAINED/PAUSED/FINISHED各一件、decode8/actual window42、二capture controlのみ。source macro OFFおよびpath unsetはmarker/clock/channel操作無し。cold/wrongTはMISSEDだけ、後続再arm無し。failure_after_startとfailure_after_drainedはFAILEDでincomplete。ackを送らないcase、phase/PID/window不一致、line/byte/file budget超過、owner identity不一致、control非0/確認無しも失敗することをMainが確認する。新runtime wrapperもfake nativeの複数argv受渡し、exit0/非0/signal、start/exit receipt、pidfd death前のreap無し、private file/no overwrite、外側session closureをCPUで確認する。fake制御PASSはproviderのcoverage証明ではない。

GPU診断はその後、一process/一GENのみ。実input34603（input hashはpins.json）、KV resident32768/int8、cache128 per layer、pool5+host、MTP4、FP32 residual GPU rows4096、MTP expert退役ON、prefill4096、max_new128、既存native capture0、copy-offload disable1、DirectSubmission0を維持する。full argv/environmentはbaselineの明示whitelistからMainが新job pathだけ置換し、秘密envscanをしない。owner/source/model/binary/runtime/driver/input/profile file pins、前後fault/journal/boot、native終了、session空、出力fingerprintを新receiptに保存する。

最後にController.incomplete=false、partial marker無し、FIFO最終drain、native/launcher/control全closure、expected provider report、全bytes内、warning/zero-row/timestamp failure無しを確認する。観測したkernel/command各行を全てcompact JSON/CSVへ写し、個別呼数/合計nsを保存する。coverageは上記の独立gateが通るまでincomplete、queue gap/absolute GPU→RAW join/clean性能/quality/adoptionはfalseを維持する。次のreviewはこの集計で最大serviceのkernel群が識別できた時点。timelineが必要になったら別processと別bounded proposalで検討する。

成功rawはMainがcompact結果・commands/env/pins・original status・各hashを検証した後、artifact retentionのmanifestを伴って退役する。未解決failureはdistinct mechanismの代表だけ保持する。このproposal作業では収集もcleanupも実施していない。
