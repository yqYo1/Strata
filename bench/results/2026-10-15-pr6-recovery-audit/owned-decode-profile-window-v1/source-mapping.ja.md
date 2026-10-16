# source境界と将来hookの契約

source pinはpins.json。resolverはroot CMakeLists.txtのSTRATA_ENABLE_SYCL分岐 → cmake/sycl_backend.cmake。strataはsrc/program/generate.cpp、strata_engineはsrc/core/verify.cpp、互換層はsrc/sycl/compat.cppとruntime.cpp。root CUDA generator由来の別実装を推定しない。

ordinary serve decodeはgenerate.cpp:6006付近のDecodeServiceBudgetScope、:6060付近のbegin_window → ver.run。promptをVerifier経由で読む:5722付近にもrun callerがある。従ってVerifier::windowsや単なる第8回runだけではdecode request/windowを選べない。将来selectorはordinary serveのdecode callerだけでrequest-local ordinalを付ける。新規native process、text request一件、input >=32768、normal MTP4、singleGPU、greedy/no penalties、非coupled、suffix/lookup/oracle/corrupt無しに限定。ordinal=8はfirst T1も含むdecode attemptの1-based番号で、当該Tが4でなければ未収集として終了し、第9回へずらさない。

選択keyはowned PID+startticks、caller request ordinal=1、Verifier identity（pointer値はそのprocess内だけの識別子）、decode ordinal=8、T、pos0、owner Linux TID、cs_/copy_ opaque identities。親controllerのrequest receiptと対応付ける。Verifierはlb_=0/le_=g.n_layers、next_=nullptr。別Verifier、別thread、nested run、複数requestへの暗黙fallbackを禁止する。defaultは完全OFF、exact opt-in switchとselectorが揃って初めてarmする。switchだけ、parse error、識別不足、provider不適合は収集に入らない。

第8回でもT4のcold capture回避は番号だけでは保証できない。run入口でsegments_[4]が既に存在し、commit_exec_も既に存在したことをstateとして確認する契約が必要。存在しなければcaptureを収集対象にせずmissed/coldcaptureとして返し、そのprocessで再armしない。capture/capture_commitは既存のまま呼ぶ。現在のproposalにhookは実装していない。

現在のnormal SYCL順序（verify.cppのrun）:

```text
capture(T), capture_commit(err)          [first-useのみ重い準備]
host staging, flags, bookkeeping
initial graph_launch_with_completion   [host launch span; ms_waitの外]
cudaStreamQuery(cs_)                    [既存query]
for k=0..steps-1:
    a=Clock::now()
    progress_at
    cudaEventSynchronize(layer_done_)   [既存event success boundary]
    released check, reached_.store
    b=Clock::now()                      [ms_wait += b-a]
    set_plan_slot等
    pool(...)                          [独立call span; 共有/hit graph launchも起こり得る]
    status/progress/flags/launch_work fallback等
    ms_pool += ms_since(b)              [pool callより広いhost区間]
    cudaStreamSynchronize(copy_)        [ms_wait/ms_pool両方の外]
    next graph_launch_with_completion  [ms_wait/ms_pool両方の外]
cudaStreamSynchronize(cs_)               [既存final compute drain]
cudaStreamSynchronize(copy_)             [既存final copy/callback drain]
window_scope.drained()                   [通常collection offの候補境界]
optional profiling/sampling/logits等
out copy to caller, window_scope.finish(), ++windows, return true
```

aはinitial/next graph launchの**後**であり、host launch自体をms_waitへ含めない。b-aにはprogress通知とhost event completion経路もある。明示pool callとcopy drainはms_wait外。ms_poolはpool call単体より広いため、RAW pool spanと等号で結ばない。これらを加算してwhole windowの排他的分解にはしない。

compat.cpp:478付近のlaunch_graphはq.submit→ext_oneapi_graph、graph->last依存、completion eventへのbind。cudaEventSynchronizeはruntime event wait/error checkを経る。sourceだけでhost waitをGPU activeへ変換できない。launch_workはpool中publish_planまたはpool後fallbackからcs_へ共有/hit graphを発行し得る。fetch_dmaはcopy_へ既存copyと既存host callbackを発行する。このcallbackや新queue/host task/waitを計測用に追加しない。実callback serviceをhost pool spanから推定せず、actual traceが識別できなければunknownとする。

将来安全なgateが証明された場合、on候補はrunのcapture成功・host staging成功後、initial graph launchの直前。offは既存final cs_/copy_の両success後。現在はこの位置にunsafe setenvを追加しない。前windowのasync commitは次window開始時に未完了の可能性があり、同queue依存は成立してもon直前がGPU idleとは限らない。新drainで隠さずcarry-in項目として扱い、実device coverageで前window残件を識別できなければstrictなwindow-exclusive serviceをqualifiedにしない。first segment完了後からonにする案は第一segment欠落のためfull-window captureの代わりにはしない。

早期return/例外はcapture incomplete。安全な両drain前のdestructorでpauseしたり新waitでdrainしたりしない。pending workがあり得るまま「complete」を出さない。失敗markerを可能な範囲で記録しMain ownerがfail-stop/terminate/reapし、byte/wall capで尾を有界化する。両drain後off済みでも後段runが失敗したならrequest/run outcomeは失敗を保持する。正常capture終了と正常run returnは別bit。

将来のRAW host記録は固定長buffer（例1024records、overflowはincomplete）、CLOCK_MONOTONIC_RAW ns、PID/TID、scope IDs、source site ID、begin/end、layer/group/segment、API resultを持つ。run owner threadだけが記録し、loop内printf/files、per-expert記録、atomic shared callback bufferは増やさない。initial/next launch、event wait、pool call、copy drain、final cs/copy drainを低頻度で記録、成功drainまたは終了後にflushする。共有/hit launchを追加でtagするなら実呼出threadのownershipを先に確認し、別threadをownerと偽らない。

RAW host nsとPTI device nsは独立列。PTIのhost API→flow→device operation ID、native PID/startticks、Arc B570 metadata、actual Level Zero context/queue/lane、device engine coverageを確認後にのみ操作対応を付ける。absolute RAW clock joinはさらに別gateであり、UNITRACE_SystemTime=1やCPU header検査だけでは通らない。queue gapはsame-queue gapで、全GPU idleとは呼ばない。kernel名filterは用いず、copy/dequant/gather/GU/Down/その他全workを残す。copy allocation prefixがpaused中に欠ける問題もあるため、endpointが不明ならtransfer direction/physical trafficはunknown。KV/H2D wire bytesをsource geometryやflow名から推定しない。
