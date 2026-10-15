# Orca R761–763 Main review

Mainは3報告を全文読み、worker_doneのTask/Dispatch、succeeded settlement、report pathを実launchと
照合した。全effective modelはgpt-6-luna。Codex内spawn_agentではなく、Orca worker-startである。
Mainは実装とactual78K診断を並行して管理し、調査開始後にsteering/interrupt/追加taskを送っていない。
初回startup update promptはTask未開始のfailed attemptで、研究作業とは区別する。

R761: 64-route segment/128-segment block、5段階のstable prefix/scatterは具体案になった。
2,651,144B追加scratchは既存ids/slot/srcを除く。scatterで64 IDsをcooperativeに一度localへ置く
等の実装が必要で、全比較のIDアクセスをglobalに実装してreportの2-read byte式を流用しない。
H[segment,expert]はscan時のstride/coalescingも検討する。2.53MiBだから速いとは言わない。
C/F/statusのhost返送約1.97MBへpayloadを減らせても、host ne依存のwaitは残る。
不正ID/overflow/ordering/async lifetime等の実装検証は未実施で、設計は採用していない。

R762: 同じhostUSM stagingでcopy_i32対queue.memcpyを比較すればprimitive差を分離できる。
STRATA_GROUP_COPY=1だけのモデル比較は通常host vectorへも変わるので、このcomponent比較と
同じ因果比較ではない。現在のcopy-offload安全設定は維持し、実engine未確認ならDMAとは呼ばない。

R763: database fusion/output pages、device descriptors、stencil recomputation等の知識候補。
75.3msは6GB/sを仮定したbulk payload算術で、mapped pathの実測やGPU化の削減見込みではない。
per-expert row descriptor consumerはkernel全体の変更で、64MiB arenaだけでは全GU/Hhを
materializeするmemoryはまかなえない。現行FP16 Hhを含めた全routed行GU/Hhはfull chunk500MiB、
compactからの増分450MiB（R755 Main訂正）も別に計上する。FP16全expert weightsも4.6875GiB。
列挙された論文の数字をXe2へ適用せず、必要な採用候補ではprimary sourceの箇所を再確認する。

3 worker-releaseを実行したがOrcaは全てretained/user_takeover、processAction noneを返した。
その権限判定を尊重し、terminal closeで迂回していない。新規branch/worktreeは作成していない。
レポートと索引をGitへ保存し、次のgoal反復でも異なる有限scopeを新規Orca sessionで調査する。
