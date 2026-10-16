# MTP4 route/service観測案のMainレビュー

2026-10-15 JST。Solの原patch `4bf7911c5f4fcccd52e62a10cb34313c7aa137cc12c7ed58f888a380b14632d8`
とreportを保存した。8箇所のbase bytes/新規file不在は現在のtreeに一致する。
[base照合](mtp-decode-ledger-base-join.json)。適用、compile、fixture実行、GPU観測はまだ行っていない。
現thinking suiteのsource/client/recipeを変えず、検証後のPR#6調査を優先する。

現在のscriptはMTPを読み込み、spec4・splitOFF・suffix0を指定している。旧observerの
no-MTP/NT1 gateをそのまま使うとこのdecodeを測れない。新案は既存route/serviceの2observerを
単一request/closeへ統合し、初回T1と以後の実T1..4、jobの実nt1..Tを分ける。
window attempt、verify成功、commit return、最終wait、実emitted数を分離し、D=R-W、C=W+A、
E<=Cをチェックする。layers×emittedを期待callback数にしない。

sourceではinput activationはlayerあたり実T rowsで一度quantizeされ、そのbufferをjobへ共有する。
一方、poolのDown activation quantizeは各CPU jobの実nt rowsに対して行う。
案の`quant == cpu_rows`は後者のrole1/4 counterの和であり、input quantization回数ではない。
input側は既存`d.ms_actq`のdeltaに残す。jobごとの同じweight blockの共有と、この二つの
activation quantizationを混同しない。

route countを`multi_misses`/`multi_entries`/`cache_hits`/`pcie_experts`/poolのlogical byte countへ
照合し、ゼロCPU-job layerもcoverageへ含める。format tableのcopyと範囲確認はsourceと合う。
ただしjobのblob pointerを独立したexpert-ID oracleに照合するものではない。first-miss順の
source構築と実ntを数える観測であり、重みやtokenの数学的正しさを証明するfixtureではない。

workerのrole slotは一writer。pool bindingをjob publication前に置き、元のpark完了後に解除する。
window/roleを閉じる際にworkerへ追加waitはしない。実native CPU callがpark boundaryを通って
戻らなかった場合はworker slotを読まず、一つのstateを保持しcomplete=0とする。この分岐は
observer storageの寿命保護で、推論の再開やfault recoveryではない。
CPU clocksは実thread CPU時間とwall spansを測るが、その差を純粋な待ち時間としない。
既存phase intervalsも重なるため単純に足してrequest timeの排他的内訳としない。

storage capとoutput capのstatic_assertは未compileで、実sizeof/RSSの測定値ではない。
outputが256KiBを超える場合には省略数とfalse completeを保持する。serviceだけのcompleteを
joined closeのfalseより優先しない。source上のOFFは新しいclock/allocation/outputをしないが、
pointer/branchやstack handleの追加をゼロoverheadとは主張しない。

採用前に必要なのはMainのserial fixture/buildと、同じ新binaryのOFF/ON短い正常requestでの
source/counter/終了status/数値比較、続いて32K以上かつ実長生成でのcoverageとoverhead比較。
clock/syscallのobserver costも実測する。必要な情報は実nt・format別のroute exposure、各roleの
CPU時間、pool task/row distribution、既存GPU wait/pool/host spansである。
これを得る前にdecoded cacheやgraph dependencyの小変更を主要ボトルネック対策としない。

実装案にはdevice queue/event/copy/wait/graph、math、affinity、task scheduleの変更はない。
ここでの判断は観測案のsource reviewが進んだことであり、decode速度やbottleneckを実測で
特定したという判断ではない。原reportとfixtureの未実行statusを変更しない。
