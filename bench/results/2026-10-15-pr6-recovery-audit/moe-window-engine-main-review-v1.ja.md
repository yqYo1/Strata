# Main確認: 実MoE窓のsource audit

2026-10-17。Orca Luna R971/R972をnative Luna max R973がレビューした。
Mainは元report本文を読まず、完了通知の要旨とR973の短いdigest/JSONだけを読んだ。
reportは1455語/1805語で予算内。実エンジン統合の状態/寿命とresource costを独立に調べた。
候補6ソースは2b4399a0のhashと一致し、syntax receiptのscopeをGPUへ拡大していない。

R971のcancelによる部分更新は、現在のserverではlive_ok=false、失敗時の非公開、次requestで
以前の有効checkpointへのrestoreまたはsession_zeroによって再利用を抑止する。rollbackではない。
周期checkpointへの新窓適用gateと、cancel後の次request検証を維持する。
R972の非compact gather比較は新候補の入場条件と一致しない。比較は同じcompact/MMQ=0で行う。
balancedの仮定なら二8Kのgather1024回対窓512回、top10へne==W集中なら20回対20回。
不要なspan検査のhoist算術はR973の1533回が正しい。ただし現時点ではsourceを書き換えずas-isを測る。

## Mainによる同期回数の追加訂正

R973はintermediate pairをbaseline4対candidate4、final pairを4対6と記したが、
baseline run_implのcallback前のchecked syncを数え漏らしている。R973原文は変更せず、ここで訂正する。
基準のsrc/prefill/prefill.cpp:2810はroute drain、3508はcallback前drain。
その後native16 on_chunk16が中間層ではlayer_major.inc:637のstore drain、最終層では618のbridge drainを行う。
いずれも各8Kに3回、二pieceで6回。候補のwindow pieceは旧callback終端へ落ちずcontinueする。
候補はprefill.cpp:3506のattention-bank drainが二回、moe_window.inc:104のID drainと153の両HC-write drain。
中間層ではcursorのみで計4回、最終層はlayer_major.inc:573のbridge drainが二回増えて計6回。
すなわち同じeligible構成のpair-local明示syncは中間層6→4、最終層6→6。
共通のrun_impl末尾/pass/leaseのdrainや、library内部の待ちはこの数に入れない。実行時間の比較ではない。
最終層のpost-attention bank copyが増える点は依然測定事項。

logical route planeのbytesは同じだが、既存のmapped-host copy_i32と候補のpageable vector memcpyを
同じ物理PCIe traffic/overlapと断定しない。現行clean recipeではSTRATA_GROUP_COPYはunsetで、
mapped allocationを試すコードだが、その成功/実効modeはtraceで確認する。
新counterのdequantsは一expertのGU+down展開pairの数であり、二kernelの合計数ではない。
初回traceではこの区別、actual ne/overlap、gather/GEMM形状、CPU submit/wait、queue時間、device clock等を記録する。

独立auditに現在のsource defectという結論はなく、Mainの同期数訂正もコード変更ではない。
次は資源が空いた後のcandidate full build/小さいtraced GPU検査。一次基準profileを完了してから比較する。
実queue寿命、tensor/state、VRAM fit、full physical262144 consume+RESTOREは未検証、速度採用なし。
Mainの大窓64K scatter案はsource arithmetic仮説として別記し、sum順序/traffic/production精度が未確認のため
新実装や追加researchは現時点で開始しない。
