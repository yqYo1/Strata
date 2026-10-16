2026-10-16、R833の変更していないpublic Gemm fixtureをMainが私有compile/linkし、current6354のPASS libraries/native image+host sliceを保存したまま検証した。v3 receipt SHA256 `1f2293b46be99f4922e7bd145e1922b530732da55f26d6ef75759c26f5c5d407`、controller manifest SHA256 `39ed2fe7da8812ad021f6ae39261459810be634acafc52f0a6f04551e9008622`。本番source/build productsは変更なし。

| fresh native process | positive cases | successful public calls | rejected calls |
| --- | ---: | ---: | ---: |
| compact smoke | 48 | 52 | 160 |
| wide smoke | 48 | 52 | 160 |
| compact full | 52 | 56 | 160 |
| wide full | 52 | 56 | 160 |

4入力出力形式FP16/BF16→FP16/BF16、strict IEEE edge、dyadic、cancellation、chain、ragged T、pitch/full span、beta/overlap/context/queue等の拒否を変更のない独立oracleで検証し、実Gemm destructor後の成功markerまで確認した。全4process正常exit、owned PID/start/session閉鎖、source432/build314/runtime等の前後pin一致、新規GPU fault0、dumpなし、全体240.728秒。first compact smokeのみflushed UR tracing+parameter validation、他はwarningsのみ。これはpublic component correctnessであり、速度・一般品質・physical262144/RESTORE/decodeの受入ではない。

元v1はMainによる説明文書更新が旧contract pinと不一致で、GPU未実行のまま停止した（receipt SHA ed08c462cbf43d6fd95aa8edcbbc09876c6987df0d9c316f99563691728dbcfb）。v2はselected receiptのbinding.binaryに存在しないrealpathを要求したschema不一致と、same-UID非dumpableサービスの/proc/exe PermissionErrorでGPU未実行のまま停止した（SHA11864fa7d28469989f381fbf926d4ccc8fcbd91fd26c22be896d592f825b3d62）。元のFAIL/incompleteを保持する。

v3はcanonical path+bytes+SHAを照合し、読めないforeign executableのPID/startと可視性制限を記録する。既存cmdline scanと、見える実executableの照合、Main所有PID/start/normal reap/session-emptyは維持する。hidden foreign GPU owner不在は証明しない。権限変更やサービス停止をしていない。数学oracle/許容値/fixture/compile/link/実GPU環境は変更していない。

成功API全文traceの現在のconsumerはこのcompact証拠のMain検証のみであり、次の直列cleanup境界で有用な代表windowを抽出して退役する。privatefixture本体は後続profiler-controllerの最新healthy receipt照合のconsumerが閉じるまで維持する。GPU測定中のcleanupはしない。
