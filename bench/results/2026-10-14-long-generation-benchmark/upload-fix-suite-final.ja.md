# thinking有効のbench拡張検証

2026-10-15 JST。Arc B570 10 GiB、Ryzen 5 5600X、128 GiB RAM、native IQ3_S。
同じwarm serverで`bench.sh --long --long-generation --think --runs 3`を実行した。
context設定262144、chunk4096、KV int8/resident32768、MTP4/splitOFF/suffix0、
expert-cache128/per-layer、pool5+host。生成上限はlong16384、agent8192。
long生成のthinkingは有効で、reasoning budgetはunlimited。1Kは最小生成量である。

## 個別結果

| 測定 | 実入力token | prefill t/s、run順 | decode t/s、run順 | MTP採用率、run順 |
| --- | ---: | --- | --- | --- |
| 標準nominal32K | 32089 | 188.9 / 187.2 / 171.8 | 24-token capのため長いdecode評価に使用しない | 使用しない |
| long regular-code | 33214 | 145.1 / 180.7 / 180.3 | 8.4 / 5.8 / 9.6 | 82.28 / 81.98 / 82.32% |
| long varied-prose | 33288 | 151.5 / 174.3 / 179.4 | 4.8 / 3.8 / 6.4 | 37.17 / 39.77 / 38.87% |

コード生成のdecode中央値は8.4 t/s、小説は4.8 t/s。生成量はコード3346/4164/3198、
小説8631/9056/8869 tokenで、thinkingを含む。全6件が正常なSSE終了、finish=stop、
実usage、fresh prefix、回答本文を持つ。強いMTPは各run>=60%、弱いMTPは各run<=40%を満たす。
採用率が近くても速度差があり、採用率だけでは原因を説明できない。
標準nominal32Kは32768未満であり、その実値を丸めない。long6件は全て32768以上。
agentは3 sessions×6 turnsの全18件が本文と正常なstopを持って完了した。

本文のpresenceはtask品質の合格ではない。小説run1/2の本文は3129/3391文字で、
指示した4000文字には達しない。モデル能力、精度変更、完全なtask品質は未評価。
context設定256Kは実入力256Kの検証ではなく、physical262144/RESTORE qualificationもfalse。
prefill1000/decode70 t/sの目標は未達である。

## 終了・保存検証

native/server/bench/runnerは全てexit0。Ownerはdirect child reaped/session empty、cleanup/errorsなし。
元PID/start identityは不在。boot不変、新しいkernel GPU fault0、driver dumpなし。
source239files/client6files/launch recipesと実binaryはlaunch時のhashのまま。
source revisionは16d9d2b993a56a874d63324bbe9b55782ab1b6b3、
binaryは6180b2bb535a4841385a4988de54ca405b6cdeccbb51fb62b134c5608830d1a6、
実engineはrootの0.1.38で、old fast overlayを使っていない。

`suite-v4-upload-fix-final-summary.json`に個別46responseと終了gatesを保存した。
`receipts/main-primary-long-generation-suite-root-v4-upload-fix-v1/index.json`に
元structured receipt9件のpath/bytes/raw/archive hashesを保存した。decoded bytesは元と同一。
kernel420entriesのfilter/boundary/zero faultsと、startup/prefill/long cache・KV情報を別summaryに記録した。
一般benchのmachine.gpuは空であるが、native startupはArc B570(SYCL)を明示する。

`test_finalize_upload_fix_suite.py`はmeasurement lock内でreceipt copiesのみを操作し、
8 CPU casesが通った。active/ownership/hash/result不整合は拒否、実long input不足とcapped agentは
独立gateで未qualifiedとした。正常系のidempotenceと元status/bytes不変も確認した。GPUは使っていない。
今回のimmutable draft upload source修正とsmall two-batch safety probeの範囲を越えて、
過去の全GPU障害原因を解決したとは主張しない。

## 次の作業

PR #6の645.85 t/sは当fork自身の実測であり、XeStrataの測定ではない。
現在rootにはそのlayer-major selectorがないことを確認した。旧source/config/公開Codexログを照合し、
queue・host/VRAM/graph寿命とfull physical256Kを点検して、有効なprefill部品を回復する。
旧測定は32768入力/8192chunk/65538context/K8V8 fullyresident/no timed MTPであり、
現在のstreamed KVとの相違を明記する。prefillとdecodeの採否を別々に判断する。
並行するLuna source safety/provenance reportsは`../2026-10-15-pr6-recovery-audit/`に保存する。
