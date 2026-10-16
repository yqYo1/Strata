# phase診断の実行完了と同binary対照

2026-10-15 JST。b65d5d81で小入力513tokenのOFF/phase-only/boundary+phaseが全て正常。
全selected/first logitsと8生成IDは同binaryOFFおよび旧同dtypecontrolに一致。
続く34,603入力/34,602batched・chunk4096・FP32・MTP4のlong fixed4096/auto24576も正常。
layer26の6行×11tensor、721,008raw bytes/armはactual byte比較で全て一致した。

phaseOFF/boundaryONの同b65 auto追加対照も正常・旧fixedcontrolと一致した。
3つの新longrunは19行×48層・各37,355,520raw bytes全てが旧fixedrawとactual memcmp一致。
旧b683autoは同じ場所から差が残り、26層/row24575以降132選択行が異なる。
phase-boundary-controls-v1.jsonと二つのanalyzeスクリプトがこの比較の再現手順と証拠。

従ってphase同期が原因を隠したとも、phase patchが修正したとも結論しない。
新binaryで旧差を再現できなかったという限定結果。原b683/44cafcc2 FAILとrawを維持する。
全token recurrent inputや全状態は比較していない。数値原因、通常経路の全品質、full256K/RESTOREは未解決。
6GPU arm全て新fault0、dump無し、boot不変、正常exit/reap/session空。正式bench性能は更新していない。

# 部品候補の次の判断

KV copy groups v2はdefaultOFFの同byteコピー。v1のruntime後setenvを廃し明示引数の共通実装へ変更、
fixtureのstagingvector/USM/graphはunknowncompletion時にunwindせずMainowner終了まで保持する。
全copy配置はk mod groupsで一意、任意validID/fullslots/int8/fp16/q4を切り捨てず、同queue順序と
既存CLOCK/counterを維持する。次はMainが直列CPU+ASan/UBSan、実SYCLbuild、GPUoracle/graph、ABBA測定。
実model改善や7秒境界waitの解消は未認定。

QSA score-only GEMM案はR593安全レビューも受領。明確なstride/causal bugは指摘されず、nonfinite、
actualheadroom/hiddenlibraryalloc、意図したFP16/BF16のroute/品質は実行前後の独立gateとして残る。
未適用である。external-session profiler v2も未適用。まず小GPU graphcapture/replayでprovider可視性を
検証する追加proposalを依頼し、実model全traceを先に実行しない。有限hosthandshakeは別診断の侵入として扱う。

R592は旧GDN tuned sourceがsycl/src/prefill/kernels.dp.cppであることを再確認した。旧SG16/GRF256は
currentnormの32laneをそのまま16laneへ変える案ではない。現在の寄与は別source/非exclusiveintervalなので
1.7%を今回の厳密Amdahl上限にしない。GDNの回復案は保持するが、CPU GUとQSA/actualprofilerを優先する。
R594はprefill1662831872Bのowned領域がdecode中も存続することをsourceで確認。cache128実hit3%に対し、
static256/384と別adaptivearm、prefill専用scratch退役/再作成は有望だが未実測。512は現在のprompt予算で境界的。
R595はlosslessCPU-GU事前展開候補。packed sourceを維持し、まず16/64expert単体prototypeを別Solへ依頼した。
nt1のsign_epi8(-128)等の整数挙動を保ち、expandedread2.3–3.1倍とpreparecostも測る。8GiB cache統合はまだしない。
R596は同queue scratch/hostsource所有権に具体的違反を発見しなかった。最新OFF対照も一致したため、追加の
selective-drain診断は現段階では実行せず、旧差を解決扱いにも変更しない。

agent原reportはbyte/hashを保存する。R590/R595原文末の空行にgit diff --check警告があったが、codeではなく
元報告hashを保持するため既存commitのbyteを変更していない。PR622のURL訂正は前Mainreviewを参照。
主branch feature/sycl、source/レポート/全標本を区切り毎にcommitする。新branchは作成していない。
通常benchmark/品質/物理262144/RESTOREとforkdefaultの非推論機能を保った反映は継続課題。
