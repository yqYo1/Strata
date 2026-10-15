# Prefill scratch aliasの実確保と容量差分

2026-10-15。MainがSolの`prefill-scratch-alias-v1/source.diff`を適用し、
`STRATA_SYCL_PREFILL_SCRATCH_ALIAS=1`のときだけemb/mixed/bo/ple_embを同じ領域へ配置する。
既定OFF。明示in-order queue・単一GPU・全層配置を要求し、split/peer/handoffを拒否する。
演算の精度・順序は変更せず、同時に生存しない三つのN-wide確保を省く。
C4096で125829888 B、C8192で251659008 B削減する。

## 実行した確認

`qualify_prefill_scratch_alias_v1.py`はproduction prefill object/nativeをcompile/linkし、
実production TUをincludeするhost fixtureの11プロセスでprefix carve、重なり、canary、
サイズ、GR/BF16分岐、不正なflag拒否を確認した。ASan compileはdeviceへsanitizer指定が
伝わったため失敗した。元FAILED receiptは不変で保存した。
`qualify_prefill_scratch_alias_host_v2.py`で指定を`-Xarch_host -fsanitize=address,undefined`へ
直し、実sanitizer symbolを確認して同じ11プロセスが通った。計22 hostプロセスPASS。
警告抑制やテスト条件の削減は行っていない。

`check_prefill_scratch_allocation_v1.py`はfixture側で`bytes_needed==carve`を期待した誤りにより
native exit88で停止した。実際にはAPIが末尾に8 MiBのloan slackを足す。runtime/queue初期化後だが
GPU allocation/copy前で、fresh fault0、dumpなし。元FAILED status/hashは保持する。
v2はproductionを変更せず、期待値とfixtureだけを訂正した。host-only planで先に次を確認した。

| chunk | 実carve | bytes_needed、alias ON |
| ---: | ---: | ---: |
| 4096 | 1503447296 B | 1511835904 B |
| 8192 | 2588395008 B | 2596783616 B |

実GPUでC8192 owned allocation/release、実carveぴったりのborrowed init、C4096 relayout、
両carve末尾とslack末尾のcanary、C4096実carveから1 byte不足時の拒否、null/out-of-order queue、
splitの拒否を確認した。native0、全owner退去、同boot、fresh fault0、dumpなし。
これは262144-capacityの一層分streamed-KV stageを含む**確保/再配置だけ**のfixtureであり、
KV state payloadや推論kernel、品質、RESTORE、最大context model実行を確認したものではない。

C8192 owned初期確保前free10623217664 B、確保後7940370432 B、Prefill破棄後9792364544 B。
破棄後にも初期との差830853120 Bが残った。providerのpool/cacheや別ownerを分離していないため、
この差を特定済みのリーク又はruntimeの固定必要量とは扱わない。実数値はreceiptに残す。
現容量表の512 MiB runtime余裕は実測最大ではない。1 GiB余裕での感度表もJSONへ追加した。

## 容量計算の更新

最大一層の実packed weightはlayer47、1440161792 B =1.341255 GiB。
一層bank/no decode・MTP weightsというユーザー仮定を継承し、全12 QSA stateをGPUに置く。
FP16 hidden単価は20480 B/token。KV+indexer等はK8V8で14476*C+43200 B、
仮定のK8V4で11596*C+43200 B。pageは4 token単位で切り下げる。

actual streamed carveから、resident時には使わないstage276825088 Bを引く。
未使用GEMM workspace32 MiB削除後の値を基準にalias節約を追加した。
その他16 MiB small-owner/512 MiB runtime余裕とGDN/PLE/tokenの控除は元表と同じ。

| chunk | alias | resident work | KV+hidden予算 |
| ---: | --- | ---: | ---: |
| 4096 | OFF | 1352452096 B | 7201798144 B |
| 4096 | ON | 1226622208 B | 7327628032 B |
| 8192 | OFF | 2563228928 B | 5991004928 B |
| 8192 | ON | 2311569920 B | 6242663936 B |

| chunk | alias | KV | KVと全hiddenを同じCにする上限 | 256K KV保持後のhidden行数 |
| ---: | --- | --- | ---: | ---: |
| 4096 | OFF | K8V8 | 206020 | 166355 |
| 4096 | ON | K8V8 | 209620 | 172499 |
| 4096 | OFF | K8V4 | 224520 | 203219 |
| 4096 | ON | K8V4 | 228444 | 209363 |
| 8192 | OFF | K8V8 | 171384 | 107234 |
| 8192 | ON | K8V8 | 178584 | 119522 |
| 8192 | OFF | K8V4 | 186772 | 144098 |
| 8192 | ON | K8V4 | 194616 | 156386 |

最大context262144の全hiddenは5 GiB。全QSA stateはK8V8で3794839744 B、
K8V4で3039865024 B。C4096/alias ON/K8V4でも、全hiddenまで持つには1080946112 B不足する。
一方、現在の一層のQSA stateだけGPUへ置くなら、同条件で全hidden後に1644080368 B余る。
これは一層KV/indexerのRAM退避・復元を実装して初めて成立する別案である。

R692原本には古いworkspace込みの表と、新しいworkspace削減後の説明が混在している。
冒頭のC8192 OFF予算5892438656 Bは元表5957450240 Bとも違う。
Mainは原本を変更せず、本表と`prefill-capacity-alias-delta-v1.json`を採用する。
C4096のhidden増分は現no-workspace基準では6144行であり、原本の6145行を転用しない。
`measure_prefill_capacity_alias_v1.py`は全caseで上限page/次page、上限row/次rowを検査し、
streamed実carveとの一致を確認した。新GPU推論は行わない。

K8V4は現在のlayer-majorで拒否される仮定。FP16保存の実用品質も未合格。
さらに一層combined weight bankは未実装である。NativeDense-only bank提案はcanonical全体を残すため、
この容量表の一層配置を実現したものではない。正確な容量保証には実model/allocator/最大contextの
admissionとRESTORE確認が必要である。aliasは後続の同binary OFF/ON境界照合を通してから評価する。

## 新しい調査の扱い

R690のsource snapshot093af5f…はalias適用後、測定binary5de4…は適用前source3e36c1…である。
両者を同じbinary sourceと扱わない。clean実78453入力の三回は235.3/239.5/239.2 tok/sで閉じた。
R691はNativeDense componentのsourceレビューのみで、Mainの全caller再compileはまだ予定である。
R693はGDN単独では600を説明しないと正しく限定している。既知profileではGDN約3.54秒に対し
attention約44秒であり、現在の容量/hidden転送削減とattention最適化を優先する。

## ログの保持

閉じたallocation v1/v2の全UR history124485 Bを、必要なalloc/copy/free/drain/release、
全API件数、失敗点、個別carveとcanary結果へ抽出・照合して削除した。
`prefill-scratch-allocation-log-evidence-v1.json`89432 Bと
`prefill-scratch-allocation-log-retention-v1.json`に保存した。
元FAILED/PASS receiptのbytes/status/hashは不変。controllerが参照する原receiptは残す。
