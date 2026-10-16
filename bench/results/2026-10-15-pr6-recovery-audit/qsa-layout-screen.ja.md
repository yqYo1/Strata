# QSA query layout復旧の初回実測

2026-10-15 JST、Arc B570。旧当forkのlayout1に対応するquery local-memory置換を
現在のroot SYCL fallbackへ実装した。自然順decode entrypointとmatrix dispatchは維持する。
候補はexact `STRATA_PREFILL_ATTN_LAYOUT=1`のみで、defaultは自然順。

現在のconfigured compiler/oneMKLでbuild成功。runtimeを作らないhost-onlyのselector、
geometry、byte overflow拒否も正常終了した。診断付きGPUではformat0..3と
query数1/17/32、cap63/64/65/129の計20casesが通った。
自然順API/layout0/layout1の全outputとpartial scratchはbitwise一致、
独立double oracleとの最大絶対誤差は1.816e-7。
nonidentity page map、全nonresident、empty/invalid width、guard、logical262143を含む。
実モデル262144 tokenを処理した証明ではない。

次の別processではvalidation/API tracingを外し、int8、32 queries、cap2051、
全active page residentの同一入力を比較した。10warmup後に0/1/1/0順、
各sample8repeats、各arm20samples。event intervalはchunk+mergeとsubmission gapを含む。

| layout | 中央値 ms/batch | min–max ms/batch |
| --- | ---: | ---: |
| 自然順0 | 1.047806 | 1.041133–1.418594 |
| 並び替え1 | 1.090664 | 1.084531–1.484258 |

この一つのcomponent screenでは候補の時間が約4.09%増えたため、採用しない。
他のshapeを含む普遍的な遅さや実モデルのprefill速度差とは主張しない。
全40個別samplesは`qsa-layout-microbench-v1.json`、全20caseとsource/binary/環境は
各build/validation receiptに保存した。全owned processはexit0/reaped/sessionempty、
cleanup/errorsなし。両GPU jobでfresh fault0、dumpなし、boot不変。

初回診断のUR traceは4,008,952 bytes、warning0。
成功APIの代表entry/resultとhash/countを抽出し、終了済み4runのnormal text log28件、
logical4,788,515 bytesをmeasurement lock内で削除した。
元structured status/sample/fixtureは変更していない。
`closed-trace-retention-v1.json`はpath/hash/size/replacementを記録する。
物理的に空き容量がその分増えたとは測定していない。

次は同じoneMKL/library/queueで現在row-majorと旧column-majorの記述を比較する。
production GEMMを変える前のstandalone diagnosticであり、全model速度とcold-source、
実7format rotation、decodeの採否は別の検証である。
