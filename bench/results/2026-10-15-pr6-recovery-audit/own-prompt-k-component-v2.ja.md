# 独立K読込案のcomponent検証と不採用

2026-10-16 JST。XeStrataは性能の参考値のみ。本変更はこのfork自身のprompt_attention.cppと
R725のアドレス解析から独立実装した。Xe source・helper・testを使っていない。

## 実装とbuild

Mainが6.1 Solのv1/v2の全差分をレビューした。default OFFの
STRATA_SYCL_PROMPT_K_CELL_MAJOR=1はMode1 K8V8/Direct1だけで、Kコード読込をcell-majorへ
変え、元のSLM [dimension,cell]へ書く。演算、V、scale、barrier、CH32/WG128は保持する。
flag0とDirect0は元の読み方。新しいactual SYCL manual fixtureは、process環境を途中変更しない。
Mainだけがbuild/test/GPU処理を監督した。

最初のstrict buildはfixtureのunused scratch引数でFAIL。GPU未実行で全owner退去した。
元own-prompt-k-cellmajor-build-v2.jsonは変更していない。
Mainはmatrix fixture内で(void)scratchと明示する一行だけを修正した。
fix receiptはbefore/after hashと差分を保持し、test期待値やkernelは変えていない。
rebuild v3はPASS。native e4565a32d5453f9c6caf43f8d4606363dd44f395c7127f100b649232b3e22538、
component 0315eb51af4db11a9eb5be1c5cddff9f5c5926c9e2fdcd552563f7e935d0e72c。

## correctnessと個別性能

Direct1/0 × flag0/1を四つの別processで実行した。各16case、全format/page/scaled、
FP64 oracle、finite、mask/empty/invalid width、tail guard、batch/single比較はPASS。
四つのdump各1180672Bは全byte一致。CPU logical tile proofもall256dimension/32slot、
partial、invalid row、page indirectionで一致。これはSLM実体の直接captureではない。
初回debugではflushed UR tracingとparameter validationをexec前に設定した。

続いてvalidation/collectorを外した別processでnq256/cap2051を20回呼ぶcomponentを測定した。
一度completed warmupした後のsubmit～final drainだけを測り、copy/oracleは区間外。
benchmarkのscalar oracleはqueries0/1/2/lastだけ、全出力finite/guardは全件検査し、
六つの全query dump各6291520Bは全byte一致した。

| 実行順 | flag | 20反復 total ms |
| --- | ---: | ---: |
| 1 | 0 | 396.634 |
| 2 | 1 | 442.666 |
| 3 | 1 | 441.227 |
| 4 | 0 | 398.779 |
| 5 | 0 | 393.245 |
| 6 | 1 | 441.371 |

medianは396.634→441.371ms、+11.279%。ONの全三標本がOFFの全三標本より遅かった。
このcomponent screenでは不採用。long model ONを走らせず、default flag0を維持する。
これはcomponentの結果で、whole-prefill PPや品質・物理262144/RESTOREの検証ではない。
source strideだけからSLM conflictが原因と断定しない。全10GPU process正常終了、
全owner退去、sameboot、新規fault0、dumpなし。
command/environment/source/binary/個別結果と比較は各receiptおよび
own-prompt-k-cellmajor-comparison-v2.jsonに保存した。

## 次の候補

R735/R736はMain全読した。pitch33はhalf byte-stride66でAPIの8-byte倍数制約に合わない。
R735が挙げるpitch40/48はstride条件を満たす候補で、速度・実bank数・occupancyの測定ではない。
R736のCH16/64/WG64も算術と配置の仮説で、採用結果ではない。

FINAL後に別のsource-only padded案を6.1 Solへ依頼し、返却された全差分もMainが読んだ。
own-prompt-k-padded-v1はflag0/1を維持してflag2でK pitch40を追加する。
alignas(4)の一つのhalf array objectをlocal_accessorに置き、type punningや小配列を跨ぐ
pointer算術を避ける。全QK pointer/strideとboundsを更新し、V/PVは元のstride256。
candidateだけの64976B admission、device-copyability/alignment静的検査、CPU footprint proofを持つ。
これは現在未適用・未compile・未test。installed compilerの契約照合とMainのqualificationが先である。

現在のflag1実装は次のpadded比較のnegative controlとしてのみ保持する。
性能採用ではなく、次の三arm比較が終われば再審査する。600/1000 tok/sは未達。
実モデルでの性能採用には同じ長入力と独立反復が必要で、component値をPPへ換算しない。

## retentionと状態

閉鎖後、exclusive lock下で96files/48608516Bの成功trace、duplicate record、比較済みrawと
旧full proposal mirrorを退役した。全個別sample・元FAIL/PASSとhashは不変。
closed-prompt-k-component-{compact,retention}-v2.jsonにpath/hash/reasonと必要なerror/trace windowを保存。
旧f01e nativeはMain padded/defaultOFF regression consumer向けに32MiB budgetで保持する。
fine-model selected raw31457280Bも次の同78453モデル比較consumer向けに保持する。
この境界でbuild/test/GPU作業は全閉鎖。次はpadded案のsource/API/build/component qualification。
