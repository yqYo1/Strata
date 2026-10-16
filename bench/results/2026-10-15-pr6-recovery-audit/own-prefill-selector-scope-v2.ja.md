# 独立selector実験の長文prefill専用化

2026-10-16 JST。Niko由来の個人forkと独立した自作候補だけを使用。
XeStrataは性能参考に限定し、そのsource・helper・test・保存案を利用しない。

共有APIで実験flagを読むとdecode/verifyにも候補が適用されるため、専用の
`qsa_block_topk_prefill`を追加した。SYCL layer-major prefillの2 callsiteだけが使う。
従来APIとreferenceはflagを読まずMapping0。短文経路、core layer/verifyは従来APIを使う。
device kernel templateは修正前後でbyte一致。両候補はdefault OFFのまま。

Mainが4 fileの全diffと寿命・分岐・検証をreviewし、strict SYCL compile/linkを実施した。
実native SHA256は`beda08d42902471475f37ce0ee1c43baadbcc527e44a5f7df9217ffc100a93ef`、
fixtureは`aa1360cc30db3d5ea2bf14233ced9db5dc1e4a0834248e2d5197e3f375f318a0`。
build receiptとMain reviewはそれぞれ`own-prefill-select-scope-build-v1.json`と
`own-prefill-select-scope-main-review-v1.json`。

3個の独立した初回診断processで17 batch/674 queryを専用・従来・referenceの各APIに通し、
CPU展開oracle・guard・入力不変を検証した。合計6066 API queryがPASS。
78452/78453と262141..262144、NaN/inf/zero/tie/tail、無効stepとhost spanを含む。
全dump 5,450,408Bは旧fingerprintともbyte一致。262144はselectorの行境界検証であり、
エンジン全体のphysical context検証ではない。両flag同時1の独立した拒否実行は未実施。

続くclean 18 processは入力patternごとに各arm3回。nq256/n78453/cap2051、
各process内5反復のsubmission+drain時間。CPU oracleと従来/referenceの検証は時間外。
環境はexec前に固定し、collector/tracing/validationなし。全256 query・guard・入力不変がPASS、
全owned正常退去、同boot、新規fault0、dumpなし。個別値と実行順は
`own-qsa-select-scope-bench-comparison-v2.json`に保存した。

| 入力pattern | Mapping0 median ms | radix4 median ms | cyclic median ms |
| --- | ---: | ---: | ---: |
| varied | 25.1779 | 9.85398 | 11.4546 |
| equal | 24.6856 | 12.7100 | 11.1483 |

時間削減はradix4が60.86%/48.51%、cyclicが54.51%/54.84%。synthetic scoresに対する
selector単体の結果で、score producerを含まない。専用化後の実モデルの正しさ・clean速度、
task品質、実RESTORE・captured graph復元・physical262144は未検証。採用は未確定。
旧共有dispatchでの長文診断346.6/348.0 tok/sをこのbinaryの結果として扱わない。

閉鎖・比較後に183 files / 55,851,085Bの重複raw・成功trace・full source mirrorを退役した。
`closed-own-qsa-select-scope-retention-v2.json`が全path/size/SHA/reason、compact evidence、
元receipt不変を記録する。独立したcompact patch/manifest/prepareと実sourceをGitに保持する。
次は同じ新binaryのactual78453モデル比較とclean反復。attentionの実JIT資源取得も継続する。

## 専用化後の実モデル閉鎖

Mainの`run_own_qsa_select_model_v2.py 1`はactual78453/reuse0/thinkingON/24生成で正常閉鎖した。
実nativeは上記beda08、radix4は長文prefill専用APIだけに適用。48層全activation→Full復元、
native/server0、全owned退去、同boot、新規fault0、dumpなし。768 FP32行/7864320値は旧referenceと
byte一致・finite、同runの752 store/inputも正常。比較と元receiptは`own-qsa-select-model-a1-comparison-v2.json`、
`own-qsa-select-model-a1-v2.json`。生成tokenの一致は合格条件にしない。
診断1回は345.3 tok/s。新binaryのclean baseline/candidate反復は未実施で、速度の採用は未確定。
比較後に重複raw・成功trace等26 files / 32,465,083Bを退役した。
