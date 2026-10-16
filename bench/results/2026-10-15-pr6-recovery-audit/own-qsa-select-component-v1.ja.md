# 自作selector 3方式の部品比較

2026-10-16 JST。MainがR738/R740、Solの全patchと331行fixture、Mainのcyclic追加差分を
全てレビューした。XeStrata実装の参照・移植はない。実working treeへ適用し、strict SYCL buildを完了。
native SHA256 `0e62c5fa74cb6a077aef5c5977617e6d74fbd8c2c712ac8f80f3445946289a36`、
fixture `55a140792bb7528161cb38d41a44499d86c6881b51e49870bed30dd9498e1404`。

0は従来32bit threshold、1はweighted4-bit MSDで8round/各16bin reduction、
2は従来32roundのcount走査だけをcyclicへ変更する。
keyのNaN/zero順序、score producer、final gt/eq/scansと昇順ID emissionを原版とbyte照合した。
defaultは0で、`STRATA_SYCL_SELECT_RADIX4=1`と`STRATA_SYCL_SELECT_CYCLIC=1`は別々の実験指定。
両方1はsource上で明示拒否する。後者の拒否専用processは未実行。
全環境はexec前に設定し、runtime初期化後に変更しない。

## 正しさ

方式ごとの独立3processで、各17batch/674queryを全expanded-cell CPU oracleへ比較し、
3個の5450408B出力はguardも含めbyte一致した。
8patternのNaN/inf/zero/隣接値/tie、tail0..3、identity/zero-width、invalid step/host span、
actual相当78452とsynthetic78453、physical262141..262144のselector入力を含む。
物理境界は各process32rowで、engine全体の256K consume/RESTORE/品質の合格とは異なる。
scores/stepsは前後byte一致、出力IDの全件oracle/range/unique/ascending、row/outer guardも合格。

## 反復時間

cleanな独立18process、patternごとに各方式3sample。
nq256、n78453、cap2051、tail1、warmup完了後5回submitしてchecked drainまでを測る。
scoreはsyntheticで、score producerは含まない。全256queryのoracleと転送は区間外。
各patternの全9個の2100360B出力がbyte一致し、全sample正常終了。

| pattern | 方式 | 独立sample ms | 中央値 ms | 従来比時間 |
| --- | --- | --- | --- | --- |
| varied | 0 | 25.1202 / 25.0847 / 25.0507 | 25.0847 | — |
| varied | 1 radix4 | 9.86276 / 9.91622 / 9.95324 | 9.91622 | -60.47% |
| varied | 2 cyclic | 11.4798 / 11.4499 / 11.4520 | 11.4520 | -54.35% |
| equal | 0 | 24.7306 / 24.9385 / 24.6941 | 24.7306 | — |
| equal | 1 radix4 | 12.6321 / 12.6578 / 12.7149 | 12.6578 | -48.82% |
| equal | 2 cyclic | 11.0593 / 11.1518 / 11.0874 | 11.0874 | -55.17% |

両方式とも各patternで全sampleが従来の全sampleより短い。
radix4はvaried、cyclicはequalで互いより速いので、全入力で一方が優れるとは言わない。
18timing+3fullの全21processは正常終了、全owned退去、sameboot、新規fault0、dumpなし。
速度はselector単独のsubmit+drainであり、24.4秒のselection区間全てやモデルPPの改善率ではない。
score分布、実compiled transactions/register/SLM/occupancyの効果は未測定。

## 次のgateと保存

実モデルの同78453入力、thinking ON、C8192/R_GPU32768/FP32、prompt mapping0、
native/canonical bank1、coarse/fine observer1で、方式1の正しさと区間を次に確認する。
official bench.shのprefill-onlyを使う。24生成はprefill完了markerでdecode評価ではない。
selected768 FP32行のcaptureを旧fine-profile referenceと比較する。
このreport時点でモデル実行は未開始、部品の改善をモデル採用とは扱わない。
prefill反復600超→1000、decode70は未達。

全個別sampleとsource/binary/compiler/env/owner/faultを保存した後、raw出力・成功trace・
duplicate record・source mirrorを186 files / 55073738B退役した。
`closed-own-qsa-select-retention-v1.json`はpath/hash/reasonと元receipt不変を記録する。
f01e nativeとlatest selected FP32 rawはMainの次の同入力比較consumerへ各32MiB予算で維持する。
旧prompt fixture48eebはR739の実JIT資源取得consumer向けに、同じbuild場所で未relinkのまま残した。
R739のIGC/NEO dump recipeは未実行。AOTを実JITの資源測定として代用しない。

build receiptのreferencesは二つのmanifest.json basenameが衝突するmetadata上の制限がある。
両patch SHA、実source pins、両manifest本体と履歴は別途保存し、Main reviewへ明記した。
元receiptを後から書き換えていない。
