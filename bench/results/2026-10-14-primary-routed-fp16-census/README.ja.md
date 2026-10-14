# 実モデルのrouted FP16観測

2026-10-14、Arc B570 / Ryzen5600X / 128GiB。
旧stock343.8、現在compact0/1の187.5/194.2の差を説明するため、実route形状・転送要求量を取得する。
現在sourceへ完全一致STRATA_PREFILL_ROUTED_FP16_OBSERVE=1のhost観測を適用した。既定OFF。
既存同期後のne histogram、実GU/Down call/row数、producer join後の要求payloadを集計する。
新しいdevice copy/event/waitやqueue/math変更はない。転送要求bytesは成功DMA/物理PCIe trafficではない。
既存PfTimerはSYCL barrier間の非exclusive区間で、host gapと観測負荷を含む。quiet速度と分ける。

Mainがbuild/test/GPUを直列に管理する。旧636nativeはdefault-OFF数値/観測overhead consumer、
128MiB以内、次reviewはその比較終了。新build4GiB、各diagnostic64MiB raw text・1GiB job、
短いtensor pair16MiB、品質/正式速度/262144入力/RESTOREの合格に数えない。
既存short residual/logit fixtureを保持し、閉じたrunを上書きしない。

Luna R519は実選択の旧/現GDN差分、R520は無変更本家clientでの長いdecode反復を調べる。
過去registry共有、FINALまで介入せずMainが実装・buildを継続する。調査は性能の実測ではない。

ビルドnative68da0262、既存host range contractは正常終了した。全SYCL TUのsize_t range旗を
compile databaseで再確認し、コンパイラ/runtimeやGDN/GEMM kernelsを変えていない。
129 batched tokensの観測OFF/ONは両方とも正常native/owner終了、fault0、dumpなし、同じboot。
sampled residual30720要素とfirst logits248320要素がfinite・bit一致し、変更前636nativeの
保存済みfixtureにもbit一致した。観測ONでexact histogram、GU/Down rows/calls、payload分類の
整合を確認し、OFFで観測出力がないことを確認した。[数値・観測結果](numeric-results.json)。
これは合成短文の確認で、言語品質・長文速度・full-input/RESTOREの合格ではない。

次は`check_long.py phase`による32768以上の実batched入力、chunk4096、context262144の
補助CLI診断。本家clientの公開word-listから生成し、serverと同じtokenizer/templateを用いる。
input/body/rendered hashesを保存するが、以前のtimestamp付きstock promptとのbyte一致は主張しない。
1出力の全logitsを確認し、同じsource/runtime policyでroute形状・要求payload・既存phase区間を得る。
timing barriersによる負荷を含み、本家bench.shの正式速度には数えない。
その後の`reference`/`census`も同じ入力recipeを使い、観測負荷とphase解釈を確認する。

R519は実選択の旧kernelに4つのcolumn分割と先読みがあり、currentとは異なることを確認した。
48 value headsと36 GDN layersを混同せず、分割kernelに必要なnorm completionも考慮する。
R520は本家32Kの24出力capをCLIで増やせないことと、補助clientの条件を記録した。
いずれもsource候補で、原因・速度・採用の確定ではない。[原文とMain訂正](research-index.json)。

evidence commit7bb5b19aをbyte/hash検証後、閉じた成功ログ・重複tensor・research原文を34件、
2,626,725 logical bytes退去した。[manifest](deletion-manifest.jsonl)、[集計](retirement.json)。
実空き容量差は未測定。長文controllerが使うraw build/numeric gate recordsとcanonical compact1
tensor fixtureは保持し、元結果の状態を変更していない。再開はGitのレポートから行う。

## 実34Kの観測完了

long phase v3は実入力34561、batched34560、8*4096+1792の9chunks、ctx262144で正常終了した。
全first logitsはfinite、native/owner正常0、全child回収、source/binary一致、fault0、dumpなし、同じboot。
v1/v2はそれぞれ管理Pythonのregex不足、保存argsにない--serveの削除でGPU起動前に失敗した。
元FAILED状態を別JSONへ保持し、既存venvへのre-execと実argsの確認をしてfresh v3を実行した。

| 非exclusive区間 | 秒 | timeline比率 |
| --- | ---: | ---: |
| gemm down | 61.535 | 29.68% |
| qsa attn | 49.455 | 23.85% |
| dequant | 32.529 | 15.69% |
| gdn recurrence | 3.543 | 1.71% |

wall207.576秒、記録区間はwallの99.88%。cold JIT、host/queue待ち、既存timing barrierの負荷を含む。
Down区間をDown GEMMだけのservice時間にしない。QSAの実matrix/fallback選択も未観測。
GDN移植を先行せず、QSAとcopy/stager/dequant・small-shape dispatchを先に切り分ける。
標準quiet194.2 t/sとの直接比較や支配的原因の確定に使わない。

routed expertの要求payload183,862,579,200 bytes、89,743コピー試行すべてstager経由、
direct-pinned試行0。legacy『0 by DMA』はdirect-pinned統計で、stagerのH2Dがない意味ではない。
host staging75.519秒は上の区間と重なるので加算しない。
以前のhost-USM到達6.447 GB/sへ単純に割ると28.519秒だが、実DMA/物理trafficや絶対下限の測定ではない。

GU/Down各90,297calls、各16,588,800rows。active callのne中央値12、90%点428、99%点3245。
ne<=16はcallsの55.13%だがrowsの1.39%。一律80行の合成GEMMを実分布として扱わない。
exact histogram/format/payloadは元receiptに保持した。[観測概要](long-phase-summary.json)。

次は同じinput/body/hash・native68da0262・fresh CLI/runtime policyの`check_long.py reference`。
観測OFFの全first logitsとwallを照合し、必要ならcensusのみを分離する。正式bench.sh比較は別。
phaseのfirst logits/token IDsはそのconsumer用、16MiB以内、比較終了時に見直す。
[保持計画](long-phase-retention.json)。

phase evidence a8212928を検証後、成功全文・重複receipt/research11件、466,579 logical bytesを
退去した。[manifest](long-phase-deletion-manifest.jsonl)、[集計](long-phase-retirement.json)。
元のstatus/route histogram/format/phase/owner/fault結果はbyte-identical archiveで保持し、
比較中のfirst logitsとtoken IDsは残した。実空き容量差は未測定。phase raw recordは退去済み。
R521が参照したsummaryの旧hashは、注釈3fieldを除くと同じbytesに復元できることを確認した。

## 同一入力の無観測CLI比較

実34561入力・34560 batched、同じ68da0262 binaryで無観測referenceを正常終了した。
first logits全要素はphaseとbit一致、fresh fault0、dumpなし、boot不変、owned process全回収。
prefillは無観測201618.3ms、phase207565.9ms（観測差+2.95%）。各1processなので
安定した計測overheadや因果関係の証明ではなく、正式bench.shの速度値でもない。
[照合結果](long-reference-comparison.json)に元receipt・入力・正常終了を記録した。

R523/R524と別SolのQSA transpose案が完了した。kernel/headerの未適用patchであり、
実装・build・GPU・採用の合格に数えない。ユーザー指定のagent sessionと長いMTP生成の
benchmark追加を先に進め、QSA route countersとkernel評価はその後へ継続する。


R531/R532の読み取り専用Luna反復調査が完了した。Mainは返答まで介入せず、live v4測定と
独立したcaller-only QSA route censusの[未適用実装案](qsa-route-observation-proposal-v1/report.md)を
準備した。元の一度だけのbool呼び出しとfallback loopを保持し、host返却結果/shape/query数を
数える。追加queue/event/copy/wait/math/能力再問い合わせはない。kernelの実行時間や選択cellの
占有を測ったことにはしない。decline理由は未観測と明記する。build/test/GPUは全て未実行。
benchmark→PR#6点検の順を保ち、その後にsource baseを再確認して観測案を直列検証する。


R533はspec4/serveのhost pin→Stager作成順と、Linux親affinity継承を確認した。
R534は正常generation/event/sourceの並行契約と、wait/copy/eventRecordのreturn無視を指摘。
Mainもadapterがbody前にsticky errorをcheckするため、失敗waitが実event waitなしで返り得る経路を
[確認](stager-error-ownership-review.ja.md)した。hangやraceの実再現ではない。
別Solの[affinity案](stager-affinity-proposal-v1/report.md)は原allowed maskをvalueで渡し、workerの
配置を読戻し確認するdefaultOFF案。CPU0..62のLinux/SYCL限定、他maskは明示refuse、
issuer/pool/event/math/queueは変えない。全fixture/build/GPUは未実行。失敗伝達/lease問題は
この案に含まれず、Mainは追加レビュー・CPU error fixtureを行ってから並列配置GPUを検証する。
source/code hash付きでGitに保持し、benchmark→PR#6の順を保つ。
