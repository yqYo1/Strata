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
