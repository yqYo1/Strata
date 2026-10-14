# 34K FP32 gateの数値不一致

2026-10-15 JST。`mtp-expert-release-long-v1.json`は **FAIL / incomplete / inactive** のまま保存する。
同実行は正常exit/reap/session empty、fresh GPU fault 0、devcoredumpなし、boot同一。
生成1IDは同じだが、同dtypeで演算を変えない変更に対するbitwise regressionは通過しなかった。
GPUハングではない。FP16 armは開始していない。

## 比較条件

controlは`selected-prefill-long-v1.json`のFP32。新armは同じthinking-template履歴の
34,603入力/34,602 batched、chunk4096、context262144/KV8 resident32768/cache128perlayer/
pool5+host/MTP4 splitOFF/maxnew1。engine argsは同一。新armだけMTP expert退役を指定した。
入力、model、runtime、安全設定は共通で、diagnostic path/cache directoryは別。
旧binaryはc2e46f2f…、新binaryは44cafcc2…、変更sourceはMTP/headerと生成器wrapper、
新しいRAM source helper。build receiptと全source pinsを別途保存している。

MTP675 MiBの返却後、FP32 residual planが自動的に以下へ変わった。

| 項目 | control | 新arm |
| --- | ---: | ---: |
| GPU rows | 4,096 | 24,576 |
| GPU bytes | 167,772,160 | 1,006,632,960 |
| host bytes | 1,249,525,760 | 410,664,960 |
| temporary layer cache bytes | 1,363,148,800 | 1,363,148,800 |

新armのdetach前後のfreeは2,229,821,440→2,937,614,336 B。prefill終了後・restore前は
2,763,431,936 B、restore後は2,053,406,720 B。detach1.161 ms、restore160.250 ms。
これは実free queryであり全差分をpayloadだけに帰属させない。diagnostic prefill176,222.4 ms
は独立したclean speed sampleではなく、数値gateもFAILなので採否に使わない。

## オフライン比較

`analyze_mtp_release_divergence.py`をGPU jobの終了後、同じ測定lockで実行した。
`mtp-expert-release-long-divergence-v1.json`に全19位置の個別比較を保存した。
helperの固定scope文にsynthetic-historyとあるが、今回のinput identityは上記自然履歴であり、
receipt/hashがdatasetの根拠である。比較結果そのものは変更しない。

- 位置0,1,63,64,255,256,257,4095,4096,8191,8192の11点は全248,320 logitsがbitwise一致。
- 16383以降の8点で不一致。各点のmax absoluteは約0.035–0.332、KL最大約6.08e-5 nats。
  32768位置のrelative L2は約0.02435であり、単なる1ULP差と分類する根拠はない。
- 全値はfinite、選択位置のtop5は同じ。最初の生成位置の全logitsも不一致だった。
  同じ生成IDや小さいKLだけでメモリ/順序の不具合がないとは判定しない。

これらは選択位置なので、最初の不一致位置はまだ分からない。
現在のsource監査でGPU/host span、copy/drain、ownerに明白な違反は見つかっていない。
許容誤差を緩めたり、原gateをPASSへ更新したりはしない。
次の`long-fixed`は同一新binary・MTP退役ONでGPU rowsだけ4096に固定する別実行。
必要なら選択rowのstore前/保存後/次layer restore後を対にする有限観測で局所化する。

## 4096行固定の結果

別receipt `mtp-expert-release-long-fixed-v1.json`はPASS/complete/inactive。
19位置すべての全logits、metadata、末尾first logits、生成1IDが旧FP32 controlとbitwise一致。
process正常終了、fresh fault 0、dumpなし、boot同一。675 MiB退役ONのまま、GPU rows=4096、
host bytes=1,249,525,760 Bで確認した。detach4.952 ms、restore167.063 ms。
diagnostic prefill194,926.8 msも速度採否には使用しない。

従ってMTP source/release変更と旧residual配置の組合せはこの34K gateを通過する。
大きなGPU prefixとの相互作用、copy経路、memory pressureによるlibrary algorithm選択、
同期・状態依存をまだ区別しない。自動24576行armのFAILは維持する。
この通過済みFP32/4096配置で、別の所有付きserve診断から実decodeのjoined ledgerを取得する。
FP16/自動配置へ置き換えて初回の不一致を覆い隠すことはしない。

## R580/581のMain review

R580のspan/checked-wait所見は現sourceと整合。source hash単体とbinary provenanceは別だが、
Mainには実build command、終了、binary hash、compile flag proof、build前後のsource pinsがある。
現hashだけから全buildが未知とはしない。一方、この証跡だけでcopy/mathの正しさは証明しない。

R581はportable selectorの整数reduce/index順tie処理に浮動atomicを見つけていない。
ただし実際のnative batch selectorや全QSA/GDN routeのruntime対応は未観測であり、
kernel名だけで原因を決めない。原本の「MTP release occurs after prefill」は誤り。
実際は **prefill前にdetach、prefill後にrestore** する。QSA score演算variantの存在も
本比較でrouteが変わった証拠ではない。

初回不一致のselected logits/metadata/first logitsは、Mainの配置・境界比較をconsumerとして
外部job rootに最大64 MiBを保持する。解決または比較fixture更新後に再検討する。
原gateと元recipeはcommit5fbcbb16から復元可能。成功full trace・重複captureは必要性を
確認後にmanifest付きで退役し、数値FAILやfresh-fault statusは書き換えない。
