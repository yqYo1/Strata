# 実decode censusとMTP重みの繰り返し復元

2026-10-15 JST。Arc B570 10 GiB / Ryzen5600X / RAM128 GiB、root SYCL/native IQ3_S。
`check_decode_ledger_fp32_fixed.py normal`、binary44cafcc2…。
`decode-ledger-fp32-fixed-normal-v1.json`と`decode-ledger-final-summary-v1.json`に個別標本・
source/input/command/env/owner・全joined counter・fault結果を保存した。

同serverへ34603入力/max128を二回、513入力/max8を一回送信した。prompt cacheは0。
全三件がlengthで完了し、Mainのprotocol/counter/service完了照合を通過。
正常exit/reap/session empty、fresh GPU fault 0、dumpなし、boot同一。
原自動24576-row numerical gateはFAILのまま、ここは通過済みFP32/固定4096-row配置を使った。

## Decodeの実内訳

| 項目 | 初回34K | 同server二回目34K |
| --- | ---: | ---: |
| 出力 E / verifier rows R / windows W | 128 / 217 / 55 | 128 / 197 / 50 |
| 提案D / 受理A | 162 / 76 | 147 / 81 |
| CPU expert rows / 全route entries | 101036 / 104160 | 91665 / 94560 |
| CPU割合 | 97.00% | 96.94% |
| CPU unique expert jobs | 68781 | 60701 |
| 一jobの平均token rows | 1.469 | 1.510 |
| CPU logical packed weight bytes | 140503244800 | 124070988800 |
| CPU thread-time GU / Down / quant | 25.533 / 11.102 / 0.099 s | 22.740 / 9.836 / 0.089 s |
| pool呼び出しwall累積 | 8.363 s | 7.186 s |
| Verifier GPU境界待ち累積 | 7.713 s | 6.886 s |
| verify caller wall | 22.346 s | 15.316 s |
| draft caller wall | 3.183 s | 1.673 s |
| decode loop wall | 25.544 s | 16.996 s |

thread-timeはissuerとworkerの並列CPU消費でありwallと足し引きしない。既存indexの内部scopeは
重なるので、全欄の和をdecode分解にしない。CPU MoE呼び出しとGPU境界待ちが両方大きい。
GUはCPU消費の約70%、Downは約30%、activation quantは約0.3%。次のCPU対象をGUの
実format/nt別serviceへ絞る根拠が得られた。format集約の未適用案は`decode-format-service-v1`。

同じpromptの生成列は42番目から違う。MTP受理数/検証量も異なるため、初回→二回目の時間差を
全てwarm graph効果へ帰属させない。これらは診断標本でありclean速度比較・独立task品質ではない。
約5.0/7.5 t/sを新しい正式性能や改善率とは主張しない。

## Hardwareとの条件付き比較

CPU logical weight payloadは一出力当たり約1.098/0.969 GB。今のroute・重複率のままTG70なら
76.84/67.85 GB/s相当となる。過去のRAM read control36.87 GB/sと比較するスクリーニングは
できるが、logical bytesはDRAM trafficではなく、実kernelの帯域ceilingでもない。
R579原本の10倍過小のbyte算術は使わない。

一出力の実CPU thread-timeは約0.287/0.255 CPU秒。これが不変で6 service slotを全て使える
仮定でも約20.9/23.5出力/s相当となり、GPUだけを速くしてTG70には届かない。
これは今の処理量・kernelの条件付きboundで、hardwareの絶対上限や目標断念の根拠ではない。
CPU kernel、expert所有率、weight reuse、投機効率、cache/worker設定を変える必要がある。
cache128を最良設定とは扱わず、大きいcacheの入場条件と実hit/品質/速度も別途調べる。

## MTPの正常な繰り返し寿命

二つの長いpromptでdetach→prefill→restoreの四memory markが各一回、shortでは0回。
重みsourceは初回だけ675 MiBをhash付きRAMへ保持し、その後はRAMから二回復元した。
restoreは164.675/169.670 ms。二回目のnormal round/step再capture合計は約10.63 msだった。
初回の約1323.73 msはcold graph/kernel準備を含む。
既存MTP graphを持った二回目の退役/復元/再captureと、その後の短文bypassで正常生成を確認した。
全Tサイズ・coupled・cancel・異常注入・snapshot RESTORE・物理256Kの証明ではない。

二回目のdetach前のfreeは1716322304 B、detach後2424119296 B（約2312 MiB）。
初回より少ない。prefill/library workspaceとdecode graph等が残るため、初回の空き容量を
次requestへ流用しない。FP16の34K全residual+layer cache+512 MiB reserveは約2488 MiBで、
**MTP675 MiB返却だけで二回目も全VRAMに収まるとは言えない**。
毎requestの実free queryとRAM tail fallbackを維持する。decode-cache退役案は別途未適用。

最初のrequestの稼働中抽出transcriptは最終receiptの同blockとSHA256一致を確認した。
重複した暫定抽出と成功full traceはcompact evidence確定後、manifest付きで退役する。
初回数値不一致のraw selected logitsは別の未解決consumerがあるため残す。
