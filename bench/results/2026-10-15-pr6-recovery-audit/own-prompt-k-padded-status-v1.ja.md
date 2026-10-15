# K一時領域のpitch40比較を閉鎖、不採用

2026-10-16 JST。前回の継続応答は方針確認だけで、新たな速度検証はなかった。
今回は独立した自作候補の全差分をMainでレビューし、実適用・SYCL build・15個のGPU process・
全出力比較を完了した。XeStrataの実装は参照も移植もしていない。

## 実装とAPI確認

`STRATA_SYCL_PROMPT_K_CELL_MAJOR=2` はK8V8/Direct1だけの明示的な実験指定。
通常の0、前回不採用の1、今回のpitch40の2を比較した。
単一の4-byte以上に整列したhalf配列へKを`dim*40+cell`で保存し、QKの全load strideも40に変更する。
同じ領域のV/PVは従来の連続8192要素とstride256を維持し、算術・barrierは変更しない。
追加の4096Bと、device local memory >=64976Bの拒否条件を持つ。
不十分なdeviceで0へ黙ってfallbackせず、明示エラーにする。

R737の全報告と、実際に使うIntelLLVMのlocal accessor/multi_ptr/matrix headerをMainが確認した。
Intelの一次仕様のBMG-G21制約はhalf byte stride80と各base offsetの4-byte整列を満たす。
実際のstrict host/device compileとlinkも正常終了し、source静的assertも通った。
APIレビューとheader/compiler hashは`own-prompt-k-padded-contract-main-v1.json`。
仕様のオンラインsycl branchは実験的で、header hashと実compilerも別に固定した。
これは実際のSLM使用量・register/spill・occupancyの測定ではない。

実buildのnative SHA256は`9753ef1eee25a767a15a7a610ebac3af7bf5e586bcb443772199212699154cde`、
componentは`48eeb3a760975096172390cfe9643a84eefe0331509486429af4d4a79dad5559`。
元のbuild FAIL/PASSや前回の全sampleは書き換えていない。

## 正しさと実測

Direct0/1 × mapping0/1/2の6 processで、各16format/page/scaledケース、合計96ケースが合格した。
全6個の1180672B出力がbyte一致し、前版fixtureのhashとも一致した。
full検査はexec前にUR tracing・parameter validationを設定し、終了まで環境を変更していない。

続いてtrace/validation/collectorを外し、独立した9 processを
`0,2,1,1,2,0,0,2,1`の順で実行した。nq256、cap2051、warmup完了後20反復。
時間はhost submissionからchecked drainまでで、oracle・出力copyは区間外。
全9個の6291520B出力がbyte一致し、全出力finite、tail guardも正常。
速度用oracleはquery0/1/2/255だけで、前の全16ケースoracleとは範囲が異なる。

| mapping | 独立sample、ms | 中央値、ms |
| --- | --- | --- |
| 0 従来 | 397.830 / 395.857 / 392.006 | 395.857 |
| 1 余白なしcell順 | 439.601 / 437.551 / 437.824 | 437.824 |
| 2 pitch40 | 639.120 / 638.908 / 638.854 | 638.908 |

pitch40は従来より中央値で61.40%遅く、全sampleが従来の全sampleを上回ったため不採用。
長いモデル入力によるmapping2実行はしない。1も引き続き不採用で、通常指定は0のまま。
15 processはすべて正常終了・全owned process退去・同boot・新規fault0・dumpなし。
SLM競合・occupancy・driverのどれが原因かは、この結果から断定しない。
モデルPP、品質、RESTORE、物理262144 contextをこの部品検査で合格とは扱わない。

実験の1/2 specializationは、Mainによる実JIT資源調査R739の比較対象として一時保持する。
採用設定では有効にしない。調査閉鎖または候補変更時に、source/binaryの必要性を再評価する。
prefill反復600超・1000、decode70の目標は未達。

## 保存と次の作業

全個別sample・exit/owner/boot/fault・source/compiler/binary/env・比較をGitへ保存する。
比較完了したraw出力、成功API trace、duplicate record、full source mirrorを
`closed-prompt-k-padded-retention-v1.json`にpath/size/hash/reasonを記録して退役した。
133 files / 72758567 Bの論理サイズで、filesystem空き容量の実増とは区別する。
元receiptは前後でhash一致を確認し、FAIL/PASSを変更していない。

31,457,280Bの最新FP32 selected-row captureと18,671,616Bのf01e reference nativeは、
次の自作selector候補の同78453入力比較のMain consumerへ引き継ぐ。
各32MiB予算、次の採否決定かreference更新時に再レビューする。

次は実JIT資源の取得方法の調査R739を確認し、約24.4秒のselection区間に対する独立した
weighted radix4候補をレビューする。selection区間にはscoresとtopkの両kernelが含まれ、
topkだけで24.4秒を使ったという測定ではない。Mainがbuild/GPUを直列に管理する。
反復調査R740はthreshold countだけの読み込み順を調べ、既存のascending emissionを維持する。
