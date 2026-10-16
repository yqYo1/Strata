# 長文prefillへ測定を切り替えた記録

2026-10-15。長文prefillが600 tok/sを超えるまで集中する新指示に合わせ、未終了の旧IQ ONの残りの
long-generation比較を意図的に中断した。stop_iq_on_for_prefill_priority_v1.pyは実在する旧leaderと
bench childのPID/start/SID/PPID/cwd/argvを確認し、pidfdでHTTP benchmark client一個へSIGINTだけを送った。
native、GPU、foreign processにはsignalを送っていない。旧runnerのfinallyがserver/nativeを通常終了した。

exec10041はexit1で終わった。元benchは-2、server/nativeは0、全owned commandはreap/session empty/
survivorsなし、旧leader323620/start39561427・native323699/start39561505・bench326347/start39568041は
消滅を確認した。same boot、新規GPU fault0、driver dumpなし。元selective-iq-clean-bench-full-1-v2.jsonは
passed=false/complete=falseのまま保存する。これは故障での停止や全suite合格ではない。

## 閉じたprefixの実証拠

閉じたOFF/ON両journalのhash-chain/state/pendingを実検査した。ONはvalidだがcomplete=falseであり、
未完のsuiteを合格に変えない。42件のresultそれぞれの実request SHAは対応するOFFと一致した。
それには中断されたvaried-proseのresultも含み、成功件数42という意味ではない。
selective-iq-on-completed-prefix-v1.jsonに全42個のinput hashと元sampleを保存した。

nominal32Kの実入力は32089で、ON PP216.8/216.0/216.1、OFF200.5/199.8/199.5。
三回のON中央値216.1は約8.16%高いが、実32768以上の採用試験と呼ばない。
唯一終了したlong-codeは33214入力/3346生成、PP211.2、TG9.5、draft2382/2895、eligible=true。
次のproseはKeyboardInterruptで5468生成までの不完全response、usage/PP/TGなし、eligible=false。
これをPP/TG標本へ補完しない。全long-generation分類/反復比較は未完了である。

## 公式bench.shへのprefill-only mode

Solのbench-prefill-only-v1案をレビューし、旧native退去とlock取得後にhost fixtureを実行した。
proposal五testが通り、source.diffをrootのtools/strata_bench.pyとbench.shへ実際に適用した。
適用後の既存benchmark五test、failfast七testも通った。
evidence十四testの一件は、古いfixtureが比較元のnonce/workloadを持たないため失敗した。
以前から強化されていた比較拒否は正しく動いており、新modeの速度問題ではない。
元bench-prefill-only-host-v1.jsonをFAILのまま保存した。

fixtureに正しいworkload identityを与え、異なるnonceと欠落nonceなら比較率を出さないことも追加確認した。
validationは緩めていない。別root/receiptのqualify_bench_prefill_only_v2.pyで十四testが全件PASSした。
実検証した合計はproposal5+既存5+failfast7+evidence14=31件。全command退去済み。GPUは使っていない。
root client SHAはbe421b2775569cd1bd937a35c606985c532c7d98f344a42d119a005ae54e9559。

新modeはthinking有効、24-output capの短warmup一回と長文三要求だけを実行する。
--prefill-tokens 32768/65536は実usage.prompt_tokensと実timings.prompt_nの最低値であり、
nominal32089を32768以上と認めない。explicit reuse0、実context/output余白、有限正のPP、正常完了を確認する。
固定workload IDでbodyを生成し、応答に合わせた修復/再試行/calibrationはしない。
既存の通常suite/agent/long-generation機能は維持している。

run_prefill_only_clean_v1.pyは同じb1cc9cb4 native、C4096/R_GPU4096 FP32、MTP4/context262144/K8V8を維持し、
IQ1、65536 minimum、三回、公式bench.shの新modeへ切り替える実recipeである。
新rootはB/main-prefill-only-clean-iq1-c4096-root-v1、最終receiptはprefill-only-clean-iq1-c4096-v1.json。
launch前のsource/linked/model/host/normal-retirement/fault確認、有限5400秒、RSS88GiB、raw text64MiB、
job1GiB、lockを持つ。性能・品質・物理256K/RESTOREの採用値はrunが実際に閉じるまで記録しない。

## FP16/BF16とGPU内の処理

保存はどちらも2Bであり、同じ配置なら転送量削減は同じ。FP16はfraction10/exponent5、BF16はfraction7/
exponent8で、精度と範囲の取引になる。一次資料はIntelの
[BF16説明](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-deep-learning-boost-new-instruction-bfloat16.html)。
FP16を普遍的な勝者とはしない。保存済み自然historyの一castではFP16の丸め誤差が小さかったが、
繰り返しのinter-layer castでの全token NLL/task品質を代用しない。BF16 storageは現root未実装である。

GPU内で追加計算してでもCPU往復を減らす方向を優先する。ただしrouter groupingをGPUへ移すだけでは、
oneMKL呼び出しのhost M/countが必要な依存が残り得る。どの転送/待機が消えるか、GPUが行う計算と
VRAM所有量がどれだけ増えるかを実際のdispatch全体で調べる。追加計算時間と削減したCPU/転送/同期待ちを比較する。

## R669/R670のMain訂正

R669 v1は600/1000 tok/sを600K/1M入力と取り違えた。MainはFINAL後に指摘し、別v2で訂正した。
入力は実32K/64Kを対象にし、context上限262144を超える案を実行しない。
legacy PfTimerのunchecked eventを流用しない。ITTはsource/READMEにあるが、Mainが現固定providerの
dynamic symbolsを調べた限り__itt_pause/__itt_resumeは見つからず、実buildでのhook利用は未確認である。
documentだけで実接続済みと呼ばない。名前付きunitrace sessionの外部制御等、同じ環境を変えない手段を確認する。

R670 v1はC4096のworkspace確保後freeをC8192へそのまま流用した。MainはFINAL後、no-borrowでは
増加workspace1210776832 Bを先に引く必要を指摘した。別v2ではfree2423623680から1212846848 Bとなり、
cache+headroom1900019712に687172864 B不足すると訂正された。R_GPU0でもこの8K armのadmissionは
裏付けられない。盲目的に実行せず、所有領域/配置を先に改める。全32K残差VRAMまでにはさらに容量が必要。
両v1は原bytes/hashのまま保存し、v2とMain訂正を優先する。

Mainは新しいLuna R671にGPU内router→GEMM発行の依存、R672にprefill中head owner退去の具体的契約を
別々に依頼した。過去report/indexを共有し、FINAL前に介入しない。これはsource調査で、まだ新しい実装/速度ではない。
目標1000 PP/70 TG、直近600超え、品質、物理262144/RESTORE、上流全機能/default反映は未完了である。

全staged diff --checkは保存用source.diffの空行context spaceを警告した。hash済みpatchをそのまま維持し、
この一artifactを除いた実source/docsのwhitespace確認と区別する。
