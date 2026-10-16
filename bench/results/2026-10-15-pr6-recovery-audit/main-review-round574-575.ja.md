# R574/R575 Main corrections and next candidates

2026-10-15。Luna2件はMainのbuild/CPU gatesとselected-head GPU diagnostic準備に
並行してFINAL受領。研究者は実行・製品編集をしていない。

## R574: decode cache退役とVRAM算術

既存cache open/fillとnative hit recaptureは使えるが、serveではVerifierのcacheを
読むall-T graphsだけをretireし、MTPがborrowするfinal_R arenaは維持する必要がある。
通常sessionのpre/post graphsまで全廃する必要はない。Prefill loanを最初から無効にし、
全readersの正常終了、旧map切離し、refill完了後のhits/map再公開を順序づける。
実装・性能・異常時の寿命は未検証。

R574の空き容量の算術は修正して使う。1960 MiBは実行後であり、候補確保前の
admission基準ではない。2343 MiBはstandalone prompt buffers確保後の表示。
さらに一時layer cache1300 MiBとGPU headroom512 MiBを必ず引く。
以下は同じallocator状態で解放量675 MiB/約260 MiBが全て戻ると仮定した算術で、
実測free・確保成功・速度を表さない。

| 全GPU residual候補 | residual MiB | cache+headroom込みMiB |
| --- | ---: | ---: |
| 34,602 batched rows FP16 | 675.8203 | 2487.8203 |
| 34,602 batched rows FP32 | 1351.6406 | 3163.6406 |
| 65,536 rows FP16 | 1280 | 3092 |
| 262,144 rows FP16 | 5120 | 6932 |

2343+675=3018 MiBの条件では34K FP16に余地があるが、64K FP16は74 MiB不足、
34K FP32は約145.64 MiB不足。decode cache約260 MiBも返る仮定なら3278 MiBで
両者に算術余地ができる。driver/compiler/cache実常駐量の違いを無視した保証ではない。
256Kは依然host tailが必要。表示後から実plan確保までの変化と実cache bytesを
次のreceiptで測る。MTP-only候補と合わせて部品ごとに数値・復元時間・速度を評価する。

## R575: Down phase区間とtiny grouped GEMM

current compact FP16 compute()のmark順はDequant→Gather→GU/SwiGLU→Down。
次expertではcompute()先頭にDequant markerがある。従ってDown bucketへ
次expertのdequant/Gather/GUのGPU serviceが全て入るとは言えない。
Down後から次marker投入前のhost stage wait/queue idle/lookahead copyは入り得る。
R575の文章の「next expert preparation」をこの範囲に限定し、61.535秒を
純Down GPU serviceや全callの682usとして扱わない。

tiny ne<=16はactive callの約55.13%だがrouted rowの約1.386%という既報を使う。
exact shape group>=4の限定batch案はhost投入数削減の候補で、主算術の解決ではない。
4枚のDown weight保持は約12.5 MiBに加えH scratch・metadata・queue lifetimeを要し、
現在2枚を回すbufferのまま後続expertで上書きしてはならない。R496の2.85% synthetic
gainをwholemodelへ移さない。優先順位は残差/不要cacheの実capacity改善と、
decode計測・自然なFP16品質比較の後。必要な部品検証を終えてから実装を決める。
