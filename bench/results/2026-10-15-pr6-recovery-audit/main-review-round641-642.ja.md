# PCIe経路の前提修正と長文設定調整

2026-10-15。前の継続turnは独立wave CPU検証sourceの追加、研究R639/R640のレビューと
registry更新をcommit/pushしたprogress。今回も同じexec16956を直接pollしてrunningを確認。
約32Kの3標本は200.5/199.8/199.5 tok/s、最初のcode生成は3346 tokens/9.8 tok/s。
varied-prose初回はthinkingを終えてansweringへ進み、8628 tokensまで観測した。未完了のため
最終decode/MTP値や全suite PASSとは扱わない。新しいGPU/CPU tests、build、cleanupはない。

その後varied-prose初回がeligible=trueで完了した。実completion8631、decode6.2 tok/s、
drafts accepted/offered4551/12243、約37.17%。code初回の2382/2895、約82.28%とは明確に
異なる単一標本になった。各課題3回の分類・全suite/owner/fault終端はまだ未完了。
現在はrun2（逆順でproseを先行）へ進んでいる。課題間の差からMTP設定の因果効果は断言しない。

## R641の訂正と次のPCIe実行の前提

R641 v1はSYCLにCUDAのflag-B spin説明を混ぜていた。Mainは実sourceを読み、FINAL後に
訂正を依頼してv2を受領した。現在の普通SYCLでは`wait_flag_ge`は非使用で、実装もthrowする。
順序は、pre graphがroute/activation payloadを作ってevent完了、hostがplanを作ってpublish、
共有・resident expertのwork graphを開始、DMA投入とCPU expert service、copy streamの
checked drain、次のsegmentでPCIe expertとcombine、次のpre、である。
R641 v2冒頭の「pre() publishes the host plan」はこのhost publish callbackとの区別が必要。

Mainが独立に確認したエラー伝播の穴:

1. `Verifier::fetch_dma`は`cudaMemcpyAsync`と`cudaLaunchHostFunc`の戻り値を読まない。
2. `src/sycl/compat.cpp::attempt`は同期validation/enqueue例外を戻り値とthread-local
   last_error/detailへ変換する。`Runtime::Errors`のsticky async errorとは別である。
3. 後の`cudaStreamSynchronize`はqueueとruntime async errorを確認するが、このTLSを読まない。
   同期投入失敗の後に同期だけ成功し、未転送stagingを後段が使う可能性を排除できない。

これはsourceで確認した取り扱い漏れであり、過去のxe resetやstallの原因と確定した訳ではない。
現在の正式測定はpcie-frac0なので、この発見だけで停止・再起動しない。
非ゼロfractionを初めて試す前に、copy/host-callback投入のfirst error保持、成功済みcopyの
drain、依存するconsumer起動の拒否を修正する。新しいSol実装task `checked-pcie-fetch-v1`
へ隔離proposalを依頼した。root/client/binary変更、build/test/GPUは許可していない。
新observerや同期増設をこの修正へ混ぜず、Mainがsource/実CPU fault injection/実buildを確認する。

SolのFINAL後にMainは`checked-pcie-fetch-v1/source.diff`、実helper、CPU fixture、complete sourceと
manifestを読んだ。提案はSYCL専用のissuer-owned stateを分離し、copy/callback投入前に
completion未知とする。first errorのcode/operation/index/detailを固定bufferへ保持し、
partial enqueue後もchecked copy drainを通してからfalseを返す。drain failureは別に残り、
完了不明のstate resetを拒否する。success pathのcopy順・callback順・同期回数は維持する。
CPU fixtureは同じproduction helperを使いfirst/middle/last copy、callback、drain失敗、
TLS detail上書き、consumer拒否とresetを検証するsourceで、未実行である。
Mainは新geometry条件を実staging16slotsとgroup offset/bytesへ照合し、base8/artifact8の
hash一致を確認した。Mainが変えたのは可変research-indexのhard pinをauthoring時の
context/hash/commit記録へ分離したmetadataだけで、製品提案diffは変更していない。

Verifierのclass layoutが変わるため、実buildではverify.cppだけでなくgenerate.cppも
同じ候補headerで再compileする。Mainのinclude検索ではVerifierを定義/保持する両TUが
該当する。既存b1cc9cb4…へobjectを上書きせず、実link mapで旧verify/generate objectが
混ざらないことを確認する。source review/fixture sourceはCPU実行・GPU・性能の証拠ではない。

fraction0.15は最小positive値ではない。38/256へ丸められ、十分なdistinct pinned missesが
ある場合に選択される候補に過ぎず、実`pcie_experts>0`で経路通過を確認する。
SYCLのpcie modeは0へ固定される。一般engineのauto/direct/kernelコメントを根拠に
別modeが動くと判断しない。安全envとwatchdog480は現在のrecipeを保つ。

## R642: 既存calibrationを長文用途の証拠にしない

Mainも`tools/calibrate.py`、`tools/sycl/serve_bench.py`、`serve/server.py`の該当箇所を読んだ。
calibrate.pyは3つの短い英語入力、出力上限128、temperature0、thinkingを閉じたtoken列で
decodeを測る。さらに`apply(base_args,{})`は指定済みの対象knobをsetup defaultへ戻す。
現在の明示pcie0/min-p0/workers5の長文基準へ、そのwinnerをそのまま当てはめない。

HTTP `strata_tune`が対応するのはpcie_frac/spec_min_p。環境変数を書き換えずrequestで明示できる。
workersは起動時だけ。native `serve_bench.py`のspec-min-p flagもstartup argsの上書きであり、
HTTPのper-request controlではない。正式長文clientは現在tune keysを送らないが、同じbench.sh
を別config/fresh serverで起動すれば、clientを変えずstartup設定を比較する方法もある。

R642の5×3 gridはinteractionを幅広く見る一案であり、唯一/最小の必須手順とは扱わない。
まず修正済みPCIe経路とT2/T3の初回diagnosticを通し、長文のbaseとmin-p変更を比較する。
次に安全なpositive fractionとの2×2 blockでinteractionをscreenできる。非単調な傾向が
出た場合にfraction/thresholdを増やす。最初のscreenと最終3回以上の確認を区別する。
設定比較のbaselineも同じfresh-start/warm-up順、入力、KV/cache/residencyに揃える。
別課題のthroughputだけで最良設定を選ばず、code/prose別のactual長・MTP仕事・速度・品質を残す。
thresholdで提案を絞ると受理率の定義上の値も変わるため、旧weak分類の閾値だけで最適化を
拒否/採用せず、元workloadと実際のdraft量を明示する。新しいthinking-off/短文の速度採用はない。

## 準備済み候補の順序

- 現在のfull OFFを完了させ、同じ元clientのうちにnonceを回復する。入力toolをCPU検証し、
  揃う場合に限定して同じb1cc9cb4…のIQ ONを比較する。古い失敗Direct0/未実行Direct1を
  他の実験へ進む永久条件にはしない。
- 次のCPU検証はwave censusとchecked PCIe fetchを優先する。host phase observer v2の
  base5ファイルは現在もhash一致すると確認したが、新しい機能や計測器を同時に積まない。
- decodeには既存cache容量128/192/256、固定loaded MTP4でのmtp-max-t2/3/4とmin-p、
  修正済みPCIe shareが実測候補。初回は1軸ずつ、必要なinteractionだけを次段で取る。
- prefillには既存chunk4096/8192、bounded wave census、IQ3_XXS subgroup、attention head-split
  の候補がある。各境界で実際の結果から順序を選び、未実行案を積んだだけで改善としない。

物理262144、RESTORE、品質と1000 PP/70 TGの未達は維持する。現在のexplicit context262144
だけでは入力を全消費したことにならず、R642の条件付き説明でこのuser要件を省略しない。
