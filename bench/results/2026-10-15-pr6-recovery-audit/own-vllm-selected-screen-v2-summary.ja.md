2026-10-16、実ビルド元6354f0b4、engine SHA256 `85643f3926bb9d212945659cb7ddb264e101ca9b2d5fa8f19430e945140ad191`、Arc B57010GiB/Ryzen5600X/128GiBで、自然入力32863 token（prefill32862）、chunk8192、context/resident65536、GPU residual32768、samequeue、ROPE_TABLE1、MTP4のselected-only数値screenを完了した。receiptは `own-vllm-selected-screen-v2.json`、SHA256 `444649a6d4293ae3bedfa97b1f9e43738048f24bbeadf905c64d68ca7f401f88`。

FP32-a、FP32-b、vLLM BF16、BF16+typedQSAの4本を独立した新規processで順に実行した。すべて全48層、各5chunk、正常exit0、checked ownership closure、owned PID/start不在、新規kernel fault0、driver dumpなし。全体949.585秒にはGPU推論、選択head計算、ログ、FP64比較、検査を含むため、clean throughputとして使用しない。実行handle77034はexit0を確認済み。

選択位置は255+512*i（i=0..63）、各位置の全248320語彙logitsを取得した。`STRATA_PREFILL_NLL*` は全て未設定であり、既知の長大dense NLL callbackは実行していない。全logitsが有限で、変更していない独立FP64 max-shifted logsumexp/NLL oracleで検査した。全位置の個別結果、hash、command/environment、source/build/runtime/model/input pin、fault/closureはreceiptに保持する。

| 経路 | 選択64位置の平均NLL | exp(平均NLL) | 対FP32 top1一致数 | 平均KL(FP32→候補) |
| --- | ---: | ---: | ---: | ---: |
| FP32-a | 5.0572625690 | 157.1597131 | 64/64 | 0 |
| FP32-b | 5.0572625690 | 157.1597131 | 64/64 | 0 |
| vLLM BF16 | 4.7571530550 | 116.4140293 | 56/64 | 0.0681253963 |
| BF16+typedQSA | 4.4088961667 | 82.1787018 | 56/64 | 0.1438818191 |

FP32二試行は選択全logitsがbyte-identicalだった。これは実測した再現性の記述であり、LLMの生成出力一致を採用条件にするものではない。BF16の平均NLL差は-0.3001095140、typedQSAは-0.6483664023で、この選択集合では悪化しなかった。しかし平均絶対NLL差は0.493235/0.815796、最大絶対logit差は9.766536/7.109144であり、分布差を無視できるとは主張しない。1つの入力の相関した64位置であり、全32862位置のNLL、一般品質、task能力の改善/同等性の証明にはならない。データを見た後の受入閾値追加はしていない。

決定：BF16 bulk経路を次の性能最適化の候補として維持する。clean official39K中央値394.9tok/s（FP32 372.5）の既存9試行とは別の証拠である。typedQSAのclean中央値384.8はBF16 bulkより遅く、自動採用しない。いずれもdefault OFFを維持し、quality_accepted/performance_qualified/physical262144/RESTORE/decodeの広い受入フラグはfalseのままにする。

旧exhaustive NLLのFP32-b timeout/xe reset近傍事象と、旧whole-model traceのSIGXFSZは元の失敗結果を保持する。本screenのPASSで旧問題を解決済みに書き換えない。今後のボトルネック調査は現行バイナリを公式unitraceで測るR860を優先し、初回tiny component→official39KをMainが直列実行する。R856 event observerはレビュー済み未適用、R861 subgroup softmaxはsource-only作成中の仮説であり、実測改善はまだない。

raw selected4本254279712 bytesはprivate root `main-vllm-selected-screen-v2` にある。現在のconsumerは比較結果の再検証と後続の同一route数値検証、owner Main、tensor budget256MiB、private budget1GiB、次回retention reviewは2026-10-17T03:00:00Z。実行中のGPU/build/testとcleanupを重ねず、consumer更新後にcompact evidence/hash/削除manifestを保存して退役する。
