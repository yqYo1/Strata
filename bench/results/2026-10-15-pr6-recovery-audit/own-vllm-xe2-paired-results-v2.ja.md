# vLLM native16 GEMM の比較結果

Arc B570、同じ Runtime/queue/入力。単位ms。各geometry/formatで独立3プロセス、各route3warmup+3timed。表は各プロセス内中央値の中央値。270 timed submitsを保持し、warmupを含め570回の全出力/oracle/guard/input immutable検査が合格。新規xe fault0、dumpなし、全owner正常終了。

| T,N,K | 入力 | oneMKL→FP32 | FP32→GPU BF16 | 旧own16 | vLLM compact | vLLM wide |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 160,1280,2560 | fp16 | 0.238689 | 0.234702 | 2.264947 | 0.218802 | 0.174759 |
| 160,1280,2560 | bf16 | 0.236625 | 0.231827 | 2.267361 | 0.217129 | 0.177004 |
| 160,2560,640 | fp16 | 0.186962 | 0.188636 | 0.932837 | 0.140114 | 0.122230 |
| 160,2560,640 | bf16 | 0.184358 | 0.191691 | 0.935512 | 0.142338 | 0.123723 |
| 8192,1280,2560 | fp16 | 2.103212 | 2.305383 | 44.078757 | 2.929199 | 1.203226 |
| 8192,1280,2560 | bf16 | 1.760656 | 2.363191 | 44.607676 | 2.908999 | 1.197145 |

全出力の独立dyadic rank2 oracleとguard/入力不変性は計測区間外。wallはsubmitから同じqueueの完了/sticky-error確認まで。個々の9 samples、独立プロセスPID/start、プロセス中央値範囲、CPU submit/wait時間をJSONに残した。inner3回を独立n3とは数えない。

wideはこの6条件のmatched FP32に対して1.34–1.75倍、旧own16に対して7.56–37.26倍。ただし単体operatorの結果であり、whole-prefill tok/s・モデル品質・物理262144/RESTOREの合格ではない。実行形式はstandalone BMG AOT。新しいisolated-AOT engineでは再確認する。

重複したparsed stdout/rows/top-level samplesのみを除き、570組のmeasurement+validationをcases[].proof.samplesに一度ずつ保持した。元private receiptのstatus/owner/command/environmentは変更していない。compact receipt1.82MB、元3.98MB。raw API traceはexact ISA/entry照合の問いが閉じるまで代表を保持する。
