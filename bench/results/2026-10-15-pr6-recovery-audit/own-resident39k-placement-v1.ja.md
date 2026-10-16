同じ6354 engine（SHA85643f3926bb9d212945659cb7ddb264e101ca9b2d5fa8f19430e945140ad191）で、official bench.shの39K入力に対するHC residual配置を調べる。FP32やtypedQSAへ変更せず、vLLM BF16 bulk/QSA0、chunk8192、context/resident65536、thinking ON/24output、実39262token入力/39261 batchedを維持する。GPU割当て設定以外の推論コード/モデル/bench/環境は元のclean cohortと同じで、profile/capture/validationなしの新規processを用いる。

初回の数値指定65536は、実prefill行数39261より大きく、`LayerMajorResidualPlan::make`のrequested>n guardで拒否された。Mainが行数をcapのように解釈した設定ミスであり、元のFAIL/incomplete（receipt `own-vllm-resident39k-bf16-r0-v1.json` SHA b3b5485f489ac5625cbad02d3896a356e057a80d2eea6181670c39790bf84423）を保持する。prefill layer candidateは開始されず、runner exit1、全所有process/sessionは閉鎖、新規kernel fault0/dumpなしだった。以前のGPUハングをこの結果と混同しない。

修正後は既存の`auto`を使用する。これは勝手に上限を緩和した実装変更ではなく、既存plannerの予算内選択である。new private root/resultを使い、元の異常設定・閉鎖・新規faultなしを明示的に検査してから試行する。実ログに全39261行GPU/residual_host0が必要であり、部分配置へbackoffした結果を全GPUと呼ばない。

1試行目 `own-vllm-resident39k-bf16-r0-v2.json`（SHA52ad808586e347fa4c79346688a70971817ff06531c58454885bb84590f1f5b5）は **PASS/complete**、正常native/server終了、所有閉鎖、fault0/dumpなし。39261行、GPU804065280B、host0、staging0を確認し、**402.6tok/s**だった。旧固定32768行配置はGPU671088640B/host132976640Bで、clean3試行394.8/394.9/395.2、中央値394.9tok/s。現在の1試行対旧中央値は+1.95%だが、旧/新を交互に測った比較ではなく、一般的な改善率や最終採用は主張しない。新設定の独立3試行を完了してばらつきを確認する。

容量差は132976640Bであり、旧coarse profileで記録したhost residual requested bytesは各方向6249902080B（47境界×132976640）、往復計12499804160Bだった。全GPU配置はこの保存residualのhost往復をなくす。expert重み約50.3GBのuploadや小さなMoE route交換まで消えるという意味ではない。新設定の実PCIeカウンタ/全kernel serviceを測った結果でもない。

現時点で中間residualのhost往復削減だけでは600tok/sに届かない。Attentionの実device時間を測るR864 collector、Mainレビュー済み未適用のR861 subgroup softmaxを優先する。必要に応じて以前のR701/R702 Mainレビューを参照し、native16化で変わったMoE `Dm`とsuperchunkのlive bytesを再計算する。古いFP32-only容量、bitwise equalityを新設計の必須前提として固定しないが、因果state・callback・所有退役を保つ。

新設定の一般品質、physical262144/RESTORE/decode/default採用は未検証。rawログはcohort閉鎖でcompact evidenceを確認後、policyとMainのexclusive measurement lockに従って退役する。current consumerは残り反復と配置比較、owner Main、raw64MiB/job/private1GiB、reviewは2026-10-17T03:00:00Zより前。
