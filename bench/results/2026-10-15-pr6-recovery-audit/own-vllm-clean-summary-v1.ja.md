# vLLM native16: 反復long prefill測定

統合source6354f0b4、Arc B57010GiB/Ryzen5600X/128GiB。同一binary・official bench.sh、actual39262入力/39261batched、thinking有効/出力24、chunk8192、R_GPU32768、context=resident65536。各armを独立したengine/server processで3回、FP32/BF16/QSA→QSA/BF16/FP32→FP32/BF16/QSAの順に測定した。全9件が正常終了・所有回収、fresh fault0、dumpなし。

| 経路 | 個別tok/s | 中央値tok/s | FP32比 |
| --- | --- | --- | --- |
| FP32 |373.2 /370.1 /372.5|372.5|基準|
| vLLM BF16 HC/MoE |394.8 /394.9 /395.2|394.9|+6.01%|
| BF16+typed QSA |380.3 /384.8 /385.1|384.8|+3.30%|

QSA追加はBF16のみより-2.56%低い。16bit化を一括採用する根拠にはしない。次に同条件のcoarse/QSA/issuing-threadCPU・waitを直列診断し、queue区間を排他的GPU計算時間と誤解せず原因を絞る。数値差は自然32863入力でFP32反復controlを含め別に評価する。実用品質・物理262144消費・RESTORE・decode性能は未採用/未検証。600/1000tok/s目標には未達。旧custom native16の123.8tok/sは入力・resident条件が違うため直接の改善率には使わない。

個別receiptとSHA、コマンド・環境・実際のrequest/placement・正常終了・fault結果はown-vllm-clean-summary-v1.jsonと各元receiptに保存。実行中だったaggregateは全件閉鎖済み、Main wrapper exit0。今回のprofilingが閉じるまでraw cleanupしない。
