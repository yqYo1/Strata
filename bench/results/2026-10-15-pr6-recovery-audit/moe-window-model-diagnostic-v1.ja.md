# MoEの二分割まとめ実行: 実モデル診断

2026-10-17、Draft PR32の同じ完全build（strata SHA-256
243c34d671300605b50e1069c9609ad068ccfee1c5aab3ee745ffa76c1988360）で
window0、window2をそれぞれ独立processとして実行した。公式bench.shのprefill-only、
同じ39,262 input tokens、thinking有効、prefix reuse 0、context65,536、chunk8192、
BF16 native16/vLLM、optimized QSA、GPU residual32,768行。生成24tokenは完了確認用。
両方とも正常終了、所有process終了、fresh kernel fault/dumpなし。

| 診断値 | window0 | window2 |
| --- | ---: | ---: |
| engine PP tok/s | 458.2 | 490.9 |
| engine prompt ms | 85695.5 | 79977.2 |
| wrapper ms | 84586.8 | 78874.0 |
| MoEの非排他queue区間 ms | 20179.2 | 16783.9 |
| QSAの非排他queue区間 ms | 24736.4 | 24719.1 |
| 全GEMM投入回数 | 123156 | 85700 |
| main threadの明示wait CPU秒 | 57.9848 | 52.9106 |
| wait外CPU秒 | 20.3026 | 20.7857 |

window2はGPU上の8K行2個をまとめる96窓を実行した。窓部分のexpert展開pair27,679回、
GEMM job28,196個、source index H2D要求62,914,560B。これは窓部分のcounterであり、
tailを含む全expert/GEMMやPCIe上の全転送量を意味しない。

wrapper時間は6.75%短縮。weight upload要求は両方とも同じであり、
QSA区間もほぼ同じ。今回の改善はweight H2Dを減らした結果ではない。
queue区間はhost submission gapを含み、GPU利用率や排他的kernel実行時間ではない。
window pathのrouter readbackは旧route_exchange計測scopeの外にあるため、
そのCPU/wait counter減少をそのまま実際のreadback時間短縮と解釈しない。

TRACE/validation/coarse/QSA/CPU profiling有効の各1回だけなので速度採用は保留。
同じbinary/inputで、診断を外した独立6processを0,2,2,0,0,2の順に測る。
実モデル品質、state比較、physical262144 context、RESTORE、cancel/reuseは未資格付け。
600/1000 tok/sのprefill目標、70 tok/sのdecode目標も未達。

元receiptはmoe-window-model-profile-window0-v1.jsonとwindow2-v1.json、
解析と全layer値はmoe-window-model-profile-comparison-v1.json。
再現はrun_moe_window_model_profile_v1.py（既存receiptには再実行不可）、
解析はanalyze_moe_window_model_profiles_v1.py、次のclean測定は
run_moe_window_clean_cohort_v1.py。

前goal turnは実build、11GPU case、model0を実行した進捗turn。
今回もmodel2と同条件比較が完了し、次の測定判断を変更する実証を得た。
R976–978はnative Luna maxのR979レビューを経てindexに統合し、
Mainは元report本文を読んでいない。新researchはclean cohort後の問いに合わせる。
