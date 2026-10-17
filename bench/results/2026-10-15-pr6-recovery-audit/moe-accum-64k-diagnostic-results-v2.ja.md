# 64K最小入力でのexpert-major window初回検証

2026-10-17。実入力78,453、batched78,452、同じc05c2545由来のbinary。
context/KV resident131,072、chunk8,192、allGPU residual、thinking ON、24生成tokens。
旧window2、新window8の両方が全48層とSSE/生成/終了までPASSした。
全owner終了、新しいxe faultなし、devcoredumpなし。

| 診断 | prompt tok/s | window数 | dequant | jobs |
|---|---:|---:|---:|---:|
| window2 | 576.1 | 240 | 65,543 | 66,696 |
| window8 | 602.4 | 96 | 31,665 | 34,224 |

TRACE/parameter validation/coarse/CPU profile付きの単回結果なので、速度採用や
600を安定して超えるとの判定は行わない。新経路は65,536行＋12,916行で動作し、
専用FP16 mixed 335,544,320 bytesを予算内で確保した。全行がwindow内なので
今回のwindow countersは両経路とも全batched入力を覆う。source_H2D_bytesは
両方150,627,840で、inverse/slotの別uploadやweights本体の転送を含まない。
生成24-token reasoningのSHAも一致したが、一般品質やdecode性能の根拠にはしない。

39K入力の診断ではhost_group全体0.0855秒に対し、粗いQSA区間24.882秒、
そのattention区間18.885秒が残った。この区間はhost submission gap込みの
非排他的なevent間隔であり、GPU busy、GPU演算律速、PCIe律速の割合ではない。
64Kのclean比較が閉じた後、次の優先区間を実データから再評価する。

次は同じbinaryでwindow2/window8を3プロセスずつ、2-8-8-2-2-8順に測る。
TRACEとUR layersは環境から除き、coarse/QSA/CPU/command profilesは無効。
個別値とばらつきを残す。全candidate sampleが600を超えるかも別に記録する。
採用は速度だけでなく、一般品質・状態・physical262144/RESTOREの残る検査を区別する。
1000 prefill / TG70という目標は未達のままで、短入力/decode配置は維持している。

個別結果はmoe-accum-model-profile-window{2,8}-tokens65536-v2.json、
比較はmoe-accum-64k-diagnostic-pair-v2.json。32K最小入力の結果と呼出しミスの
元のfailed receiptも別に保持している。成功raw logsは必要根拠を抽出し、
moe-accum-closed-artifact-retention-v1.jsonの記録に従い211 files / 6,636,083 bytesを
削除した。元のreceipt bytes/statusは不変。audit用source複製はpushed Git commitから
同じhashへ復元できることを確認してから削除した。
