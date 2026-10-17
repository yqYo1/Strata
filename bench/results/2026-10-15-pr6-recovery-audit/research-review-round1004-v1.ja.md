R1002/1003はgpt-6-lunaの完了通知、配信ack、release receiptを確認。R1003の本文は保留し、retrieval metadataだけ確認。
R1002は有用なsource feasibility調査だが、BF16 residual bankを4分の1、HC injectionを1,280倍に数えている。
実sourceでは既存MoE arenaは最低1,087,671,808 B。49,152行の2-planeと8,192行job scratchは段階再利用すれば最大878,184,960 B。
これはsource上の容量可能性で、実装証明ではない。現helperは16,384行までで、exact carve、duplicate route処理、queue/lease/callback検証が残る。
78,452行の2-planeだけで1,205,022,720 B。512 MiB reserveを使わず、性能・採用主張もしない。
current combineはroute slot順FP32 FMA。expert順scatterは丸め/加算順が変わり、実モデルのSGLang/vLLM contract比較が必要。
R1003本文はGDN bottleneckを測った後に必要箇所だけ読む。現sourceはchunk再帰とFP32 stateで、今はretrieval不可。
共有indexは未編集。次境界でhash・補正・保留tagを登録し、clean GPU cohort後にMainがcomponent gateを担当する。
