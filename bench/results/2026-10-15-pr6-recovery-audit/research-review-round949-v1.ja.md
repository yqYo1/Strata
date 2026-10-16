R948はFP32 score staging [16][32] の pitch32→33 比較を扱う判断支援として採択（1,477語、11,272 bytes、報告hash一致）。R568 head-tile、R945/947 query union・N64とは異なる問い。pitch33は+64 bytesで、producer/consumerが同じ写像なら数値演算順を保つ。Intel資料はSLM bankが機種依存で、16 bank×4 bytesは資料執筆時の説明に限る。B570/Xe2の正確なbank mapや実測値はない。joint_matrix stride/alignment制約はmatrix load/store operand用で、scoresはscalar local_accessor。BF16 operandのoffset/strideは整列するが、local_accessor baseの4-byte alignment保証はソースだけでは未確認。

重要補正: ソースヘッダはこのkernelを「No production caller」と記載。実製品経路・性能に結び付けず、production entrypointを追うまでは診断kernelへの適用に限定する。pitch33の `*33` codegen負担も未確認。Mainのpitch33 6プロセス比較は進行中なので、途中値を使わず、R948提案の5サンプル/armを重複実行しない。Mainの11構造・数値ケースPASSは速度の認定ではない。

報告hash一致、transcriptで完了確認。worker_done未確認のままDispatchをabandoned fenceし、Main作成terminalを閉じた。report contentの採択とlifecycle successを分ける。現時点で追加研究は未適格。比較完了後、score layout・compiler lowering・production適用に具体的な未解決問が残る場合のみ本文を取得する。
