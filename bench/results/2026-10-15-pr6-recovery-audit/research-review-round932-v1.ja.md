# R932: R929/R930 decision-support review

R929はR928のQSAに隣接するGDN/HC/PLE経路の有用な補完。取得範囲はR929 lines 7–9, 13–16, 20–26。主比較はSGLang Qwen4-Expで、固定vLLM版のQwen3.5/Next GDNは関連モデルの類推。BF16残差のbit一致はstore境界の対照に有効。FP32 scaleの厳密一致は演算順を保つ比較に限り、一般の採用条件にしない。順序を変える実装は検証済みSGLang/vLLMの保存形式・累積・reduction・rounding契約で評価する。R585は発散監査、GDN geometry候補は旧R592で未認定。

R930は現行vLLMの直接Qwen4-Exp MoE経路と分離測定すべきroute-exchange領域を追加する。取得範囲はlines 7–11, 17, 23–27, 29–38。要求39,262 tokenとbatch 39,261 (=4×8,192+6,493)をline 7が混同。SGLangの「prefill時compute-bound」はQSA indexer overlapのコメントで、shared-expert MoEの反証には使えない。52.433秒のroute領域にはqueue retireが含まれ、除去可能なrouting時間ではない。BF16出力比較は対照とし、採用は本番数値契約で判定。

現時点で新規調査は未適格。R929はMain計測のHC/PLE/GDN内訳または具体的数値差、R930はqueue retire・metadata生成・初回expert GEMM・MoE全体を分けたMain計測が前提。R929報告のhashは一致するが、worker_done拒否後にDispatchがabandonされ、user-owned terminalは残存。成功settlementやprocess終了は主張しない。R930はrelease済みでagent terminal終了。
