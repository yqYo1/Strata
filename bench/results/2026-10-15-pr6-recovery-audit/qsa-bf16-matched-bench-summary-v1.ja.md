# BF16 operand QSA matched component comparison

A matched component speed result, not whole-model tok/s or a measurement of arithmetic utilization. Changes include operand precision/scale placement/products/local payload; no single resource is causally isolated.

| Path | Process medians (ms) | Median (ms) |
| --- | --- | --- |
| fp32 | 632.444841, 632.000898, 632.183553 | 632.183553 |
| typed | 679.690630, 679.607110, 679.714757 | 679.690630 |
| bf16 | 627.164980, 627.525272, 627.346436 | 627.346436 |

各経路3プロセス×10 samples。全プロセスでQ/K/V/scales/IDs/steps・VRAM確保量を一致。QはBF16で表現可能な同一値。各processの2 warmup後、割当・転送・検証・ログの書込みを計測窓に入れない。実行順: fp32-0,typed-0,bf16-0,bf16-1,typed-1,fp32-1,fp32-2,bf16-2,typed-2.

新BF16 / fp32: latency 0.765% reduction, 1.0077x.
新BF16 / typed: latency 7.701% reduction, 1.0834x.

新BF16はquery 0/1/2/lastの24,576要素を独立BF16契約で検証。旧2経路は既存の資格付けを比較の前提とし、旧FP32出力との一致をBF16採用条件にしない。全出力有限・guard・全入力不変・最終repeat一致、各armの独立3process出力fingerprint一致、正常終了・所有終了・新GPU fault無しを確認。

これはQSA単体。モデル全体のPP、品質、物理256K context、RESTORE、実メモリ帯域・occupancy・算術利用率は未認定。デフォルト採用はしていない。新経路はscale/丸め位置・積数・SLM構成も変更しており、個々の速度寄与は分離していない。

If faster, link as explicit opt-in long-prefill candidate, validate model state/quality and run repeated official bench.sh long32K+ before default adoption. Retire successful fullAPI traces after compact facts are committed.
