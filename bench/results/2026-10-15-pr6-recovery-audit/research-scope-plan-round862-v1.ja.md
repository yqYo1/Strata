# R862 調査スコープ計画

- **選定:** R863向けの知識収集「GPU FFTのオンチップ／グローバルメモリ間の分解」。単一GPUのバッチ1D/2D FFTで、workgroup内融合、Stockham multi-pass、six-step／transpose型の交差条件と反例を比べます。
- **適格性:** `schedulable_now=true`。公式仕様と原著論文だけで実施でき、ローカルGPU計測や進行中のR860/R861結果は不要です。
- **重複確認:** R854は疎三角解法の依存DAG、R855はFFT畳み込みを含まない畳み込みデータフロー。R845のFFT言及はタイミング例のみです。
- **Mainの扱い:** knowledge collection。本文はMain向けに保留し、レビュー後の短い運用digestと検索メタデータだけ渡します。
- **未実施:** この計画はdispatchしていません。launch前にindexのhashを確認し、R854/R855の別途pin済み入力を添えます。
- 詳細・入力hash: `research-scope-plan-round862-v1.json`。
