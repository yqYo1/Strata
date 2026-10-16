R861 は **SOURCE_ONLY / UNAPPLIED / UNRUN**。`proposal.patch` と `files/` が提案の全体で、現行 tree・controller・capture は変更していない。ビルド、テスト、GPU、profiler、Git は一切実行していない。速度・正しさの測定結果はない。

`STRATA_SYCL_PROMPT_SOFTMAX_SUBGROUP=1` の完全一致だけを opt-in とした。FP32 の全既存 dispatch（Mode0..3、Direct、Native KV、KMapping0..2）と BF16 のそれぞれに true/false specialization を生成する。既定は false。BF16 の既存 flag 非依存というコメントは、新しい softmax flag の例外を明示するよう更新しただけで ABI は変更しない。

小さな内部 helper で subgroup s が head s と s+8 を扱い、lane c が cell c と c+16 を読む。全 8 subgroup・全 16 lane が 2 回の collective に到達する。phantom head12..15 も省略しない。NaN は元の fmax と同様に最大値入力から除き、BF16 の score-NaN 診断だけ subgroup reduce で集約する。最大値・alpha・sum の store と alpha の exp は lane0 のみ。FP32 と BF16 それぞれの既存 alpha/無限大規則を保った。最初の全欠損 tile は max=-inf, alpha=0, P=0, sum=0 となり、欠損 lane は exp を評価する式に入らない。

同期の根拠は既存の work-group barrier：QK 完了後の barrier → 各 head/score slot を一意に所有する softmax → 独立な V/scale load → 既存 barrier → factors/P 変換 → 既存 barrier → DPAS。softmax の別 subgroup 同士は異なる head のみを書き、同じ head の旧 max は max collective より前に全 lane が読む。次 tile の rows/scores 上書き前にも既存の最終 barrier がある。新しい local array、明示 barrier、追加 kernel launch はない（true/false の kernel image は別）。元ソースの明示 WG barrier は全体で 18 箇所、提案も 18 箇所。集約を memory fence の代わりには使っていない。

BF16 の per-tile scale finite 判定、scales_bad の WG reduction、query_bad、範囲検査、零出力優先、P の割算順、FP32 accum、hi/lo 処理、DPAS は変更しない。false 側 softmax 本文と両 fixture の FP64 oracle・許容誤差は byte-identical（`manifest.json` に区間 hash）。true 側の sum は 32 セル直列和から 16 lane の pair 和 + reduction に変わるので、経路間 bitwise 一致は要求しない。経路内の repeat/batch-single 一致検査は維持した。

既存 case を順序・個数とも保ち、FP32 は元 16 ケースの後に 4 format の one-valid ケース、BF16 は元 112 ケースの後に width17/33/49/65 を追加した。空 tile より前／後に唯一の有効セルを置き、tail の後半 cell16 も検査する。期待値は既存の独立な FP64 数式で計算する。既存 smallp/range/±infinite score・無効 scale・missing page 等のケースは維持した。

出自は凍結した現 Strata source と公式 Intel oneAPI header の API 意味のみ。外部実装・XeStrata・R844/R856 提案は使わない。新規部分は repository LICENSE の MIT。全 touched-base/candidate hash、区間 hash、参考 header hash は `manifest.json` / `provenance.json` にある。compiled SHA は Main 提供値で、fixture/header の commit 所属は worker が Git で再確認したとは主張しない。

Main は既存 GPU 作業終了後、profile の根拠を確認し、`qualification-plan.md` の有限手順で新 build root を使う。通常の CTest/既定 env だけでは true branch を通らない。採否は strict oracle、selected-quality、physical262144、fault、実 device service と全 dependency の測定後に決める。
