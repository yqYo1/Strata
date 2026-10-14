# 速かった自分のforkの復元とMoE経路の切り分け

2026-10-14、Arc B570 10 GiB / Ryzen 5 5600X / RAM 128 GiB。
Niko1221/Strataが唯一の上流。XeStrataのエンジンへの置換ではなく、
Xe評価前のこのforkの実装・設定を比較対象に戻す作業。

## セッション履歴の確認

[前回の履歴照合](../2026-10-14-performance-regression-audit/README.ja.md)に加え、
Mainが元セッションの記録位置から性能報告351件と重要な採否13件を再読し、
公開メッセージの本文SHA256・時刻を全件検証した。[session-recheck.json](session-recheck.json)。
過去の約450 t/sと別条件の約646 t/sは、このforkのXe評価前の実測。
現在の遅い構成で以前の最適化を保持できたという説明は誤りだった。
旧実装はGit履歴から復元可能でも、現在のroot-nativeビルドに統合・選択されていなかった。

## 旧ソースを変更せず再ビルド

旧測定のsource `1eb89482a4afd20277ae0405780ed4f8eb98eb20` を
`.tools/pre-xe-1eb-source-v1/` にGit archiveし、全ファイルのGit blob/size/SHA256を確認した。
主ブランチから別の保存枝を作っていない。このsource snapshotは、旧版の検査と本家clientを
使った比較のための実行入力であり、進捗バックアップではない。

- compiler: `/opt/intel/oneapi/compiler/2026.1/bin/icpx`、実体2026.1.1。
- ggml: clean `3cf03257f219afbe7334045ff7c6a06ac68c627d`。
- 元のSYCL resolver、precise math、subgroup32、per_kernelを保持。source/math patchなし。
- `strata`と既存のquantize/verify/QSA/KV parity targetsをビルド。
- 初回はexport対象に`sycl/tools`がなくconfigure失敗。v2で同じcommitのbuild前提ファイルを
  exportに追加して成功した。失敗を合格へ書き換えていない。
- 新しい旧版binaryは`4a7e9eeeb5ddea917c8dc678f0df0ad4b5a181f8492c875bdcda5cb3026079bc`。
  当時の`86972697...`とは異なり、当時のbinaryとbyte同一とは主張しない。

実際のcompile databaseが証明する重要な訂正は、**旧449.7 t/s版のMoEは
FP16-dequant＋oneMKL経路だった**こと。`moe_mmq.dp.cpp`やfusedの実kernelは選択されず、
以前のR490のMMQ/fusedという分類は誤りだった。旧`compact=2`はこのFP16経路に効いていた。
compiled TU/flagsは元のbuild receipt、link/compiler inventoryと検証結果は
[source-closure.json](source-closure.json)に残す。

旧版の実行検査も直列に行った。quantize Q8_0/Q8_K・verify T1–8は元の参照と一致。
QSA fallback/layout1はbatch32対128、actual129行のtailを含め全8ケースでfinite・bit一致。
40K KV resident対streamed・ring restoreはint8/fp16/q4_0/k8v4各306 batchesで
不一致0・overflow0、全ring restore一致。API/parameter validationとfaultチェック付き。
これは旧版の検査であり、現在版のmodel品質や実262144入力を合格にするものではない。

初回QSAはMainがzero fallback_batchのXMXテストを選び、対応外Q4 Kまで呼んだためexit2。
int8/fp16は通り、対応外poolの拒否で止まった。[qsa-test-refusal.json](qsa-test-refusal.json)。
次のKVは詳細API traceのaggregate64 MiB capで停止し未完了。後処理のjournal取得も同じ
owner log capで拒否されたため、別の新しいbounded ownerで停止後のhealthを確認した。
fault0・dumpなし、APIエラーなし。同一KV binary/sourceをparameter validation＋warningへ
切り替えて完走した。[kv-log-budget.json](kv-log-budget.json)。元の失敗/未完了は変更しない。

## 現在の同じbinaryで一要因比較

現在のroot-native `d3861dd5...` で`STRATA_PREFILL_MMQ=0`を指定すると、
既存のFP16-dequant＋oneMKL fallbackへ切り替わる。製品ソース、CPU/decode、attention、
chunk4096、context262144、KV int8/resident32768、cache128、workers5、MTP4/split OFFを保持。
旧版のcompact/layout/chunk8192/split ONを一度に混ぜていない。

初回28-token/8-outputの診断はUR API/parameter validationとLevel Zeroログを取得し、
native/server正常終了、新しいfault0、driver dumpなし、同じbootを確認した。
instrumented coldの値は速度評価へ使わない。

速度測定には無変更のNiko `bench.sh --long --runs 3`、同じstock clientを使う。
短い項目を含む全スイートの結果を保存し、性能判断にはactual32049入力・24出力、
prefix再利用0の3件を使用する。quiet processでAPI trace/profiler/phase timingなし。
1 process/arm内の3 fresh requestsであり、独立processのばらつきを除いた証明ではない。

| 現在の同じbinary・設定 | 32K prefill t/s、各3回 | 24-token decode t/s、各3回 |
| --- | --- | --- |
| MoE MMQ、従来の比較 | 118.5 / 118.4 / 118.9 | 4.2 / 2.8 / 3.4 |
| FP16＋oneMKL、MMQ=0 | 182.2 / 175.3 / 170.1 | 2.8 / 3.7 / 2.5 |

prefill中央値は118.5→175.3、+47.93%。**decode中央値は3.4→2.8で悪化方向**。
FP16側のdrafts offered/acceptedは全3件45/10、従来側は30/15、45/10、33/12だった。
target側の計算経路が同じでもprompt stateや生成・draft採択が変わることに留意する。
この3件から独立processの統計や性能・品質の採用合格を主張せず、decodeへ一括採用しない。
全stock項目、client/server/nativeは正常終了、fault0、driver dumpなし、boot変更なし。
prefillは改善したが、旧449.7 t/s版の回復はまだ完了していない。

採否、個別値、実環境差と制限は[stock-comparison.json](stock-comparison.json)。
prefill計算の数値経路が変われば生成内容やdraft acceptanceも変わり得るため、
decode速度差をdecodeエンジン改善と同一視しない。品質・実262144入力・RESTOREの
新しい合格をこの測定から主張しない。設定はまだ製品既定へ採用していない。

## 並行調査と省メモリの提案

読み取り専用Luna R494–497を反復し、さらにR498（compact提案の独立安全性レビュー）と
R499（旧/現GEMMの実効契約）をMainの測定・検査準備と並行して実施した。
報告が返るまで途中介入していない。原文、hash、Main訂正、前のregistryは
[research-index.json](research-index.json)。採否はMainの測定・レビューで決める。

別の6.1 Sol実装エージェントが現在root向けの
[compact FP16のpatch案](compact-fp16-proposal/compact-fp16.patch)を作った。
製品への適用・build/test/GPUは行っていない。Xs/GU/Hhをexpertごとに再利用し、
全chunkのDm/route/scatterは維持する。理論payload差は405 MiB@4096、810 MiB@8192。
実際のregion削減や速度ではない。R498では具体的な安全性バグは見つからなかったが、
resident/streamed、actualT tail、relayout、admission、数値・容量を実行検証する必要がある。
Mainは隔離したtemporary treeでpatch適用を確認し、元のcandidateと全byte一致を検証した。
製品ソースは変更していない。[main-review.json](compact-fp16-proposal/main-review.json)。
重複したfull candidateソースはpath/size/hash/reasonを記録してretireし、pin済み元ソースと
patchから復元できる状態にした。実装エージェントの元report/hashは原文を保持する。

R499によれば旧column-major/現row-major GEMMは代数的に同じ契約。
oneMKL内部のdispatch差は未観測で、測定値の差を単独GEMMへ帰属しない。
R497が確認したverify最大T=8は既存契約で、T4–8の設定比較はdecode側の別候補。

## 再現、保持と次の比較

Mainがexclusive `owned-v0141-measurement.lock`でbuild/test/GPU/cleanupを直列管理する。
source/config/provider ABIを確認し、DirectSubmission0、copy offload disabled、
旧版KV prefetch0・own-stage1・bounded watchdog300sを保持する。
GPU reset、unbind、service状態変更は行わない。

```sh
python3 bench/results/2026-10-14-pre-Xe-replay/build_reference.py
python3 bench/results/2026-10-14-pre-Xe-replay/build_reference_v2.py
python3 bench/results/2026-10-14-pre-Xe-replay/check_current_fp16_short.py
python3 bench/results/2026-10-14-pre-Xe-replay/check_current_fp16_stock.py
python3 bench/results/2026-10-14-pre-Xe-replay/check_reference_gpu.py
python3 bench/results/2026-10-14-pre-Xe-replay/check_reference_gpu_v2.py
python3 bench/results/2026-10-14-pre-Xe-replay/check_reference_kv_v3.py
python3 bench/results/2026-10-14-pre-Xe-replay/close_evidence.py
```

各controllerは新しいprivate outputを要求する。既存receiptや結果を上書きして再実行しない。
build v1の失敗は再現手順の履歴であり、修復済み実行にはv2のexport recipeを使う。
コンパクトな元receiptはbyte-identical gzip、個別測定・所有process終了・fault結果はGitで保持。
成功API/build/poll全文は証拠抽出・commit後に閉じたrunだけretireし、path/hash/reasonを記録する。
snapshotは64 MiB、旧buildは4 GiB以内で、現consumerは旧版parityと同一clientの32K比較。
source/component採否が確定したら両方を再評価してretireする。

次は旧選択経路の小規模検査を経て、同じ本家clientで旧版/現在版の形状・設定を揃えた比較。
compact処理を主へ戻す場合もcurrentのrange/lifetime/graph退行対策を保持し、prefill/decodeを別に採否する。
`feature/sycl`への実コード統合とfork default反映は、履歴保存や比較用buildの成功とは区別する。
