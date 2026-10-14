# 過去ログの再読と旧実装の本家client比較

2026-10-14、Arc B570 10 GiB / Ryzen 5 5600X / RAM 128 GiB。
**XeStrata評価前のこのforkには、現在の主ビルドより速い実装・設定があった。**
現在の主で以前の最適化を保持している、過去の速い値はXe側だけである、という説明は誤り。
Gitの履歴保存と、現在のビルドが選択する実コードの統合を混同していた。

## 元セッションと採用説明

同一セッション `01a0fc19-f45e-7c80-be7c-d1e00e327d4b` の元ファイルから、保存した
公開性能記述351件と重要な採否13件をbyte offsetで再読した。本文SHA256・時刻は
364件全て一致し、監査時点の873,883,929 bytesのprefix SHA256も再確認した。
[session-recheck.json](session-recheck.json)。全文、tool出力、秘密情報、analysisを複製していない。
元の公開記述・行番号・offsetは[前の監査](../2026-10-14-performance-regression-audit/README.ja.md)を参照。

- 10月5日の453→646 t/sは、このforkのlayer-major、32K入力、8K chunk、context65538、
  MTPなしの3組比較。Xe由来ではない。後のKV streaming条件では427.5対429.5と差が小さく、
  その候補は未採用だった。全ての処理順変更を同じ候補として扱わない。
- 10月9日のNiko 0.1.41統合版 `1eb89482` は、実32768入力・64出力、chunk8192、
  compact2、attention-layout1、MTP4/split ONでPP pooled449.698、直後decode pooled16.635。
  RESTORE後の22.165は別の測定phase。これらは現在のroot-nativeで選択されていない。
- その後のcopy-only541.376候補には、fresh32K直後decode約8%低下の独立反復がある。
  候補全体は採用保留。これは449.7基準のCPU/SYCL更新を全て取り消したという意味ではない。
  256K末尾・SAVE/RESTOREは当時完了しており、「256K検証だけが残る」という後の説明も古かった。
- 過去の物理context検証は262140入力＋4出力でcell262143へ到達したもの。
  ユーザー指定の262144入力・0出力を処理した証明とは区別する。現在のrootへ旧合格を移さない。

現在はroot `cmake/sycl_backend.cmake` がrootのprefill/CPU/SYCLを選び、旧版は
`sycl/CMakeLists.txt` / `sycl/src` resolverを使う。旧compact/layout/layer-majorは履歴から
復元可能だが、主の有効sourceへまだ統合されていない。現在の主の基礎は0.1.38系、fork mainは
0.1.39系。非推論機能の一部は移植済みだが、parallel slots、elastic VRAM、最新Nikoの機能差分、
defaultへの反映は未完了。[機能対応表](../../../docs/UPSTREAM_INTEGRATION.md)を維持する。

## 旧1ebを同じstock clientで再測定

旧sourceは595 filesをGit blob/size/SHA256で照合し、source/math patchなしで再ビルドした。
旧binary `4a7e9eee…` は当時のbinary `86972697…` とbyte同一ではない。
quantize/verify/QSA/KV parityの先行合格は[復元記録](../2026-10-14-pre-Xe-replay/README.ja.md)。
28入力・8出力の最初のモデル確認ではparameter validation、Level Zero warning、
native進捗・phase出力を取得。instrumented coldの値は性能評価に使わない。

性能比較は無変更Niko 0.1.43 `bench.sh --long --runs 3` と同じ標準client。
同じ現在のPython serverを介し、旧nativeの実PID・binary SHA・正常終了を確認した。
stockの全項目・各個別結果を保存し、以下はactual32049入力・24出力、reuse0の3 requests。
同じengine CLI flags：context262144、KV int8/resident32768、cache128、workers5、
MTP4/split OFF、chunk4096、no-prefill-borrow、reserve700 MiB。

| ビルド・prefill経路 | prefill t/s、3回 | 中央値 | decode t/s、3回 | 中央値 |
| --- | --- | ---: | --- | ---: |
| 現在の主、MMQ | 118.5 / 118.4 / 118.9 | 118.5 | 4.2 / 2.8 / 3.4 | 3.4 |
| 同じ現在binary、MMQ=0、FP16＋oneMKL | 182.2 / 175.3 / 170.1 | 175.3 | 2.8 / 3.7 / 2.5 | 2.8 |
| 旧1eb、compact2/layout1、FP16＋oneMKL | 341.5 / 343.8 / 347.5 | 343.8 | 7.3 / 4.1 / 2.8 | 4.1 |

旧版PP中央値は現在MMQの2.90倍、現在FP16の1.96倍。これは比較した実装・工具全体の差で、
単一kernelやqueueだけの原因を証明しない。旧版はicpx2026.1.1の整合したoneAPI stack、
現在はLLVM7.1.1/Clang22の別stack。source、layout、queue、GEMM契約、CPU routeも異なる。
同じclient/形状/CLIであり、request payloadのbyte identityは未記録。独立process反復でもない。
旧449.7条件の8192/split ONを再現した結果とは呼ばない。

24-token decodeの範囲は重なり、draft offers/acceptsも異なる。
旧版は30/13、42/9、45/8、現在MMQは30/15、45/10、33/12、現在FP16は全3回45/10。
この短い3件からdecode退行率や復旧を確定しない。prefillの改善でdecode採用を正当化しない。
長い出力の反復・prefill直後・RESTORE後を別に測る必要がある。

旧short/stockはnative/server/client正常0、新しいkernel fault0、driver dumpなし、同じboot。
全595 sourceとbinaryも終了後に一致。[stock-comparison.json](stock-comparison.json)、
[receipts-index.json](receipts-index.json)、[model-diagnostics.json](model-diagnostics.json)。
品質、現在rootの実262144入力、RESTORE、新しい部品の採用は未完了。

## ハードウェアの測定は既に存在する

過去ログの10月10日のハードウェア報告も、祖先commit `7ea2650b` / `b392b0f5` の
元データ・sourceへ照合した。未測定として繰り返さない。次はこの操作別の到達性能と、
現在の実モデルで必要なbytes・形状・呼び出し数・依存関係を結び付ける。

| 操作 | 当時の実測 | 限界 |
| --- | ---: | --- |
| CPU FP32 register FMA、6物理cores | 0.818 TFLOP/s | 量子化integer-dotの上限ではない |
| RAM読み取り、6cores | 36.87 GB/s | 512 MiB arrays、logical bytes、実DRAM counterではない |
| RAM non-temporal copy | 37.58 GB/s | 読み＋書きの合計 |
| host USM RAM→GPU / GPU→RAM、256 MiB | 6.447 / 5.643 GB/s | one-way payload、host-inclusive completion |
| pageable RAM→GPU、256 MiB | 4.605 GB/s | host USMとは経路が違う |
| VRAM内copy、256 MiB | 330.07 GB/s | 読み＋書きの合計 |
| GU / Down GEMM、8192行 | 52.69 / 41.61 TFLOP/s | FP16 inputs、FP32 output、hot |
| GU / Down GEMM、80行、8 weights rotating | 13.16 / 10.08 TFLOP/s | 実kernel形状、合成入力、cold-cacheの証明ではない |
| SSD連続 / random4KiB、16workers | 2.266 / 0.05099 GB/s | 同じfile・ZFS経路、アプリpayload |

RAM/GPU/FMAは3 fresh processes×7 samples/cell。関連14 cellsの全294個別値、FMAの
全個別値、storageの個別process値を保持し、残りはpin済みGit blobから復元できる。
[選択した個別値](historical-hardware/selected-individuals.json)、
[Git出典](historical-hardware/source-index.json)、
[当時の判断](historical-hardware/ATTAINED_CAPACITY_AND_DECISION_BUDGET.md)。
元レポートの相対リンクは当時のGit tree基準で、ここに全文の測定treeは再複製していない。

過去190.241 GBのexpert転送をそのまま仮定すると、6.447 GB/sで約29.5秒。
32Kを1000 t/sで処理する予算は32.768秒。これは別操作の到達帯域による条件付き見積もりで、
現在のcritical pathの実測ではない。旧safe two-queue合成controlでoverlap0だった結果も、
現在の実モデル全経路が重ならない証明にはしない。旧合成GEMMやCPU IQ22/20 fixtureを、
現在の全expert形式・形状へ一律に当てはめない。

## 並行レビューと次の順序

読み取り専用Lunaを反復し、R500–505に加えてR506/507を独立scopeで起動した。
結果FINALまで介入せずMainが元ログ照合・receipt closureを続けた。原文、hash、訂正は
[research-index.json](research-index.json)。R501の既存compact/別queueという誤記、
R503のcontext typoとRAM未測定扱い、R506の存在しないcurrent COMPACT2 toggle等を訂正した。
現在stockの `STRATA_IQ_SINGLE_EXACT` は完全なenv whitelistに存在せず、opt-in parserは
absenceをfalseへする。exact=1のpacked IQ helperを現在stockの有効routeと決め付けない。

Mainも「currentのring8は16へ丸められる」と一度説明したが誤りだった。
旧/現の両方にexplicit8を保持する分岐があり、直ちに撤回済み。ring変更はしていない。
実際に確認できた差はcurrentの `copy=cs` と旧private copy queue等。

別Solが [HC compact案](compact-hc-proposal/report.md) と
[private copy queue案](separate-copy-proposal/report.md) を作成した。以前の
[MoE FP16 compact案](../2026-10-14-pre-Xe-replay/compact-fp16-proposal/report.md)も含め、
いずれも未適用・未build・未実行。既定へ採用していない。
HC案はunfused・特定形状等に制限があり、off/onで同じunfused条件を揃える必要がある。
payload節約値は実workspace削減・速度と区別する。

R504とMainのsource確認で、large-ring threaded skip処理の `used_of` について実C++ raceが
見つかった。skipがissuerのrelease publicationをacquireで待たずにmappingを更新する。
現在のring8測定では `stream_all=false` で対象経路を使わない。過去GPU障害の原因確定でもない。
large-ring/threadedを試す前に、group flush→既消費prefixのpublication→issuer acknowledgment→
mapping更新→新prefix publicationの順序へ修正・検証する。まだ製品sourceへ適用していない。

1. threaded skipのUB対策を先に行い、現在のDMA source lease・queue drain・size_t guardを保つ。
2. currentの同じFP16経路でMoE/HC workspaceの部品復元を別々に検証・測定する。
   currentに存在しない旧環境変数を設定しただけで採用済みとしない。
3. attention layout、copy queue、chunk8192、spec splitを独立した比較として進める。
   stock8192 modeは未実行。4096のshort確認だけで8192 admission/安全性を合格にしない。
4. 数値・用途品質、actual262144入力/0出力・上限付近出力、実RESTOREを新しいsourceで確認する。
5. 成果をfeature/syclの実sourceへ統合し、機能適合・検証後にfork defaultへ反映する。

## 再現と保持

実行済み手順：

```sh
python3 bench/results/2026-10-14-pre-Xe-model-compare/check_reference_model.py short4096
python3 bench/results/2026-10-14-pre-Xe-model-compare/check_reference_model.py stock4096
python3 bench/results/2026-10-14-pre-Xe-model-compare/close_evidence.py
```

controllerはfresh private outputが必要で、既存結果を上書きしない。
planの61055-token expectationは過去の別plan値で、実際のstock結果は32049。
採否は上記actual countersを用い、61K測定と報告しない。
GPU reset、unbind、service変更は行っていない。embed serviceは停止のまま。
全closed receiptはbyte-identical gzipでGitに保存する。
成功全文・poll・重複researchはcommit検証後にexclusive lockでretireし、path/hash/reasonを残す。
旧source/buildと現gate receiptは[名前付きconsumer](retention-plan.json)が使うため保持する。
再開はこのGitレポートから行い、外部checkpointを作らない。
