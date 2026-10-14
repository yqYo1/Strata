# セッション履歴と過去の速い実装の再照合

2026-10-14、Arc B570 10 GiB / Ryzen 5 5600X / RAM 128 GiB。
ユーザーの指摘を受け、この同一セッションのログ、当時の測定の元データ、Gitの実ソースを
照合した。**XeStrata評価前に、この個人forkで現在より速い長文測定が存在する。**
「Xe側の速度と主ブランチを混同しているだけ」「主の最適化を保持した」とした直近の説明は、
10月5日以降のNiko由来の別SYCL系統を見落としていた。現在の118.5 t/sを、既存の最良構成を
維持した性能として扱えない。

## 過去ログと元測定

最初のXeStrata評価依頼は2026-10-10 **23:33:07 JST**。
その前の実測をXe由来と分類しない。対象session ID、ログの読み取り範囲のSHA256、
行番号、byte offset、公開メッセージのSHA256を [session-performance-claims.json](session-performance-claims.json)
に保存した。公開された性能記述351件を抽出し、重要な採否・実装説明13件は
[session-decisions.json](session-decisions.json) に原文と位置を残した。
ツール出力、秘密情報、推論内容、約874 MBのセッション全体は複製していない。

| 日付・系統 | 実入力・測定条件 | prefill t/s | decode t/s | 判断・制限 |
| --- | --- | ---: | ---: | --- |
| 10月4日、root-native | 4007入力、context8192、chunk4096 | 約184.6 | 別の短文条件 | 直近の再監査が重視した古い記録。32Kの基準にできない |
| 10月5日、このforkのlayer-major | 32768入力、context65538、chunk8192、int8 KV全量と残差32768行をVRAM保持。quiet CLI、1出力、MTPなし、warm PLE、3組 | 元453.294 → 645.848 | 未評価 | Xe由来ではない。全6 first-head/ID一致。物理256K全入力・MTP込みの速度ではない |
| 10月5日、cache解放比較 | 65536入力、context65538、chunk4096、残差VRAM保持量を調整 | 保持519.73 / 半解放532.53 / 全解放539.76 | 未評価 | 各1回、転送診断あり。実装・容量の参考で、quiet反復の採用結果ではない |
| 10月9日、Niko 0.1.41統合 `1eb89482` | fresh32768入力、64出力、context262144、int8 KV/resident32768、chunk8192、cache128/325 MiB、MTP4/split ON。2プロセス・後続6回 | 中央値449.674、pooled449.698 | 中央値16.182、pooled16.635 | 当時の検証済み基準。初回prefillはpooled408.648。prefix再利用なし |
| 10月9日、同基準のRESTORE反復 | 32831-token stateをRESTORE、64出力。6独立プロセス×12後続decode | 対象外 | pooled22.165 | prime/RESTORE/warm decode時間を除く継続測定。直後decodeと混ぜない |
| 10月9日、copy-only候補 | 上記32Kと同等形状。候補ON/OFF/基準を分離 | ON541.376 | 直後は基準16.826 → ON15.464 | 後述の独立反復で約8%decode退行。全体採用は保留 |
| 10月14日、現在のroot-native | 無変更Niko bench.sh、actual32049入力・24出力、context262144、KV resident32768、chunk4096、cache128/268.60 MiB、MTP4/split OFF | 118.5 / 118.4 / 118.9 | 4.2 / 2.8 / 3.4 | 全3回正常終了、fresh fault0。最適化された最良設定ではない |

現在と過去にはsource、compiler/runtime、chunk、cache実バイト、spec split、入力、生成長、
測定方式の差がある。したがって表から単一kernelの原因や厳密な退行率を計算しない。
一方、その差を理由に過去の速い測定を捨てない。元の設定と呼び出し経路を復元して比較する。
今後の正式速度比較には、引き続き無変更の本家bench.sh/標準client、32K以上、各arm3回以上を使う。
過去のGEN/CLI測定は元の意味で保持する。

## 実ソースにあった差

`1eb89482a4afd20277ae0405780ed4f8eb98eb20` は、このforkのNiko 0.1.41統合版。
binary SHA256は `86972697ecb1903750d202f8be29a7a1da351eb3415678c3e3b6af790804d1c8`。
root CMakeは `sycl/CMakeLists.txt` に入りreturnし、resolverは `sycl/src` の移植済みソースを
rootより優先する。旧build receiptはcleanなこのcommitとggml `3cf03257`、oneAPI2026.1での
configure/build、実測binaryを対応付ける。全TUのcompile commandやlink inventoryは未保存。
旧control DD5の観測commit `452044ab` はsource同一の再buildを証明せず、その注意を保持する。

現在の基礎 `4fa478f5` はcommit日時が10月9日でも、直前のproduct境界は10月4日の `22e61b95`。
現在は `cmake/sycl_backend.cmake` がroot `src/kernels/sycl` / `src/prefill/sycl` をコンパイルする。
旧実測の `compact=2` のworkspace再利用、`attention-layout=1` の選択経路はここにない。
旧QSA XMXと現在のmatrix attentionは異なる実装であり、全てのXMXを失ったとは言わない。
verifier/CPU expert/prefillの選択ソースとbuild契約も異なる。

これは「現在の主ブランチを丸ごとXeへ置換した」ことの証拠ではない。
**過去の速いNiko由来の作業系統を主の実コードに統合せず、古い主系統で再開・測定していた**
ことが確認できる。`45268002` 等の履歴保存mergeは実コードの統合ではない。
旧source/report commitsは現在の `feature/sycl` の祖先で、commit指定で復元可能であることも
実際の `merge-base --is-ancestor` で確認した。新しい保存用ブランチは作っていない。

## 採否と検証範囲の訂正

- copy-only541.376の候補は、262144物理cell末尾とSAVE/RESTOREの独自検証を10月9日に完了していた。
  10月10日に「256K検証待ち」とした説明は古い。残っていた採用阻害要因はdecode退行。
  fresh32K直後は6独立プロセス/条件で−7.985%、paired95%区間[−12.806%,−2.897%]。
  継続decodeは72回/条件でも方向未確定。prefillの改善でdecode採用を正当化しない。
- 旧0.1.41のfull-context記録は262140入力＋4出力でcell262143に到達し、全262144保存cellと
  実RESTOREを比較した。262144入力＋1出力は拒否した。**262144入力・0出力を実際に処理した
  証拠とは区別する。** 復元・新しいstate/scheduling候補には、このユーザー指定のfull-input
  境界も含めて検証が必要。現在のroot系統へ旧容量合格を移さない。
- 10月9日のstager配置比較で報告したRESTORE21.686も旧基準の別測定。候補は21.644で不採用。
  当時のREADMEは保持したが、そこから参照される全phase summaryを今回の保存Git treeでは
  確認できていない。今回保持したdecode反復summaryの全個別値22.165を確認可能な基準にする。
- R489や直近の「主の最適化を保持」の説明は、10月4日系統のbyte比較としてのみ有効。
  10月5日以降の最良構成を保持した証明ではない。R490/R491とMainの実測照合で訂正する。

## 次の実装・比較

現在の遅い構成を新しい基準として局所最適化する前に、旧Niko系統の実装・設定を比較対象へ戻す。
新しいモデル/profiler実行はこの監査中に開始していない。

1. 旧0.1.41とcurrentの有効ソース/dispatchを部品ごとに比較し、compact workspace、attention
   layout、expert GEMM、CPU量子化、spec split、layer-major/残差配置、cacheの違いを記録する。
2. 過去のGPU障害対策、DMA実eventによるsource lease、queue drain、graph retire、typed KV、
   indexer spare、spec commitの契約を保つ。compiler/UR/SYCL ABIを混ぜない。
   旧Linux watchdogは既にstop-token付きjthreadを観測対象より後に構築しており、joinがある。
   Windows detachだけをLinuxでの再評価阻害要因にしない。cold JITの60秒既定と現在の大range
   compiler契約は、実際の旧ソース・選択工具に合わせて対処してから実行する。
3. 検証済み旧系統の隔離比較、または主への部品復元を行い、同一client・入力・設定・工具で
   再測定する。prefillとdecodeの採否は別。過去の安全対策を外して数字だけ再現しない。
4. 実体を `feature/sycl` に統合し、区切りでfork defaultへ反映する。履歴保存で完了としない。

読み取り専用LunaのR490–493を、結果が返るまで途中介入せずMain監査と並行して実施した。
原文・SHA・Main訂正は [research-index.json](research-index.json)。研究提案は未採用。
[comparison.json](comparison.json) は旧16 reads、旧3 pairs、独立decode反復、現在の全個別値を
まとめる。元のGit blob/commit/SHAは [historical-source-index.json](historical-source-index.json)。
小さい結果・コマンド・数値だけを再掲し、成功API trace、raw tensors、profiler timelineや
重複pollログは追加していない。
