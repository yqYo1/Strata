# 主ブランチの再点検とNiko由来の機能移植

2026-10-14。ベースと唯一の上流は **Niko1221/Strata**。この個人用forkをArc B570、
Ryzen 5 5600X、128 GB RAMへ最適化する。上流への貢献やCUDA/HIP推論の実装・実行は対象外。
XeStrataは任意の比較・設計・部品の参考先であり、機能同期やエンジン全置換の対象ではない。

## ブランチと実装の実際

主作業は `feature/sycl`。今回の開始点は `36cc848a0eb68b96a1ed2ec007c5d86cfb3b13a7`。
推論ソースの基準は先の監査時の `4fa478f50f8da396e651e3ae862f8b1849d49b54` と同じで、
CMakeのバージョン表記は0.1.38のまま。個人forkの `main` は
`beddab2fd6b61631d73b39b7484bc9a585ebe1de`（0.1.39）。その実装から必要な機能を適合させている。
履歴保存のmergeによってmainや実験のcommitが祖先になっていても、機能統合済みを意味しない。

残る参考ブランチ `feat/layer-major-bulk-upload-20261013` は
`28a7b9ae7ca5eba28eecd65f6453a7479c04fce8`。Xe由来0.1.40.2系に選択したNiko機能を移した
別の実装であり、主ブランチではない。最初の主実装への寿命管理・診断の移植と同条件比較が
consumerで、その終了または候補棄却時に保持を見直す。新しいブランチは作っていない。
現在のlocal/remoteの3ブランチとworktreeは [branch-state.json](branch-state.json) に固定した。
終了した8つのlocal作業枝と3つのremote作業枝の保存・削除は
[前の整理記録](../2026-10-14-primary-experiment-retirement/README.md) にある。

過去の667–694 token/sの測定はこのXe系参考実装のもので、現在の `feature/sycl` の性能ではない。
入力実数32049・出力24・再利用なしの公式client測定であり、今後の厳密な32768以上の比較や、
64Kの目標達成を証明しない。decodeのcontrol自体も変動したため、bulk方式の採用は保留。
主実装の新しい同条件測定はまだない。prefill改善だけでdecodeの変更を採用しない。

## 主実装に移したものと残る依存

| 機能・部品 | 今回の変更と確認 | 残る確認・移植 |
| --- | --- | --- |
| Responses API、設定API、UI、metrics、停止・再起動処理 | fork mainのserve差分14ファイルを移植。既存primary固有の変更がなかったことを先に確認。サーバー268テスト合格、7 skip | 本物のSYCLモデルでのHTTP/engine連携。batchとVRAMは下記の別依存 |
| 引用されたthinkタグのtokenize | 必要な `Tokenizer.encode(...plain=...)` をmainから移植。失敗した2ケースのうち1ケースを修正 | モデル入力の全体品質・性能は別評価 |
| JSON SchemaのResponses変換 | optional jsonschema付きの隔離したテストtargetで型違反を含むテスト合格 | serving venvは変更していない。jsonschemaなしでは本家の仕様どおりJSON objectのみ確認する |
| setup、既存設定の保持、MCP、draft語彙、profile作成 | Niko系mainの機能を移植し、このforkのネイティブSYCL build・ホストpathへ適合。setupと関連ツール295テスト合格、4 skip | 選んだ実バイナリの能力、実機VRAM検出、画像encoder、モデル形式ごとの対応。sourceのCLI存在だけで古いbinaryを認定しない |
| 通常起動・更新時のbackend選択 | 保存済み設定が全てSYCLの場合と、過去のSYCL設定を採用する場合はCUDA updaterより先にnative adapterへ渡す。引数保持と先行処理をfixtureで確認 | native自動calibration、複数GPUは未対応として明示。現時点のnative updateは既存binaryを保持する |
| reasoning effortの末尾turn checkpoint | `--tail-role-token` を主generatorへ移植。requestのtokenは削らずcheckpoint境界だけ変更 | 実モデルのpark/RESTORE・再利用・長文連携 |
| 並列要求 | Python側の機能とmock testsは移植 | C++側 `BGEN/BGENI/BSTOP/BYIELD`、slotごとのstate・partial prefill・cancel・画像入力の適合が未完了 |
| VRAMの動的変更 | Python側API・設定・UIとmock testsは移植 | SYCL側のsegment/lease/resize、使用中buffer/eventの退役、失敗時の容量保持、実VRAM reporting |
| その他のC++ CLI/build機能 | R447/R449と既存CLI inventoryに対応候補を残した | batch groups、trim-stage、remote cache/auto配置、elastic cache等。CUDA VMM等をそのまま移さず、適用可否とnativeの意味を確認する |
| CPU/SYCLの推論計算 | kernel、prefill、core、specの現在ソースを開始点とhash比較 | 既存の経路を保持。Niko/Xeとの部品・役割別の数値・速度評価が必要 |

setupのネイティブadapterは偽のRAM容量、架空のengine version、Docker内path、Xeのstreaming前提を
導入しない。未知のCLI、未移植のparallel、消える手動engine flag、外部listenのAPI key欠落は
設定の置き換え前に拒否する。画像設定は保持し、encoder欠落を無言で画像無効化に変えない。
最初の実install/updateの成功を今回のoffline testsから主張しない。

[inference-preservation.json](inference-preservation.json) は対象の計算関連ファイルの全hash比較。
変更した `expert_source.hpp` は `reads()` に `override` を追加しただけで、関数本体は同じ。
別に変更した箇所はgeneratorのcheckpoint選択とSYCLの異常時teardown。Xeのlayer-major経路や
量子化kernel群を主実装へ一括導入していない。既存経路が最速かどうかは未測定であり、
「保持した」と「このPCで優れることを確認した」を区別する。

## 異常終了時の所有権修正

読み取り専用R450で見つけた具体的な制御経路を修正した。
`Allocation::release()` はchecked waitが失敗したらUSMをfreeせず、RAII/atexitへ進む前に
異常終了する。graph executableのdeleteはchecked waitが成功したbodyの中に移し、
sticky errorでbodyが実行されない場合やwaitの失敗後にはdeleteしない。

実際のproduction helperを使うCPU fixtureで、正常時のdrain→free一回、wait/freeの失敗時の
終了コード、free・処理再開・destructor・atexitへ進まないことを確認。normalとASan/UBSanで合格。
checkpointもdefault/role/resume/空入力/不正位置/262144要素の末尾をnormalとASan/UBSanで確認。
Clang22のWerror syntax checksはruntime、compat、generatorで合格。

これはGPUハングの既知原因の証明や、全てのsubmit/free競合の解消ではない。GPU・モデルを
実行していない。262144要素のCPU index検証は、物理256KのKV・出力なし・near-limit・RESTOREの
実モデル検証の代わりにならない。古い障害binaryの再実行、reset/unbind、サービス操作もしていない。

## 失敗も含む検証記録

最初のsetupテストは259件・48 failure。46件はLinuxで実行ファイル名をログ名のprefixにも
置換する比較用関数、1件はWindows ZIP fixtureに `strata.exe` の設定が欠けていた。
両者は変更前のmainでも新しいPython processで再現した。期待JSONを再作成せず、ログ名を
実行ファイル名として扱わない比較とWindows fixtureの設定を修正。1件は未移植MCPの
AVX2なしの扱いで、必要な依存をmainから移した。

最終setup/関連ツール295件は全件判定合格・4 skip、individual subtestも保存。
mainのmigration subtree用 `test_setup_sycl.py` はこのnative構成の合否には使わず、native用の
9テストで対象の差を検証する。未対応機能や実機検証をこの置き換えで合格にしない。
268 service testsはoptional schema validator付きで7 skip、なしの途中実行のschema errorも残す。
最初のartifact budget guard、pip欠落、syntax harnessのversion macro欠落、既存警告、
isolation controllerのPath/bytes取扱い失敗も元のfailed/incomplete statusのまま保存する。

[receipts-index.json](receipts-index.json) は元JSONをbyte一致で圧縮保存したpath/hash表。
[outcomes.json](outcomes.json)、[初回の48失敗](initial-setup-failures.json)、controller source、
各環境・command・所有PID・終了/cleanup・全個別判定を保存した。
ソースと証拠のcommitは `51ad49e895c7dce75c0fb74b254255d504f22704`。
中間ソース211件のhashを全て照合し、未commitだった4種類の原文と12 controllerの出典も
`06da66fe052934be4d1f66ca755458edc905bf7a` に保存した。
新規テスト用dependency targetは32 MiB budget、後続の機能検証がconsumer。main機能適合の
完了またはdependency更新時に再検討する。commit後に元データと保存済みgzipのbyte/hash一致を
確認し、12個の終了済みjob rootを直列に削除した。100ファイル、7,509,335 logical bytes。
個別path/hash/理由は [deletion-manifest.json.gz](deletion-manifest.json.gz)、実行結果は
[retention-results.json](retention-results.json) に残した。receipt内の元のB配下pathは履歴であり、
以後はGitのgzipとmanifestから読む。source fixtureと共通owner、compiler、現在consumerがある
dependency targetは保持した。閉じたCPU build/binaryはsource・recipe・hashを残して退役した。

研究用LunaのR449/R450は返答まで介入せず、Mainは別作業を継続した。実装用Solとは役割を分け、
Solはbuild/test/GPUを実行せず、全検証はMainが直列管理した。原文・hashと過去レポートの参照は
[research-index.json](research-index.json) にあり、次の反復にも引き継ぐ。

## 未完了項目と次の境界

この監査は主の実装・参考実装・履歴保存・機能移植を区別する作業を完了した。
**main/defaultへの全機能反映と新しいNiko更新の評価は未完了。** 既存mainのbatch/VRAM等を
失う状態で主ブランチをdefaultに置き換えない。まずnativeへの機能適合とbounded runtime検証、
実モデルの機能・状態確認を閉じ、個人forkへのPRでdefaultを更新する。その後、スキルに従って
Nikoの新しいSHAを確認し、無改変上流＋このPC用に設定を調整した上流を公式bench.shで測る。
推論は部品・prefill/decode別に採否を決める。今回のホスト検証を1000/70 token/s達成とは扱わない。
