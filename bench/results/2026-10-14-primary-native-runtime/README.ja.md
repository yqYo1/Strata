# Nikoベースの主実装とブランチの再点検

2026-10-14。唯一の上流は **Niko1221/Strata**。この個人forkをArc B570 10 GiB、
Ryzen 5 5600X、128 GB RAMへ最適化する。上流への貢献とCUDA/HIP推論の移植・測定は
対象外。XeStrataは比較、設計、部品の参考先であり、機能同期や全エンジン採用の対象ではない。
この方針は主作業の [AGENTS.md](../../../AGENTS.md) と
[プロジェクト専用スキル](../../../.claude/skills/strata-upstream-sync/SKILL.md) に保存済み。

## 系統と実装

開始時の主作業は `feature/sycl` の `91c1a18b6563e97ca5d4e3e3fa9a4fcdf7d28133`。
計算経路の比較基準は `4fa478f50f8da396e651e3ae862f8b1849d49b54`、表示版は0.1.38。
fork defaultの `main` は `beddab2fd6b61631d73b39b7484bc9a585ebe1de`、表示版0.1.39。
表示版、祖先関係、機能の採用範囲は別に記録する。最新Nikoの取得・同期を今回の監査とは数えない。

旧 `feat/strata-upstream-sync-skill-20261013` は、ユーザーの指摘どおり **Xe由来の枝**だった。
元tip `97f8b40ca99ec298c24d7ecc7b270e530d43194d` と親 `8b48d17c...` の実treeにも
Xeのbuild・engine系統がある。枝は退役済みで、元commitを主作業の祖先として保存している。
スキルの実ファイルは現在の主作業の `.claude/skills/strata-upstream-sync/` にあり、
`.agents/skills` は `../.claude/skills` への相対リンク。ユーザー共通領域へのinstallはしていない。

履歴保存merge `4526800257fe49c0db02a414f21b38dac2586e54` は比較実験とレポートを復元可能に
しただけで、主作業の計算ソースをXeへ置き換えていない。そのため現在 `main` や比較tipが
祖先に含まれていても、通常のmerge判定を機能統合の証拠にしない。R455の元レポートに
archive前後のproduct tree差分と旧skill枝の由来を残した。

| 部分 | 主実装で確認した経路 | Xe由来候補の状態 |
| --- | --- | --- |
| prefillの処理順・転送 | `prefill.cpp` のchunk/ring/stager、native MMQ・fusedの形式別dispatchを保持 | layer-major・owned bulk転送は未採用 |
| decodeのCPU expert | `expert_source.cpp`、`native_expert.cpp`、AVX2 kernel、poolのGU/Down別処理を保持 | paired IQ3_S GU、追加のroute/service診断は未採用 |
| decodeのGPU・state | session、QSA/indexer、KV streaming、expert cacheを保持 | Xeのdecode群の全置換はしていない |
| SYCL所有権・終了 | 主runtime/compatを使用。前境界でwait失敗後のUSM freeやgraph deleteを防ぐ修正 | 過去のGPUハング原因が解明済みとは扱わない |

「既存経路を保持」と「このPCで最速と確認」は異なる。最近の667–694 token/sは
Xe由来の参考実装の値で、現在の主実装の性能ではない。閉じた全個別値は
[前の比較記録](../2026-10-14-primary-strata-source-audit/reference-measurements.json) に保存済み。
実入力32049・出力24で、厳密32768以上、64K目標、長いdecode、物理256Kの証明には使わない。
実model indexも確認した。GUはIQ2_S20層、IQ3_XXS17層、IQ3_S10層、IQ4_XS1層、
DownはIQ4_NL39層、Q2_0 9層。ファイル名IQ3_Sで全層を一つの形式として扱わない。

## Niko機能の対応状況

Responses/API、設定・UI、tokenize、setup/MCP/profile等は前境界でmainから主へ移植した。
service 268件・7 skip、setup等295件・4 skipの結果と元の失敗を
[機能移植レポート](../2026-10-14-niko-main-service-port/README.ja.md) に保存している。
これらはmock/hostの範囲であり、下記のnative機能を完了させない。

| 残る機能 | 現在の具体的な不足 |
| --- | --- |
| 並列要求 | Nikoの共通非推論機能の対応義務は残る。C++ `BGEN/BGENI/BSTOP/BYIELD`等はmainの選択されたArc engineにもなく、今回初めて失うArc機能ではない |
| 動的VRAM | mainの選択されたSYCL sourceに実装あり。primaryでのsegment/lease/resize、使用中buffer/eventの退役、失敗時の容量保持、実容量報告が不足 |
| CLI/build互換 | batch groups、trim-stage、remote/auto配置、elastic cacheを機能単位で適合・対象判定 |
| 実運用 | nativeモデルとHTTPの連携、実binaryの能力、install/update、継続・RESTORE・context末尾 |

単一GPUに存在しないpeer GPU等の計算は移植しない。共通CLI/API機能はファイルにCUDA名が
あるだけでは落とさず、nativeで意味を満たすか明示的に能力を報告する。
main/defaultの反映は上記が未完了のためpending。mainの機能を失う全置換はしない。
整合・検証を閉じて自身のforkへのPRでdefaultを更新した後、Nikoの更新確認・無改変単体の
設定最適化付き評価・部品別の取り込みをスキルで再開する。Nikoの必要な非推論機能は全件対象、
推論は部品・形式・形状・prefill/decode別に選ぶ。

R458は実際のmain-SYCL CMake resolverを追った。`STRATA_ENABLE_SYCL=ON`ではroot CMakeが
`sycl/`へ入りreturnし、migrated `sycl/src/program/generate.cpp` とsession/cache/sourceを
選ぶ。前のR447/R449がroot CUDA generatorのbatchを列挙したことと、Arc engineがその機能を
持つことは別だった。batch移植義務を免除せず、CUDA/HIP推論一式の移植を前提にしない。
動的VRAMはmainの実SYCL sourceにnative virtual-memory extensionで実装されており、
こちらは真の機能差。ただしextensionの実機対応とその機能の動作は今回測定していない。

## この境界の互換修正と確認範囲

主のRelease SYCL buildにClang22/IntelLLVM v7.1.1、oneMKL2026.1、ggml `3cf03257...` を使い、
CUDA/HIPをOFF、CPUをznver3/native、`-Wall -Wextra -Werror`で確認した。
各失敗attemptは元のfailed状態のまま別receiptに残す。

- `weights.cpp` の19フィールドscanfを固定幅整数の `SCNu64/SCNd64` へ修正。
  LP64でのpointer型不一致を解消。production loaderのhost-only fixtureで4 GiB超の各field、
  符号付きshape、skip metadata、18/19 field拒否をnormal・ASan/UBSanで確認した。
- SYCLでは使わないregistration引数と既存の未使用localを注釈。KV gatherの`else`に
  括弧を追加し、元の内側`if`への結合を維持。allocation・添字・計算順を変更しない。
- SYCLでは使わないGEMMのHIP stateとMMQのbackend contextを注釈。object layoutは保持。
- experimental MMQのGRF128指定をdeprecatedなqueue overloadからkernel functorの
  `get(properties_tag)`へ移す。grid、ESIMD本体、型・量子化・sum順は保持。
- selected compilerでIntel `ha::exp`のdevice APIをcompile probeする。
  APIがない場合の標準 `sycl::exp` は、明示的な `--expert-cache-cpu-order` 経路だけの
  **数値未評価の互換候補**。全expertがQ2_0/Q2_0・2560x640の条件があり、今回の混合形式モデルでは
  このoptionは拒否される。通常のGPU hit計算は変更しない。HAとのbit一致、SwiGLU後の
  scale/code/sum、固定履歴のモデル品質・速度を未確認のまま採用済みとしない。

native dispatchの初回はstandalone Q2 quantizerのscale完全一致に失敗し、後続GEMMは未実行で
止めた。APIエラー・fresh faultなし、所有processは正常なexit1で終了。GPUは最大値に
丸め済み1/127を掛け、現在のCPU AVX2は最大値を127で除算していた。二回の丸めと直接除算は
一致保証がない。GPUを`fdiv_rn(maximum,127.f)`へ修正し、同じscale/code/sum完全一致条件を
維持した再検証が合格。standard-exp/HAの完全一致をこれで認定しない。

R457は極小の有限入力でもreciprocalが無限大となり、float-to-int変換が有効範囲を外れる
問題も特定。実験quantizerは変換前にその場合とNaN/Inf入力を検査し、NaN scaleとzero
code/sumでinvalidを示す。normalの完全一致検査に加え、最小normal float・Inf・NaNでこの
markerを確認する。CPU oracleの範囲外を品質一致と呼ばず、実験routeのmodel側のinvalid
拒否・HA parity・固定履歴の検証は別の未完了gateとして残す。

最終full buildはv13で成功。MMQ通常・GRF128 exact/packed/compact、13形式組合せのfused、
native dispatchとquantizer、22 CPU parityケースと3種類×200jobのGEMM pipelineが合格した。
GEMMの現実的な1280x2560行列はsingle queue、小さい32x640だけをtwo queueで検証し、
過去に止まった大きいtwo-queueケースは実行しない。scale修正後のnative再検証とinvalid-range
追加後のnative再検証を別receiptで保存。MMQ/fused/GEMM binaryは最後の修正前後でhash一致の
ため再実行していない。全所有sessionの終了と各GPU実行後のfresh faultなしを確認。

小さな主runtime GPUテストは、4097整数の64回のcross-queue handoff、bounds/move、
隔離したhost_taskのsticky error、通常graphとnative graphのreplay/lifecycleに合格した。
全owned processの通常終了、同一boot、fresh kernel faultなし、GPU dumpなしを確認。
UR/LevelZeroのprogress/APIログと実providerのloadを取得している。
これはモデル・速度・物理262144入力・RESTOREの合格ではない。

**検証ログ全体がcleanだったとは主張しない。** UR v7.1.1 full lifetime validatorには
optional pool=nullptrを無条件にreference検査する誤警告があり、成功したHost/Device USM allocで
`There are no valid references to handle 0` を出す。R453はAPIと実validator sourceを照合した。
graph-capture問合せの`0x78000023`はQUERY_FALSEで、UR呼び出しは成功。
LD_DEBUGのteardown hook探索は任意hookとしてnull検査される一方、
`zelLoaderTranslateHandle`のlookup表示の発生元は未確定。代表windowを残し、
APIの成功と診断上の未解決事項を別に記録する。

## 証拠と次の反復

R451–R458は読み取り専用Lunaの原文をhash照合して [research-index.json](research-index.json) から
参照する。過去のregistryもリンクし、Mainは返答まで介入せずcompile・source監査を継続した。
調査提案とMainの採否、host/APIテストとモデル測定を分ける。

この境界の最終build・個別検証・branch整理・retention結果は構造化receiptとboundary記録を参照。
成功full API traceや重複build outputは、必要なcommand/source/env/hash/判定と診断windowを
Gitへ保存・照合後に退役する。必要なcompiler/buildは次の主モデル測定のconsumerと予算を
付けて保持し、終了済み枝や別の再開checkpointは増やさない。

次は主モデルのbaselineとボトルネック計測を、無変更のNiko `bench.sh`/standard clientを使う
速度測定と別のdebugプロセスで行う。容量・state・処理順を変更する場合は、262144の実消費、
出力なし、出力余地を残した近傍、継続・RESTOREも確認してから採用する。

## 最終整理の結果

sourceと実行証拠は `e442e85e` にcommit/pushした。個別receiptのdecoded byte/hash、
250種類のsource fingerprint、最終binary hash、Git connectivityを照合後、
参考枝 `feat/layer-major-bulk-upload-20261013` をremote/localとworktreeから退役した。
現在このforkの枝は `feature/sycl` と `main` の2本。元tip `28a7b9ae...` と旧skill tip
`97f8b40...` は主作業から元commit指定で復元可能。参考sourceは必要なときにそのcommitから
読むため、今後の実装用にXe系の枝を保持しない。

[reference-retirement.json](reference-retirement.json) と
[reference-deletion-manifest.jsonl.gz](reference-deletion-manifest.jsonl.gz) に5,157ファイル等の
path/hash/理由を保存。別cloneのtoolchainへ向くsymlinkはunlinkだけで、targetは保持した。
[log-retirement.json](log-retirement.json) と
[log-deletion-manifest.jsonl.gz](log-deletion-manifest.jsonl.gz) は19個の閉じたrunの
raw trace・重複build output・終了済みhost fixture binary・外部の重複研究原文等134ファイルを
対象にした。合計292,056,714 logical bytesを退役。snapshotが保持するextentや実空き容量の
増加はこの数字から推定しない。成功だけでなく初回scale不一致と全compile失敗も保存済み。

receipt内の元B配下pathと研究のoriginal_pathは履歴。以後はこのGitディレクトリのgzipと
研究原文を使う。従来のcontrollerを厳密に再現する場合は、必要な入力record/sourceを
Gitのreceipt/source indexから戻す。新しい主モデルcontrollerは、最終v13 receiptの
source/binary/envを起点にする。
主build `B/main-primary-native-build-root-v1/build` は4 GiB budgetで保持し、次のnative
モデル/HTTP baselineと機能適合がconsumer。baseline完了またはbinary/toolchain更新時に
再検討する。最終の照合は [final-verification.json](final-verification.json)。
`main` はまだ `beddab2f` のまま。native機能差の適合・状態検証、default更新、
その後の新しいNiko更新の確認・評価は未完了であり、この整理をその成功とは報告しない。

GitHub APIでもこのforkのparent/sourceはともに `Niko1221/Strata`、defaultは `main` と
確認した。設定済みremoteはownforkのoriginだけ。残っていた未設定upstream remoteの古い
tracking ref2件は、各tipの復元可能性を確認して削除した。本家の実remoteは変更せず、
この古いsnapshotの整理を新しいNiko更新の確認とは数えない。
