# 主ブランチの再監査と実モデル診断

2026-10-14。唯一の上流は **Niko1221/Strata**。`yqYo1/Strata` は
Arc B570 10 GiB、Ryzen 5 5600X、128 GB RAMを使う個人用fork。
現時点で上流への貢献は予定しない。CUDA/HIP推論の移植・測定は対象外。
XeStrataは独立した比較・アーキテクチャ・部品の参考先であり、全エンジンや機能を同期しない。
この方針はAGENTSとプロジェクト専用のstrata-upstream-syncスキルに既に保存されている。

## Gitと実ソース

開始点は `feature/sycl` の `5fdf9a1517526085c9a2fef2eccfb24f87b4f241`。
GitHub APIでもforkのparent/sourceはともにNiko1221/Strata、defaultはmain。
local/remoteの枝はmainとfeature/syclの2本、登録worktreeも2つ。
ghq rootのmainは `beddab2fd6b61631d73b39b7484bc9a585ebe1de` のcleanなmirror。
新しい枝やPR、上流への投稿は作っていない。

| 対象 | 実際の内容・状態 |
| --- | --- |
| 主推論の基礎 `4fa478f5` | Niko 0.1.38 `99f3dbd0b21d1401b3769e0c0d963913607f380b` を基礎にした、このforkのSYCL・CPU最適化系統 |
| primary HEAD | 表示0.1.38。主のchunk/ring/stager、native MMQ/fused、CPU GU/Down、cache・sessionを維持し、Niko機能と互換・寿命・数値修正を部分移植 |
| fork main `beddab2f` | 0.1.39系。SYCLではroot CMakeがsycl/へ入りreturnし、移植済みsycl/src/program/generate.cppとcore/prefillを選ぶ。primaryとは別の実ソース |
| 旧skill `97f8b40c` | ユーザーの指摘どおりXe由来。退役済み。スキル実体はprimaryの.claude/skills、.agents/skillsは相対リンク。ユーザー全体へのinstall無し |
| 旧比較tip `28a7b9ae` | Xe由来の実験。枝/worktreeは退役済み。元commitから復元可能。主実装・上流評価の代用にしない |

[primary-audit.json](primary-audit.json) は元tree、代表sourceのblob、現在のworking
source SHA、実remote、枝/worktree、archive mergeのfirst-parentとの差分を保存する。
`8927eb69` は45親のXe側archive、`45268002` は主側archive。いずれもfirst-parentから
product pathsの差分は空。前者がprimaryを祖先に含み、後者がXeを祖先に含むのは履歴保存であり、
計算ソースの統合ではない。通常のmerge済み判定や版番号で機能採用を認定しない。

| 部品・機能 | 主実装の監査結果と採否 |
| --- | --- |
| prefill処理順・転送 | 主のchunk/ring/issuer/stagerと形式別MMQ/fused dispatchを保持。Xe layer-major＋owned bulk span uploadは未採用 |
| decode CPU | 主のAVX2 GU/Downとnative poolを保持。Xe paired IQ3_S GUのactivation-load共有は未採用。NT1の合成差をモデル利益としない |
| decode GPU/state | 主のsession/cache/QSA/KV streaming/verifyを保持。Xeのdecode全群へ置換していない |
| Responses/API/UI、setup等 | 前境界でNiko系mainから適合。service 268件/7 skip、setup等295件/4 skip。mock/hostの合格と実運用は分ける |
| 動的VRAM | mainの実Arc SYCL sourceにvirtual-memory segment/resize機能がある。primaryでlease/安全な退役・失敗時capacity/reportingの適合は未完了。実機対応も未確認 |
| 並列・CLI | Niko共通の非推論機能は対応義務が残る。mainのroot CUDA batchを既存Arc機能と誤認せず、実能力・エラー契約を含む機能単位で適合する |

Nikoの非推論機能は全件対象。推論は形式、形状、prefill/decode、GU/Down等の部品ごとに
正しさと速度で選ぶ。Xe側の機能不足を必須同期項目にせず、参考forkの配布・build方針にも
合わせない。compiler/ggmlを既存Xe調査worktreeの保持領域から使うことと、Xeの製品ソースを
採用することも別である。

最近の約667–694 token/sはXe由来の比較実装の値で、現在primaryの速度ではない。
全個別値は前のsource-audit/reference-measurements.jsonで保持している。
main/defaultの反映、必要なnative機能・状態の整合、fresh Niko更新の確認・無改変単体の
設定最適化付き測定・部品別取り込みは未完了。履歴整理をその完了と呼ばない。

## 実モデルの初回診断

使用modelはQwen3.8-Flash-Next-GSQ-RCO-IQ3_Sの2 shardと既存native pack、MTP。
modelのGUはIQ2_S20層/IQ3_XXS17層/IQ3_S10層/IQ4_XS1層、DownはIQ4_NL39層/Q2_0 9層。
max-context262144、int8 KV/resident32768、cache128/per-layer、workers5、PCIe share0、
spec4/prefill4096、prefill borrow OFF/reserve700、reuse cache OFF。
現在のCLIとmetadataを実sourceで確認し、別系統のunsupported flagsを混ぜない。
ユーザーの保存済み設定・サービス・GPU bindingを変更していない。

| attempt | 元の結果 |
| --- | --- |
| v1 | 主CLIに無い--no-kv-growでengine exit2。Mainの設定取り違え。参照実装の他3flagもsource preflightで除外 |
| v2 | --serveに必要なspec/MTP/prefill条件を満たさずengine exit2。model読み込み後に拒否。終了時のUR reference診断を成功ログ扱いしない |
| v3 | MTP metadataをindex.txtと誤認したpreflightで停止。GPU/owner起動無し。実ファイルはdense.txt |
| v4 short | HTTP応答成功、28入力/reuse0/8出力。heavy validation・timing・cold JITの診断であり速度比較ではない |
| v4 long | 実61055入力、engine prefill61054、chunk0でSYCL range exception→SIGABRT。engine -6、wrapper134。新GPU fault0、dump無し、同一boot、全所有process終了 |

v4 shortはprompt_ms26959.0、decode_ms7887.3、TTFT31.059 s、total34.851 s。
その1.0/1.0 t/sをチューニングのclean baselineにしない。長文は標準clientのchat/long_promptを
使う補助診断で、無変更bench.shの速度suiteではない。実61055を厳密65536と呼ばない。
正式な比較は無変更のNiko bench.sh/標準clientで各3回、実入力数・reuse・phaseを保存する。
出力24 tokenの短いdecodeと長いdecodeも区別する。

## 大きな実行範囲の契約

例外は `Provided range and/or offset does not fit in int`。
Intel LLVMのhost headerと公式UsersManualを照合すると、defaultは各kernelのglobal rangeが
INT_MAX以内というコンパイル契約。多次元の線形積も検査する。
R463で調べたQSA/Attention/GDN/PLE等の最初のchunkの式はINT_MAX未満で、正確な発生元は
未確定。oneMKL内部のlaunchはwrapperのshape検査だけでは分からない。

cmake/sycl_backend.cmakeで `-fsycl-id-queries-range=size_t` のcompiler対応を検査し、
SYCL runtimeから全consumerのcompile/device linkへPUBLICに伝える。既存のtensor容量・
explicit int変換の検査は残す。input/context/chunkを縮めて回避しない。
full native engine、runtime/compat/MMQ/GEMMとhost契約fixtureを再buildした。
v1はhost専用のSDK helperをdevice compilationでも参照したfixture compile失敗。
原sourceをrange-contract-failed-v1.cppに保存し、SYCLの通常headerをincludeしてhost側だけで
helperを呼ぶv2が合格。v1 failed statusは書き換えない。

host fixtureはINT_MAX+1、2D線形積2^31、2D nd_rangeの8 Gi-itemをhost guardが許可することと
size_t contractの伝播を確認するだけで、巨大GPU launch・実モデル・256K全消費の証明ではない。
最後のnative engine SHAは3fa2375eab38dbd3be187e2bed410099b03103a73429d29e67d5b11e975943ee。
例外観測用のLinux preloadはhost-only selftestで元SYCL exceptionのcatchとstack出力を確認した。
診断engineだけに適用し、正式速度には使わない。型・引数・destructor・再throwを変更しない。

小GPU確認と同じ実長文の再診断はこの境界時点では未実行。range flagは対策候補であり、
モデル問題が解消したとはまだ認定しない。物理262144全入力/出力なし、近傍の出力枠、
最終cell、継続・RESTOREも未完了。

## 証拠・反復・保持

[receipts-index.json](receipts-index.json) は全closed attemptの原record/result/exitを
decoded hashで保存する。[model-diagnostic-windows.json](model-diagnostic-windows.json) は
別のCLI/startup/range機構と必要なprogressを抽出し、full原logのhash/行位置も残す。
成功full API trace・重複build outputは検証後にretention policyで退役する。

R459–R463のLuna原文は [research-index.json](research-index.json) から参照する。
R461系統/R462部品を並行依頼し、返答まで介入せずMainはsource/ref監査・buildを継続した。
続いてR463でrange契約を再調査した。調査提案、Mainの実装、実機測定の合格を分ける。
旧registryを共有して調査を反復する。

保持buildはB/main-primary-native-build-root-v1/build、4 GiB budget。
consumerは次の主モデル診断・機能適合、binary更新/診断終了時に見直す。
例外observerは16 MiB budget、次のbounded診断がconsumer。新しいGPU reset/unbind、
サービス停止、既存server置換は行っていない。

## 追加の履歴・機能照合

R464は旧primaryの `dc2ca4eee42c5870e11a1339c0fccc9fd9e856f6` を直接調べた。
同じ5600X、実GGUFのIQ3_Sを含む3形式、NT1のQ8_K活性、E=8/64個のexpertと
unroll1/2/4でgate/upのload共有を試し、全180sampleが数値条件を満たしたが、全中央値は退行。
この結果はprimaryのGitに既にある。Zの狭い合成NT1の平均差だけで候補を再提出しない。
current compiler/code/実routeの条件変更を示す場合に限り再検討する。
R464の「NT8もprimaryで測った」は、expert数E=8との混同であり採用しない。
引用された3つのE64数値にも小さな相違があるため、原run.jsonと
prior-cpu-decision.jsonを正とする。旧recordはcompilerのversionを固定しておらず、
現在toolchainと同一だと主張しない。原研究レポートは訂正履歴とともにそのまま保存する。

R465はNiko 0.1.39、fork mainの実SYCL overlay、primaryを機能単位で照合した。
このPCで適用可能なnative不足は、単一GPUの並列要求・admission/cancel/response契約と
動的VRAM reserve/resize。serverが保存済みparallel>=2を--batchとして渡すと、現在の
primaryはunknown flagで起動拒否する。vram_elasticも同様。設定を無言で消してはいないが、
機能実装の代用になるものではない。静的vram_reserve_mibは別の既存機能として保持する。
Nikoの非推論機能を保持する指示は既にあるので、この2項目の適合を任意の確認事項にしない。

batch-groups、trim-stage、remote helper cache、peer/tier、複数deviceのauto-placementは
複数GPU・layer splitの機能。ここを単一Arcでのdefault反映のblockerにせず、対象外理由と
保存設定の明示的拒否を記録する。mainのSYCL VMMはNikoのAPIを独自に適合した実装で、
device aspectによる拒否もあり、このPCで稼働済みとは扱わない。

R464/R465も過去registryを渡して反復し、Mainの実モデル診断と並行で進めた。
返答まで途中介入せず、調査だけでsourceやGPUを変更していない。

## 後続の実機結果

size_t設定でruntime、MMQ、GEMM、native graphの4つのdebug実行が合格した。
UR successは各832/9494/14284/639件、各プロセスexit0、fresh GPU fault0、dump無し。
GPU receiptとgpu-diagnostic-summary.jsonにAPI error・既知null-pool警告・QUERY_FALSEを分ける。
full validationのログ全体がcleanだったとは主張しない。

同じ61055入力のv5はshort HTTP成功後、prefillの53248位置のchunk開始まで進んだ。
元のrange exceptionとobserverのSYCL throwは再発していない。ただし完走前に1200秒の
owned wall deadlineで終了。runnerは-15、serverにも最後はSIGKILL、nativeへSIGTERMを
送った記録はあるが、real-engine exit receiptは無い。observation_complete=falseのまま保存し、
正常終了・長文合格・速度合格へ書き換えない。session_empty=true/survivorsなし、各owned identityは
消失し、同一boot/fresh fault0/dump無しを確認した。longのDONE/sampleは得ていない。

終了前にowned native stdinへSTOPを送ったcontrol-action.jsonも保存する。送信は
22:52:15.300 UTC、owner開始22:32:46.777 UTCで、wall上限まで約31.477秒しかなかった。
sourceはstop_reqをstdin threadで設定し、sp.should_stopがchunk境界で確認する。
従って停止が期限に間に合わなかっただけで、STOP未伝達やengineのbugを認定しない。
Mainの停止猶予が不足していた。最後のchunk開始markerを、61055全入力の消費や出力成功と
数えない。観測された範囲では対策後のrange guard failureは消えたが、正確な元のsubmitと
完走の確認はまだ残る。

次の長文診断は、既に小fixtureで行ったfull lifetime検査と分け、LevelZero/UR parameter
validation・warning/progressを保持してログ負荷を減らす。事前wall budgetとchunkの停止猶予を
十分に確保する。さらに別のquiet processで無変更Niko bench.sh/標準clientの速度測定と
設定調整を行う。今回の約0.9 t/s short診断や未完了longを、性能退行の採否に使わない。

v5の代表deadline/STOP raw証拠は次のnormal-exit/停止猶予の比較がconsumer。
責任者Main、text budget64 MiB、現在の少量ログを保持し、比較完了/recipe置換時に再検討する。
他のclosed成功trace・重複build outputと終了した研究原文は、commit照合後に退役する。

## 境界の整理結果

build・host契約・系譜の証拠を94bc3148、追加のCPU棄却照合・機能対応・実GPU/モデル結果を
7ed2ecd9でcommit/pushした。原receiptのdecoded byte/hashとGit内容、全known owned PIDの
start ticks消失を照合し、retirement.json/deletion-manifest.jsonl.gzに92ファイルの
path/hash/理由を保存して24,781,120 logical bytesを退役した。実空き容量の増加は推定しない。
original failed/incompleteとobservation_complete=falseは変更していない。

v5のengine.stderrだけは40,511 bytes、保持上限256 KiB、上記のnormal-exit/STOP-grace比較を
consumerとして保持する。full successful API trace・重複build output・外部の重複research
原文・同じclosed receiptのB側copyは退役した。以後はこのディレクトリのgzip/original research
とregistryを使う。既存controllerを再現するときは必要なseed receiptをGitから復元し、新しい
controllerはcanonical Git receiptから読む。B側の元pathを存在する再開checkpointと扱わない。

保持buildは約136 MB、次の主モデルbaseline/native機能適合がconsumer。observerは次の
corrected-range診断の例外切り分けがconsumerで、normal long completion/recipe置換時に
再検討する。モデル・ユーザー設定・サービス・GPU状態はcleanupで変更していない。
mainはbeddab2fのまま。default反映とfresh Niko同期は未完了であり、この監査の完了とは別。

強制deadline後の別プロセスでも小runtime healthを確認した。64回のchanging cross-queue
handoffと境界/ownership/sticky errorのfixtureがexit0、UR success832/ZE success3325、
fresh fault0/dump無し。現在GPUを使えることの確認であり、終了したモデルの正常終了や
長文/速度合格を後から成立させない。このcontrollerはcanonical Git receiptから直接読む。
その成功raw traceもreceipt/summaryのcommit照合後に退役した。今回の累計は99ファイル、
26,221,242 logical bytes。保持する代表engine.stderrは引き続き1件/40,511 bytes。


## 61055入力の正常終了と設定復元時の新しい失敗

v6は同じ3fa2375eab38... binary、model/context/KV/cache/spec/prefillで、URを
FULL_VALIDATIONからPARAMETER_VALIDATIONへ切り替えて完走した。
ZE parameter検査・warning、phase timers、進捗・SYCL throw観測を維持。
全体wall2700秒、単一longのsoft STOPは開始1500秒後、少なくとも300秒の猶予を設計した。
実行前のhost-only FIFO fixtureで別prompt/古いPID identityを拒否し、正しいowned pipeだけに
STOP改行を送ることを確認した。実モデルはsoft STOPを使わず正常終了したので、モデル上の
STOP試験が合格したとは扱わない。

- short28/reuse0/8output: prompt19140.8ms、decode6468.7ms。
- long61055/reuse0/1output: prompt968021.2ms、decode173.1ms、TTFT968.601s。
  63.1/5.8t/sは補助診断の値で、quiet速度・長いdecode・正式bench.sh値ではない。
- native engine0、server0、wrapper/owner0、全owned session空、cleanup signal不要。
  fresh GPU fault0、dumpなし、同一boot。元のSYCL範囲例外なし。
- 61055全入力の正常処理を確認した。厳密65536入力、262144全入力、近傍出力・最後cell・
  継続/内部conversation RESTOREの合格には置き換えない。

R466の過去primary184.61t/sは4007入力/context8192/FP16/cache512であり、現条件と別。
現在診断で有効にしていなかったexact XMX tile8/compact、same-queue event省略、ring8を
別候補として明示した。native GGUF presetとexpert-hit graphは元から有効。
FP16-only prompt native-KV設定は現int8には適用しない。

current compilerのXMX MMQ fixtureは105.36秒で正常終了した。元のreductionとのbitwise比較、
17形式/odd shape/padding/canary/read-only/compactの検査を含み、UR success10307、
ZE success20956、fresh GPU fault0。小componentの合格だけで実model採用にしない。

続くv7はXMX exact tile8/compact/pack0、same-queue1/ring8だけを上記model設定に足した。
最初の27 batched入力のlayer0で進捗が止まり、既定60秒のwatchdogがstall_reportを出した。
2秒後、まだ未使用のVerifier queueのreleaseはfinished0ms、その後stack smashingでnative -6
(wrapper134)。server/runnerはこのengine失敗を受けて終了し、所有identityは全て退役、
fresh GPU fault0/dumpなし/同一boot。longを開始していない。quiet stock3を起動しない。

cold XMX componentの全体105秒とwatchdog60秒からJITの誤監視は候補だが、v7で停止した
正確なAPI/stackは未観測。host stack canary破壊の原因も未確定。timerを伸ばすだけで修正と
認定せず、次は引数別の実行/ZE・URの初回ログとhost abort stackを取得する。
同じ設定を無対策で再実行しない。GPU reset/unbind/services変更は行っていない。

run_server_job_v3以降は失敗時もreal engine exitを保存し、両native/server正常終了だけを
passedにする。準備したquiet stock recipeはv7同条件の正常終了を必須seedにするので、
現在は実行不可。無変更Niko bench.shとstandard clientのhashを保持し、--long --runs3を使用。
client自身は部分失敗でもexit0にする場合があるため、Main controllerは全chat/prompt/agentの
個別成功、long3回/reuse0/実入力数/phase有限値と正常exitを検査する。stock約32049入力は
nominal32Kで、strict32768以上や64Kと呼ばない。全個別値と失敗を保持する。

## HTTPのVRAM寿命を実際の入口で照合

R471はstandalone CLI6235以降の順をHTTPへ適用したため、Mainが棄却しR472を依頼した。
HTTP --serveは4699でVerifierを初期化し、5217READYより前にprefill-owned workspace/MTPbindを
保持する。no-prefill-borrowのPrefill spはserve scopeに生存し、decode/end-of-requestでresetしない。
一方、CLIにはprompt後prefill.resetがある。2入口を同じ寿命としない。

さらにMainが原engine logと照合すると、startup freeVRAMは3936→1686→1405MiB、Verifier arena
63.8MiBとMTP subset head212.9MiB。short後のT1/T4 graph captureで1176.5→1046.7MiBとなる。
R472へ渡した「1405→1047が両upfront」はMain入力の誤りで、訂正をregistryに保存した。
約358MiBはcapture/runtimeの増分であり、Verifier arenaの容量でも解放が保証された容量でもない。
現在longには既にcaptured graphsが生存。reclaim量は破棄後のallocator ledgerが必要。

prefill中のdecode用graph/weight退役・再構築と、decode中のprefill workspace返却/ExpertCacheへの
振替は別の具体的候補。cache/raw pointer/Verifier graph/MTP/pendingcommit/STOP/RESTOREの
queueとleaseを全て排他・drainし、途中失敗を残さない適合が必要。未実装・未測定。
R468/R469のparallel protocol/elastic APIの2つの本家非推論機能も未完了。

R470はcurrent chunk-major stream_allをsourceから集計した。64K入力/cache128の4K chunksでは
expert payloadだけで最大804.677GB、最も有利な128expertの保存を仮定しても799.225GB。
1000t/sには約12.2GB/s、8Kなら約6.1GB/sが必要な条件となる。過去6.5GB/s probeを現在帯域と
断定しない。routing-only ring8、实际residency、8Kfit、overlap、必要computeは別条件。
実際のDMA完了byte counter/profilerやhardware ceilingはまだ測定していない。

R466–R472のLuna原文・Main訂正をregistry/Gitへ保存する。調査から返るまで途中介入せず、
Mainはrealmodel・MMQを直列管理しながらsource/設定/報告を照合した。R473/R474へ新失敗の
coldJIT/watchdogとstage/stagerを分けて反復依頼した。枝は増やさない。
main/default反映とfresh Niko単体の設定最適化付き評価・選択的統合は引き続き未完了。


新失敗後も別プロセスの小runtime healthがexit0、UR success832/ZE success3325、
fresh fault0/dumpなし。GPUを現在使える確認で、v7のmodel失敗を合格へ変えない。
R473/R474のLuna原文も返答後にhash照合して登録した。関連fixture105秒は全体時間であり、
単一zeModuleCreate/単一layerの所要時間を測った値ではない。次はcold初回APIログと
ホストstackを優先し、watchdogの引数で限定した長い監視猶予とphase/診断のdata raceを検討する。
R473のVerifier::diagのplain last_t_/last_pos0_読み取りは、verify実行と並行すればdata raceとなる
source上の修正対象だが、今回prefill中のcanary破壊原因としては未確定。


## この境界の保存と保持

0eb55689にv6正常終了/v7失敗/全調査とrecipe、4433319eにclosed launch config/plan原文を
commitした。全39receiptのarchive/decoded hash、全16研究原文、closed owned identityの
消失を確認してretentionを実施。累計160ファイル、37,979,345 logical bytesを退役した。
volume snapshotや実空き容量の増加は推定しない。以前のdeadline rawはnormal longと
STOP-grace/元control evidenceの記録により現在consumerが無くなり退役。

未解決v7のengine.stderrだけ4605 bytesを保持する。SHA
ea9cd15d925cf13681edd9e4707f070d9391a602eeeb15f2095accfbe21f5fd5、
Main責任、上限256KiB、consumerは次のcoldJIT/watchdog/host-stack原因調査。
解決・countermeasure/recipe置換後に見直す。原failed/complete=falseは変更していない。
B側record/config/plan等はcanonical Git gzipへ退役し、次のcontroller/researchはGit receiptを使う。
build約136MB/4GiB budget、observer16KBは次の原因調査に必要な版へ置換または退役。
MainのGPU/server/ownerは全て終了、常設serviceとGPU bindingに変更なし。
準備だけの重複plan JSONはtemplate-retirement.jsonlにhash/理由を保存して削除した。

主branchはfeature/sycl、remote/local枝とworktreeは引き続き2つ。
mainはbeddab2fのまま、必要なnative非推論機能の適合とdefault反映、fresh Niko評価は未完了。
Xeの比較値をprimaryへ移さず、source由来・APIの義務・採用した計算部品を今後も分ける。
