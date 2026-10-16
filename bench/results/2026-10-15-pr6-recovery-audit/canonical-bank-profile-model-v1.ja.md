# 独立canonical bankの実モデル検証と時間内訳

2026-10-16 JST。XeStrataは性能値の参照だけに限定する。今回の変更はこのfork自身の
sourceから独立実装したcanonical bankとbounded profileで、XeStrata sourceの移植・
翻案・書き直しを含まない。既知の派生診断撤去はxestrata-performance-only-main-review.ja.md。
この確認を包括的な法的保証とは扱わない。

## 同一入力の検証

buildはcanonical-bank-profile-align64-build-v1.jsonのnative
1694705b193d8afbcbd975632cf46489a6863e2fd91cf1197230033dcfb1f3df。
actual sourceはa39e44a4と同じ。official bench.shのprefill-only、actual/read78453、reuse0、
thinking ON、24生成、context262144、C8192、FP32 residual/R_GPU32768、Direct1、QSA device batch1、
scratch alias1、native bank1、head-release0、no-token-graphを使った。
新しくcanonical bank1とcoarse profile1を有効にした。

run_canonical_bank_profile_model_v1.pyのowned runがPASSで閉鎖。
native/server exit0、48層activationからFull復元、全owned identity退去、同boot、新規fault0、dumpなし。
legacy TIMINGやper-expert eventを有効にせず、既存checked drain後だけでprofile eventを回収した。
環境変数はexec前に固定し、実native起動記録にもcanonical/profile/R/dtypeを保存した。

compare_canonical_bank_profile_model_v1.pyは以前のclean native8fdcと、同request SHA・同設定・
レビュー済みsource変換chainを照合した。48層×選択16行=768 FP32行/7864320値は全byte一致・有限。
両runの752 stored行と752 next_input行の同run比較も一致。これは選択行とコピーの診断であり、
全tensor、task品質、generated token一致の要件、物理262144、session RESTOREの証明ではない。
no-token-graphの全phaseでsession_recapture=0をclosed comparatorが確認した。
実captured SessionGraphsを復元するケースは未合格のまま。

## 観測した時間

診断PPは以前293.8→今回331.9 tok/s。各一標本で、validation・TRACE・選択行capture・
今回のevent markerを含む。clean反復性能・改善率・ばらつきの判定には使わない。
engine prompt236351.7ms、layer-major wrapper235039ms。

| 観測区間 | queue interval ms | wrapperに対する比 |
| --- | ---: | ---: |
| QSA | 103980.0 | 44.24% |
| MoE | 43937.0 | 18.69% |
| GDN | 18020.2 | 7.67% |
| residual input | 15883.5 | 6.76% |
| residual store | 17401.6 | 7.40% |
| expert upload | 8897.67 | 3.79% |

これらはhost submission gapを含む非排他的なqueue interval。比は次の調査の優先度を決める
手がかりで、合計して排他的critical pathやAmdahlの改善上限とは呼ばない。
全12 QSA層の区間はそれぞれ8610.2〜8726.92msで、特定層だけの外れ値ではなかった。
期待するsample数input470/QSA120/GDN360/MoE480/upload48/store470も全て確認した。

CPU wallはexpert source5451.1ms、upload submit5850.95ms、upload drain3064.61ms。
outer host entry63.5312ms、run235404ms、restore881.363ms。
restoreは元RAM snapshotからのFull復元・PLE/graph/verifier処理を含み、D2H保存ではない。
startup snapshot作成とwrapper teardownはこの内訳の測定範囲外。

uploadのrequested payload50292326400B、inter-layer input/storeは各47×78452×40960B。
requested bytesをPCIe wire trafficと同一視しない。詳細と各層の値は元model receiptと
analyze_canonical_bank_profile_v1.pyのanalysis receiptに保存した。

## VRAMと次の一手

native phaseのBank時freeは4198129664B、Full復元後776114176B。
以前のcanonical OFF/同RのBank時free2666860544Bとの差は1531269120B。
異なるprocessの観測差であり、allocator/runtimeまで固定した因果測定ではない。
元decode配置への復元は今回881.363msで完了した。

最大の観測区間QSAをprojection/KV stage/append/indexer/selection/attention/outputに分解する
default-OFFの追加計測を、別の6.1 Sol実装agentへsource-onlyで依頼した。
Mainだけがレビュー・build・test・GPU実行を行う。MoE grouped GEMM案は今回優先しない。

R730を読んだ上で、もう一つの直交する配置診断を開始した。
run_canonical_bank_profile_r40960_model_v1.pyは同native・同入力・FP32でGPU prefixだけを
R32768→R40960へ変更する。増分335544320B=320MiB。
今回のBank freeはこの配置のcache+residual+512MiB要求3577741312Bを上回ったが、
次processの容量保証ではなく、そのprocess自身のadmissionが最終判断する。
このreport初版時点ではR40960は実行中で、成功・速度・数値は未判定。

## 並行調査と証拠保存

R727/R728/R729/R730を全読した。R729のsession_recapture assertion不足はclosed comparatorで0を
確認し、次R40960 supervisorにもassertを追加した。元実行recipe/receiptは変更しない。
R729の「run root不存在」はlive engine-startの記載と整合しない。実rootは存在し、final receiptが
まだ無かっただけである。「48 store phases」は470 residual store intervalと区別する。
未完了resultのpassed=falseをFAILともPASSとも判定していない。

次のLuna read-only調査R731/R732は同じ最新research-indexと新しい実測を共有し、
QSA selection/indexerとattention/KV stagingにscopeを分けた。実装agentと分離し、途中介入しない。

比較閉鎖後、旧8fdc selected raw31457280Bをexclusive lock下で退役した。
新canonical referenceは全byte一致で、次R40960/finer-QSA比較のMain consumer向けに保持する。
paired raw予算64MiB、次の比較終了時に再審査。
path/hash/replacement/reasonはcanonical-bank-profile-reference-retention-v1.json。
元FAIL/PASS/個別測定/recipe hashは不変。source-only過去reportもGitへ保存する。

prefill反復600超→1000、decode70は未達。default変更・品質採用・物理262144・RESTORE合格はしていない。
