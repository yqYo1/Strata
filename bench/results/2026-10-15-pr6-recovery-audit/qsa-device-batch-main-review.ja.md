# QSA indexerのdevice一括処理: Main実装・component確認

2026-10-15。Solの`qsa-indexer-device-batch-v2/source.diff`をMainが実際のrootへ適用した。
既定OFF、`STRATA_SYCL_QSA_INDEXER_DEVICE_BATCH=1`だけでprefill batchを新方式へ送る。
単一token/decodeのappend kernel本体は変更していない。
Luna R682/R695のsource reviewと、Mainによるproduction/planner/fixture全体の確認後に実行した。

元のbatch処理はn token分のsingle-cell kernelを同じin-order queueへ逐次submitする。
新方式は必要なposition0 seed、完了した4-cell blockごとの独立workgroup、tail/spare/block_posの
単一writer finalizerを同queueへ順に送る。通常の非zero開始の一batchは2 launches、
zero開始のmulti-cell batchは最大3 launchesとなる。n1は元kernelをそのまま使う。
新しいallocation/host readback/waitは追加しない。GPU内演算を増減させた際のend-to-end速度はまだ未測定。

## Buildと実行環境

`apply_build_qsa_device_batch_v1.py`は実78453入力のscratch alias OFF/ON比較が閉じてから、
新しい`main-qsa-device-batch-build-root-v1/build`へ119 build stepsを実行した。
旧library/binaryは不変。production strata、actual linked GPU fixture、host plannerをcompile/linkし、
production plannerの範囲/overflow/flag checksも実行してPASSした。
actual indexer/prefillの`-fno-fast-math -ffp-contract=off`と全GPU consumerの
`-fsycl-id-queries-range=size_t`をcompile_commandsで確認した。
新native SHA256は`feb1735c12d92c0ab87cada63dd146165f0622159abb328a9e5bcf48d02dc6ee`。
compiler、全source、configure/build commandとbinary hashは`qsa-device-batch-build-v1.json`にある。

全GPU processはMainがlock下で直列実行した。元のcopy-offload/direct-submission無効化、
same-queue、persistent-cache無効化などの安全設定を維持する。環境はexec前に固定し、runtime中に変更しない。
最初のsmallはURのflushed API traceとparameter validation、以後はvalidation/warnings付きで実行した。

## 実GPUの個別結果

| suite | scaling / table / mapping / API | state比較 | score/ID比較 | 不正入力拒否 |
| --- | --- | ---: | ---: | ---: |
| small | none / analytic / plain / direct | 64 | 43 | 0 |
| small | yarn / short / mrope / dispatch | 64 | 43 | 0 |
| invalid | none / analytic / plain / direct | 35 | 0 | 34 |
| invalid | yarn / full / mrope / dispatch | 36 | 0 | 35 |
| large | linear / full / plain / direct | 9 | 5 | 0 |
| large | yarn / short / mrope / dispatch | 9 | 5 | 0 |
| physical | none / analytic / plain / direct | 3 | 1 | 0 |
| physical | yarn / short / mrope / dispatch | 3 | 1 | 0 |

全223 state比較、98 score/ID比較がbyte単位で一致した。全69 rejectionは各caseで
before/afterの全stateと両guardを比較し、無変更を確認した。
smallはそれぞれ独立したzero-start n2/n3/n4、複数の途中位置・分割chunk、synthetic saved-image継続を含む。
largeは4096/8192と端数、identity selection2051/sparse2052、未来block sentinelと選択ID範囲/順序を含む。

physical2 processはそれぞれ実65537 pooled rowsとguardを確保し、262140 tokenの連続prefixを
実single-cell oracleで構築してcloneした。その後の最後4 tokenについて全state、block_pos=262240、
spare65536、次のguard65537まで比較した。これは**componentの物理境界**であり、
全modelを262144 token動かした結果ではない。

GPU8 processすべてnative0、全owned process退去、同boot、fresh fault0、dumpなし。
`qsa-device-batch-small-v1.json` / `qsa-device-batch-matrix-v1.json`に各processの個別結果、
environment、stdout、timing/owner identity、fault boundaryを保存した。

## Production flagとゼロ行経路

`qualify_qsa_device_batch_dispatch_v1.py`はactual production indexer libraryへ独立fixtureをlinkした。
新しい10 processでflag未指定/0/1のn0 no-opと、空文字/01/true/-1/2/前後spaceの拒否を確認した。
全process exit0で、拒否理由はproduction parserの正確なmessageと一致した。
UR traceでもcontext/queue作成、USM allocation、kernel enqueueが無いことを確認した。
`qsa-device-batch-dispatch-v1.json`に10個の個別結果を保存した。
これはhelperだけのflag testではない。positive dispatchは上記GPU matrixで別に確認している。

## 保持と残る確認

`retire_qsa_device_batch_trace_v1.py`は初回7999 API returnsがすべてsuccessで、entry件数とも
一致すること、実append_completed_blocks/finalize_device_batchのkernel creationを確認した。
件数・kernel名・結果をcompact evidenceへ残し、不要になった全成功trace2213882 Bを削除した。
原receipt/status/hashは不変。新しいbuildは次の実model比較がconsumerなので保持する。

既定OFFを維持する。次は同じ新nativeでflag0/1の長大入力境界とMTP callback、decode継続を比較し、
その後にinstrumentation無しの複数回official bench.shでPPを測定する。
実conversation RESTORE、cancel/error回復、全modelの物理262144、task品質は未合格。
componentのsynthetic restoreや最終4 token比較をこれらの代用にしない。
直近のclean PPは別binary/配置の235.3/239.5/239.2 tok/s、中央値239.2のままで、
このQSA変更の速度向上や600/1000達成を主張する材料ではない。
# 実78453入力の同native診断pair

後続の`qsa-device-batch-model-{0,1}-v1.json`は両方正常終了。
新規fault0、dumpなし、native/server0、所有process退去、head復元・MTP再capture後24生成まで完了した。
alias1、C8192、FP32/R_GPU0、Direct0を固定し、QSA batch flagだけを別processで変更。
`qsa-device-batch-model-comparison-v1.json`は同一request bodyと実read78453/reuse0を確認し、
全48層×選択16行のFP32 hidden全byte一致・有限を検査した。
各arm内の752 store/next-input行も一致。未選択行、追加同期無しのrace、実RESTORE、
全モデル品質、物理262144はこの比較で合格扱いしない。
診断PPはOFF235.5 / ON247.6 tok/sで、clean速度比較ではない。
固定順0,1,1,0,0,1で各arm三つの独立processを測るclean recipeを準備済み。
同じ実78453入力とthinking ON、24出力を使い、速度採用は反復結果を見て別途判定する。

上記の照合後、`retire_qsa_device_batch_model_v1.py`で成功raw二組とboundary/native診断を削除した。
全layer/chunk完了・必要なhead/memory/配置行とhashを`qsa-device-batch-model-compact-v1.json`へ保存し、
64244151 Bのpath/size/hash/理由を`qsa-device-batch-model-retention-v1.json`へ記録した。
元receipt/comparisonのstatusとbyteは変更しない。

## 独立process各3回のclean比較

`qsa-device-batch-clean-comparison-v1.json`は固定順0,1,1,0,0,1の全6processを照合してPASSした。
各requestは同じbody/hash、実read78453・reuse0、thinking ON・24出力で、native/server0、
owner退去、同boot、fresh fault0、dumpなし。source/native/linked librariesと設定はQSA flag以外同一。

| QSA batch | 個別PP (tok/s) | 中央値 | 最小–最大 |
| --- | --- | ---: | ---: |
| OFF | 227.6 / 219.8 / 225.7 | 225.7 | 219.8–227.6 |
| ON | 241.4 / 244.6 / 239.9 | 241.4 | 239.9–244.6 |

prompt_msから再計算した中央値比は+6.9969%。中央値TTFTは348.317→325.565秒。
各3標本で統計的有意性を確定したという主張はしない。
次の長大prefillチューニングではONを候補設定に用いるが、product defaultはOFFのまま。
24出力のdecode速度、task品質、実RESTORE、全model物理262144の合格には読み替えない。
別binary/配置で得た過去239.2中央値との単純比較で、この変更の効果を取り消したり上乗せしたりしない。
