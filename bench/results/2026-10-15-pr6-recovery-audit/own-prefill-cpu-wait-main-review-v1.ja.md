# CPU wait診断のMain審査と検証

2026-10-16 JST。ユーザーのCPU0占有指摘への独立した診断。
現行MIT sourceだけを参考にした6.1Sol案をMainが全diff/header/manifest/prepareと実callerを読んだ。
XeStrata implementation/helper/test/archiveは使わない。

採用した診断patch SHA256は`1acb2c04be892d72174545a9945f3c2210a007bcbf7b0c9661b61dfac3cd74df`。
既存7 filesと新規headerを変更し、12か所の明示的`wait_and_throw`を囲む。
Scopeは選択済みの長文`run_layer_major()`呼び出し全体を囲み、その関数のlocal cleanup後に閉じる。
短文warmup/decodeにはScopeを作らない。default OFF、exec前にexact1を指定した診断だけ有効。
queue/演算/同期順序/placement/待機policy/環境を書き換えない。
例外の再送出・両queueのdrain・sticky error・fail-stop cleanupの既存経路を保持する。

Mainのhost consumerは実production headerをincludeしてg++20/O2/Wall/Wextra/Werrorでbuildした。
時計はlinker `--wrap=clock_gettime`で異常を注入し、accountingの別実装は作らない。
8 process modesのoff、normal、false、unwind、nested-scope、fail-begin、fail-end、backwardがPASS。
normalは連続2 Scopeを試し、nested Waitはunionに一度だけ計上、別threadは除外。
original exception、errno、duration算術、無効時available0/duration-1を確認。
offはlineなし・CPU clock callなし。単なるimplementationの値の写しを確認する試験ではなく、
CPU仕事/sleep/throw/nesting/時計故障という異なるconsumer側条件で実headerの結果を判定した。
nonLinuxやScopeよりWaitが長生きする不正な所有関係の実行検証はしていない。

`build_own_prefill_cpu_wait_profile_v1.py`の最初のpreflightはpackage directoryをfilesと誤記して
FileNotFoundErrorになった。この時点ではroot/receipt作成・patch適用・build・GPU実行は無かった。
Mainが実packageの`proposed/`に修正して実行した。元GPU/機能結果をFAILからPASSへ書き換えたものではない。

実buildには既存の`main-canonical-bank-profile-build-root-v1/build`と、記録済みのIntel LLVM v711
compiler/runtimeを使った。案READMEの`build-sycl-aot`/icpx説明は今回の実buildでは使っていない。
build receiptは`own-prefill-cpu-wait-build-v1.json`。
strataと既存runtime/compat testsをMainのexclusive lock下でbuildし、実nativeは
`12e1303047197a23f2184dd7232a9d43a32f1e5ad570b49de6f2a763de9ca8d6`へ変化した。
既存runtime64cross-queue handoffs/bounds/move/sticky async errorとcompat event/graph/copy/lifecycleがPASS。
flushed UR tracing+parameter validationの別owned processesを使用、unexpected API error0、新規xe fault0、
normal exit/所有process退去を確認。runtime testの意図したhost_task例外は元fixtureの機能検証であり、
long-prefillがhost_taskを使うという証拠ではない。

明示的wait内のthread CPU/wallを区別できても、内部の純粋なpoll命令・callstackは同定できない。
outside-waitはsource copy/grouping/launchだけでなくsubmission内部やallocation/destructorのimplicit
blockingも含む。CPU時間の割合をそのままwall latencyや改善可能上限へ読み替えない。
clock計測・TRACE・validationの診断をclean速度標本へ加えない。

先行clean6process比較は別binary beda08、actual78453、各arm3回で正常終了。
control PP `[337.6,339.3,339.2]`、候補 `[349.3,349.6,349.1]`、中央値+2.977594%。
事前engineering screenはpositive、統計的有意性とは呼ばない。default flag採用は未実施。
`own-selector-clean-comparison-v1.json`が個別PP/TTFT/elapsed/hashを保持する。
600超の反復prefill目標は未達。quality/長いdecode/RESTORE/physical262144の新規合格ではない。

実78KのCPU診断結果と判断は`own-prefill-cpu-investigation-v1.ja.md`へ追記し、
元structured receiptは不変に保存する。成功ログはcompact検証・所有退去後にmanifest付きで退役する。
