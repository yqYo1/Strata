# 長文prefillのCPU占有調査

2026-10-16 JST。専用long-prefillの調査。GPU計算のCPU fallbackやCPU律速を、CPU使用率だけから確定しない。
ユーザーの「CPU0ほぼ100%」を改善対象として、clean selector比較終了後の独立診断を優先する。

## 実測と確定した経路

`own-prefill-CPU-thread-observation-v1.json`は実nativeのPID/start/bootとrunを記録した
read-only `/proc/<pid>/task/<tid>/{stat,wchan}`の5.002068秒観測。
mainはuser4.88秒+system0.11秒、0.997587core。他の観測threadは0CPU、sleep。
12 logical CPUに対する約8.31%と整合する。callstack、CPU命令種、phase全体の比率は未測定。
既存clean runへattach/profiler/環境変更/affinity変更は行っていない。

`own-prefill-cpu-affinity-observation-v1.json`は別の実native3194248/start43704875の
read-only `/proc/<pid>/task/<tid>/status`観測。mainのallowed CPUは`0`。
29 threadsの内訳はCPU0のみ6、0-11が18、CPU1..5が各1。
`src/core/session.cpp`の`SessionLoopScratch::init()`がhostを最初のphysical coreへ固定し、
`free()`で元のmaskへ戻す。`src/program/generate.cpp`はprefill前からこのsession scratchを保持する。
この固定はCPU0への集中を説明するが、CPU時間の消費元・性能への影響を証明しない。
プロセス全threadがCPU0だけに限定されている、という意味でもない。

`src/prefill/prefill.cpp`のhost grouping経路は、GPU routerのexpert IDをhostから読める場所へ
送ってqueue同期し、CPUがcounts/offsets/slot/source行対応を作り、対応表をGPUへ送る。
expertのdequant/GEMM自体はGPU queueへ送る。これはCPUが行うindex処理と往復・同期であり、
単にレイヤーのH2Dを提出するだけの経路ではない。現在のCPU使用率の主因かは未測定。
`src/prefill/layer_major.inc`の各層512 expertのupload slab準備もmain CPUの明示的仕事。

RuntimeはGPU deviceだけを列挙する。oneMKL GEMMもそのGPU queueを受け取る。
確認したactive sourceにはCPU GEMM/router/attention fallbackを選ぶ分岐は見つからない。
ただし、library内部の実行場所やhot stackまで証明したことにはしない。
host_taskの一般互換APIは存在するが、探索したprefill経路では使われず、verify側のflag通知で使う。

Intel公式UR Low Power Events文書はLevel Zeroのhost同期がdefaultでbusy pollingすると記載する：
https://oneapi-src.github.io/unified-runtime/core/EXP-LOW-POWER-EVENTS.html
これはCPU高使用率の候補を裏付ける資料で、当該runのstack/原因を同定する資料ではない。
`Runtime::wait_for()`のyield pollingにはactive callerが無く、そこを原因としない。

## R752のMain訂正

元レポートは不変に残す。実API入力78453に対しbatched long-prefillは78452。
chunk8192で9 full chunks+tail4724、各層10 chunks、48層で480 layer/chunk処理。
R752のtail4125は誤り。IDは48*78452*10=37,656,960個、int32の論理量150,627,840B。
これはwire counterではない。FP32 residualは78452*10240*4=3,213,393,920B、
GPU32768行=1,342,177,280B、host45684行=1,871,216,640Bで、実placement logと一致する。
R752のAPI入力行数を使った計算を採用しない。
residencyの`std::fill(begin,end,-1)`は各層で全48*512 entriesをclearするため、48層では
1,179,648 int writes。その後512 entriesをpublishする。
`STRATA_SYCL_PROMPT_DIRECT`はattention output側の設定で、MoE `grp_mapped`を決めない。
groupingは`m.grp_host != nullptr`で分岐し、mapped hostでもGPUとのデータ移動と同期はある。
`ONEAPI_DEVICE_SELECTOR=0/1`というR752の例を実設定と解釈しない。
実環境は`ONEAPI_DEVICE_SELECTOR=level_zero:gpu`。

## 次の判別

clean cohort中はsource/binaryを固定する。別source-only案をMainが全diff審査し、cohortが閉じてから
default OFFのthread CPU/wall計測を導入・build・小さい検証・独立したlong-prefill診断を直列で行う。
main phase全体と実queue/event wait内のCPU秒を集計する。大きなoutside-wait CPUが残れば
host grouping、source copy、kernel/oneMKL提出をさらに分解する。
wait内CPUが多くても内部のpoll命令まで同定したとは呼ばず、outside-waitにはimplicit blockingも含む。
低CPU化だけを性能向上と扱わず、依存関係のcritical pathとGPUの空きを確認する。
計測値やGPU correctnessのない段階で待機方式・CPU固定・GPU演算を変更しない。

CPUでの推論演算や不要なindex計算/往復を避け、GPU内で完結する長文経路を目指す。
転送準備・起動制御・入力検査のCPU仕事は明示して測り、全てのCPU使用率がゼロになるとは約束しない。
XeStrata source/helper/test/archiveは使用しない。R751/R752および追加R753はread-only Luna調査、
計測案は別6.1Sol、実build/test/GPUはMainが管理する。

## 実78K診断の結果

`own-prefill-cpu-wait-build-v1.json`のhost8 modesと小GPU2 testsがPASSした後、
`run_own_prefill_cpu_wait_model_v1.py`で別のowned nativeを実行した。
actual API78453、batched78452、C8192、FP32、GPU32768行、native/canonical bank1、Direct1、
専用prefill radix1、head release0、no-token-graph。thinking ON、24生成は完了markerである。
exec前にCPU wait profile1、TRACE1、UR parameter validation/warning loggingを設定し、その後不変。
診断なのでclean速度標本には含めない。

| 観測量 | 秒 | caller CPU時間に対する割合 |
| --- | ---: | ---: |
| Scope経過時間（local cleanup込み） | 223.946149935 | — |
| caller threadのCPU時間 | 216.981016976 | 100% |
| 明示的GPU wait内のCPU時間 | 168.473931550 | 77.6445% |
| wait外のCPU時間 | 48.507085426 | 22.3555% |
| 明示的GPU wait内の経過時間 | 168.870337914 | — |

2053 waits。内訳はcompat queue2033、Runtime compute6/transfer6/event8、direct0。
CPU時間はほぼcompat queue wait内にあり、waitCPU/waitWallは約99.77%。
available1/complete1、時計・accounting error0、open waits0、wait throws0。
48全層で各10 chunks完了、server/native正常exit、owned退去、新規kernel fault0、dumpなし。
元receipt `own-prefill-cpu-wait-model-v1.json`は不変に保持する。

このrunの1コア占有の大半が明示的GPU同期API内だったことは測定で確認できた。
Intel文書のbusy pollingと整合するが、runtime内部の純粋なpoll instructionだけを同定したとはしない。
CPU100%だけを根拠に「CPU推論が支配的」「CPU single-threadが律速」とする結論は支持しない。
一方でwait外48.51CPU秒は大きく、CPU律速を全否定する結果でもない。
source copy、host grouping、oneMKL/GPU起動、implicit blocking等への追加分解を優先する。
未対応oneMKL routineのhost実行はgeneric仕様上許されるため、GPU queueだけで内部fallbackを否定しない。
今回CPU演算fallbackの有無を全call単位で測定したものではない。
割合はCPU時間の内訳であり、直ちに削れるwall時間・改善上限・GPU稼働率ではない。
単一入力・単一診断processの結果として扱い、clean6processの速度分布と区別する。

成功full logsとduplicate recordは`close_own_selector_cpu_evidence_v1.py`でcompact evidenceを検証後、
exclusive lock下で退役した。202 files/1,618,773 logical bytes。
`closed-own-selector-cpu-compact-v1.json`と`closed-own-selector-cpu-retention-v1.json`が
progress、実wrapper/start、host/GPU gates、build proof、元path/bytes/hash/reasonを保持する。
元の全個別標本、status、CPU秒、fault結果は変更していない。physical空き容量の増分とは呼ばない。
