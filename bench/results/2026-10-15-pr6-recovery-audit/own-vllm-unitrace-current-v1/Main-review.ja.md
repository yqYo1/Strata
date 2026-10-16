# R860: 現行6354の外部unitrace collector

**SOURCE-ONLY / UNRUN / SOURCE_BLOCKED**。この納品ではビルド、テスト、GPU、モデル、プロファイラ、プロセス調査、Gitコマンド、サービス操作、cleanupを実行していない。新規のこのディレクトリだけを変更した。エンジン、既存fixture、独立FP64/dyadic oracle、許容値、既存レシートは変更していない。R856への依存・連絡はない。作成済み実装はレビュー資料として保存し、現行toolによる実行をhard gateで止める。

## 実行を止める具体的な理由

固定された公式unitrace CLIの`CreateConfigLog`（unitrace.cc:905付近）は、必須`--result-dir`で`StringifyDevice`を呼び、`zeInit`（同:840付近）を実行する。その後mainは`UNITRACE_LD_PRELOAD_OLD`（同:1171）、`ZE_ENABLE_TRACING_LAYER`/`ZES_ENABLE_SYSMAN`（同:1202以降）、`LD_PRELOAD`（同:1231）をSetEnv/setenvで書き換え、最後にexecvpする。実際のSetEnvは公式`utils/utils.h:154`以降の`setenv(name, value, 1)`。メトリクスを要求しない今回の必須CLIでもこの順序を通る。native imageがexecで新しくなることは、launcher自身のprovider初期化後の環境変更を無かったことにはしない。

Orcaの担当Dispatch経由でMainへ適用範囲だけを照会した。回答は **deliver-source-blocker**：「元のprocess-environment immutability制約を維持」「provider初期化後にlauncherが環境を書き換える挙動への例外は許可していません」。新規実装・実行の依頼ではなく、元の制約の確認として記録した。固定binaryを変更する、必須result-dirを外す、独自のLD_PRELOAD launcherへ置き換える、環境条件を免除する、といった回避は行わない。Mainが別途次の方式を判断する。

controllerは引数解析直後の`SOURCE_BLOCKER`で終了する。helper import、プロセス調査、lock/root/receipt作成、health probe、unitrace/nativeの起動へ進まず、解除フラグはない。以下の契約と起動形は保存した未実行実装の説明であり、現在のtoolを実行してよいという指示ではない。component PASSもmodel timingも存在せず、qualified collectorの過去PASSでこの未達条件を置換しない。

目的はAttentionの粗いキュー区間から実kernelの実行時間を分けて見ること。既存39K BF16のAttention33.9610秒、MoE20.2019秒、GDN9.2465秒、upload8.39997秒、wrapper98.491秒やQSAのAttention36.2963秒、wrapper100.987秒は、依頼で与えられた非排他的な区間であり、device busyやcritical pathの値へ変換しない。このパッケージは新たな測定値を報告しない。

## 出典とcollectorの実際の動作

公式intel/pti-gpuの固定commit `6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033` の [README](https://github.com/intel/pti-gpu/blob/6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033/tools/unitrace/README.md)、[CLI](https://github.com/intel/pti-gpu/blob/6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033/tools/unitrace/src/unitrace.cc)、[collector](https://github.com/intel/pti-gpu/blob/6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033/tools/unitrace/src/levelzero/ze_collector.h)、[logger](https://github.com/intel/pti-gpu/blob/6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033/tools/unitrace/src/utils/logger_factory.cc) を根拠にした。ローカルghq rootはこのcommitのmasterを指し、退役した旧tool worktreeは存在しないためroot側のソースを読んだ。公式固定URLから取得したbytesとの一致・各SHAはmanifestに記録する。Gitコマンド、help、tool実行で調べていない。

既存private tool build receipt SHAは `692da9f9d706fea659e78378a397c71d86b08cb9fba0c994e390de29b6a20647`。使用する製品は同レシートのbuild_pathにあるunitrace (`66a828c49488052487815224ad6fdb36dfe0f53b0d64b5b906cde5d113e7c2c0`) とlibunitrace_tool.so (`8a8fe37441606bf66ab42fc39cc447b5257f48d09f04fbcc60703e30826671eb`) だけ。XPTIの公式LLVM libxptifw/headerのpinも同レシートから検証する。旧kv-resolver/model profileはcollector来歴のみで、現行モデルの正常性ゲートには使わない。XeStrataのengine/helper/fixtureは読まず、既存pin中で明示された公式LLVM installだけを許可する。新規ソースにはリポジトリLICENSEを適用する。

CLIは正確に `unitrace --device-timing --host-timing --result-dir PRIVATE_PROVIDERDIR NATIVE ARGS...`。メトリクス/KMDなしの分岐はforkせずexecvpで同PIDのimageをnativeへ置換する。CLIがnative exec前にcollectorをLD_PRELOADへ設定する。Mainやserverのglobal環境には設定しない。CLIのrun_configは`unitrace.PID/run_config.json`、nativeの集計は`native_comm.PID/{device_timing,host_timing}.txt`に出る。同PIDでもディレクトリ名は異なり、Linux commは15文字に切れる場合がある。この違いを許容しつつPID・完全argvを一致させる。

CLIの既定`UNITRACE_ChromeEventBufferSize=-1`はcollector有効化ではない。Chrome/metrics/call logging/device timelineの有効化は拒否する。API全呼出ログ、UR tracing layer、Chrome timeline、device timelineは要求しない。単なるtool exit0は成功ではなく、非空のL0実compute kernel行とhost API行、両tableのns統計、native実行・fixture oracle・通常終了のすべてを要求する。

## 実行契約

- 現行source `6354f0b4ff3731466795410e11ca9ef785c9d326`、build receipt SHA `990635731be1198706abeb682817fdab247f8f51d31bab971d3d4f4db53cb639`、engine SHA `85643f3926bb9d212945659cb7ddb264e101ca9b2d5fa8f19430e945140ad191` を固定。各操作前後と最終閉鎖でsource432/build314/product13/runtime37、tool、DSO、model metadata/stat、profile、recipe/source packageを再確認する。model shardの識別は既存bytes+mtime_nsで、全shard SHAを測ったとは主張しない。
- Mainが`--prior`と`--prior-sha256`で最新の通常閉鎖current-engine PASSを明示する。active/incomplete/FAIL、別boot/GPU/source/binary、残存PID/start、fault不明、dumpあり、旧collector健康証拠は拒否する。既知の旧selected-smoke-v2 SHA `4f888624...51968f`は代用品として明示拒否する。activeなselected-screen-v2は実行時に閉鎖済みの新SHAをMainが選ぶ必要があり、この納品時には読んでいない。selected v2は固定の元compare_screen.pyを使ってschema/binding/recipe/request/capture/FP64証拠を読み直す。quality flags=falseを正当に受け入れ、品質合格へ格上げしない。
- `component`は既存`sycl_prompt_native16_test --smoke`、続いて`sycl_vllm_native16_test --device 0 --suite smoke`だけ。前者の2ケースFP64_full/finite/guards/inputs、後者の32個のcompact/wide・FP16/BF16 dyadic exact-RNEケースを元の基準で要求。新規math fixtureやtolerance変更はない。独立プロセス各90秒以下、2つを合わせたworkも90秒以内、session/native RSS4GiB、text+provider8MiB、private64MiB。whole135秒のうち45秒を閉鎖に予約する。最初の失敗で次を起動しない。
- `model --arm fp32|bf16|qsa`は同package/collector/source/GPUのcomponent PASS+SHAも明示必須。新しい独立プロセスを1armだけ起動する。coarse controllerと同じrunner、公式bench.sh/client、固定wrapperを使い、wrapperのengine/hash/direct-mode/launcher/identity公開部分だけをB内の新ファイルへ適応する。native stdioは継承する。identity JSONの公開はpartialからのatomic renameとして観測競合を避ける。元wrapper/server/clientは変更しない。
- modelは`bench.sh --prefill-only --prefill-tokens 32768 --workload-id prefill_v1_20261015`、thinking ON、24 generated、actual39262 read/39261 batched、chunk8192、context+resident65536、GPU residual32768、expert cache128/per-layer、5workers、canonical/native banks+scratch alias、ROPE1、MTP4。strict request journal、zero reuse、48layer×5chunk、要求placement/routeを確認する。prompt再利用、backoff、別構成での代替PASSはない。
- native900秒、whole1800秒、閉鎖予約45秒、RSS88GiB、CPU21600秒、private512MiB。native text/fileは62MiB（65011712B）。provider+全textはwork中62MiB、最終64MiBで、Owner aggregate64MiB内に閉鎖予約2MiBを残す。過去63569928Bのrawにもfilecapが届くが、このcollectorではtensor capture自体を有効化しない。journal用Ownerを別の`closure/`に置くため、予期しないnative aggregate超過でも独立した最終journalを試みる。元d1a98dd5失敗とbf9d6f00 dense-NLL失敗はbyte-identicalのままquarantineする。
- 共通`owned-v0141-measurement.lock`を保持。Mainが他GPU/build/cleanupの不在を管理し、controllerは既知recipeのモデルを含むsame-UID owner scanも行う。未知ownerは採用・signalしない。生存中のpollではprivate/log/provider量・RSS・deadlineと実exe/argv/env/mapsを確認し、Ownerが実PID/start/session/pidfd/stdio/reapを管理する。component nativeはcontroller direct childのexec置換。modelはrunner→server→wrapper→unitrace/nativeであり、controller direct nativeと偽らない。wrapperのwait結果とrunner/server正常終了、固定serverのQUIT-first経路を結合する。
- child環境は凍結された現在のclean recipeから明示allowlistで作り、Ruby/lint/GEM/PYTHONPATH等は継承しない。safeURv2、copy-offload無効化1、direct-submission0、samequeue1、cacheoffを要求する。collector追加環境はnative側/procで保存する。provider CLIによるexec前の設定とnative imageの環境を分けて記録し、native側の観測間変化を拒否する。モデルには既存coarse/QSA/issuing-thread CPU profilesを有効化する。
- 各操作前後とfinalにfresh kernel cursorを進め、boot/PCI/kernel/xe/dumpと全PID/start不在を確認する。失敗・fault・closure errorを配列に蓄積し、後続の空窓で消さない。後処理失敗も全体FAIL。成功前の不完全なprovider/raw/command evidenceもレシートへ保持する。リトライ、health自動発見、ゲート免除、reset、サービス、システム調整、artifact cleanupは行わない。Ownerの失敗時pidfd終了処理は有限な所有プロセス閉鎖のためだけ。

実entrypointも確認した。`cmake/sycl_backend.cmake:138`とbuild receiptのactual compile entryは`src/program/generate.cpp`を指す。同sourceのmain冒頭はCUDA_MODULE_LOADING未設定時にEAGERをsetenvする。preloadされたcollectorのconstructorより後にこの呼出をさせないため、同じ`CUDA_MODULE_LOADING=EAGER`をnative exec前の明示環境へ追加する。これはCUDAの実装・実行ではなく現行SYCL entryの既存defaultを前倒しする設定で、変更をレシートに記録する。対象fixtureは環境の切替を行わない。`/proc/PID/environ`はexec時の環境像であり、任意の将来setenvの不存在そのものを証明しないため、対象sourceとproviderのfollow-child既定1経路も併せて確認した。

## 読み方と限界

device tableはLevel Zeroのkernel timestamp global start/endをmaskし、device timer frequencyからnsへ変換した実行差分。公式collectorはwraparoundを扱う。host tableはAPI入口/出口のhost経過時間で、CPU thread timeではなくblocking/pollingを含み得る。CPU待ち時間の補助行は別のCLOCK_THREAD_CPUTIME_ID accountingとして保存する。

保存するのはkernel/API名ごとのcalls・time_ns・average/min/max・元table内のpercent、各tableとprovider全fileのsize/SHA、native環境/DSO identity、request/終了/fault証拠。kernel名のC++ template内カンマを壊さないよう、行末の6個の数値列から解析する。元tableのpercentはそのtable内のsumが分母で、GPU busyの百分率ではない。memory operationも別名の行として残す。新たなphase合計や自動のAttention/MoE分類は作らない。

デフォルト集計はnativeプロセス全体であり、startup、初回prefill、24-token decode、teardownを含む。共有kernel名をprefillだけへ分離する証拠はなく、coarse区間との差をCPU bottleneck/overheadと断定できない。per-instance timelineやbusy-unionを取得したとも主張しない。Mainがまず実kernelの分布を確認し、必要なら別の具体的なphase分離質問を立てる。全timingはdiagnosticであり、clean speed/quality/262144/RESTORE採用フラグはfalseのまま。

短命fixtureでexe/env/mapsの観測が間に合わない、providerが想定外ファイルを出す、現在のURv2/AOTとの相性が悪い、既存selected捕捉が既に退役した、pin不一致や期限超過などの場合はFAIL/blockerになる。toolの過去PASSで代替せず、Mainが証拠を確認する。source-only納品自体はこれらの現在の実行互換性を保証しない。

## 保存した起動形（現在はhard gateで拒否）

以下は将来のMainレビューに備えて残した正確な引数形で、**現行toolでは実行しない**。呼び出してもhard gateでSOURCE_BLOCKEDになる。`R860_REVIEWED_MANIFEST_SHA256`は納品時に報告したmanifestのSHA、`R860_LATEST_CLOSED_SHA256`はMainが選ぶ**閉鎖後の**selected-screen-v2レシートの新SHA、`R860_REVIEW_UTC`は実行時から48時間以内の明示UTCを表す。root/receiptの再使用や条件免除のオプションはない。

```sh
R860_PACKAGE=/home/yayoi/orca/workspaces/Strata/sycl/bench/results/2026-10-15-pr6-recovery-audit/own-vllm-unitrace-current-v1
R860_A=/home/yayoi/orca/workspaces/Strata/sycl/bench/results/2026-10-15-pr6-recovery-audit
: "${R860_REVIEWED_MANIFEST_SHA256:?Set the reviewed manifest SHA from worker delivery}"
: "${R860_LATEST_CLOSED_SHA256:?Main selects the newest normally closed current-engine PASS SHA}"
: "${R860_REVIEW_UTC:?Set an explicit UTC deadline within 48 hours}"
/usr/bin/python3 -B "$R860_PACKAGE/controller.py" component --run-id v1 \
  --manifest-sha256 "$R860_REVIEWED_MANIFEST_SHA256" \
  --prior "$R860_A/own-vllm-selected-screen-v2.json" --prior-sha256 "$R860_LATEST_CLOSED_SHA256" \
  --retention-review-utc "$R860_REVIEW_UTC"
```

未実行実装ではcomponentの閉鎖済みPASSのSHAを`R860_COMPONENT_SHA256`へ明示設定する契約である。続く関数は1回の呼出で指定armだけを選ぶ。現在はcomponent/modelの両方とも上記hard gateで拒否する。

```sh
: "${R860_COMPONENT_SHA256:?Set the exact closed component PASS receipt SHA}"
run_r860_model_profile() {
  /usr/bin/python3 -B "$R860_PACKAGE/controller.py" model --arm "$1" --run-id v1 \
    --manifest-sha256 "$R860_REVIEWED_MANIFEST_SHA256" \
    --prior "$R860_LATEST_CLOSED_RECEIPT" --prior-sha256 "$R860_LATEST_CLOSED_SHA256" \
    --component "$R860_A/own-vllm-unitrace-component-v1.json" --component-sha256 "$R860_COMPONENT_SHA256" \
    --retention-review-utc "$R860_REVIEW_UTC"
}
# Main explicitly sets R860_LATEST_CLOSED_RECEIPT and its matching SHA before each call.
# Main chooses ONE call after inspecting preceding PASS receipts:
# run_r860_model_profile fp32
# run_r860_model_profile bf16
# run_r860_model_profile qsa
```

新しいB rootは`main-vllm-unitrace-component-v1`または`main-vllm-unitrace-model-ARM-v1`、A receiptは`own-vllm-unitrace-component-v1.json`または`own-vllm-unitrace-model-ARM-v1.json`。全rawの保持ownerはMain、current consumerは現行6354 kernel attribution、byte budgetはcomponent64MiB/model512MiB、reviewはCLIのUTC期限以内。controllerは何も削除しない。

納品時に許可された確認はソースの`compile(..., 'exec')`によるメモリ内構文確認だけ。controller/helper/fixture/CLIを実行せず、productや既存reportに書き込まない。manifestはcontrollerと本reviewのbytes/SHAからcontent hashを持ち、自身を含む3ファイル全体のhashは納品メッセージ側で別途示す（自己参照hashを作らない）。
