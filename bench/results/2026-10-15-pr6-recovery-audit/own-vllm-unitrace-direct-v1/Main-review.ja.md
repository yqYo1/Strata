R864 direct collector source handoff — SOURCE_ONLY_UNRUN

`controller.py` を新規作成した。製品ソースは変更していない。構文検査、host test、build、tiny、GPU、model、profiler は全て **UNRUN**。Main が全文をレビューし、実行を管理する。

公式 `intel/pti-gpu` commit `6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033` の tree とローカル94ファイルの Git blob/SHA256 を照合した。既存 libunitrace_tool.so `8a8fe374…`、build receipt `692da9f9…`、XPTI と Level Zero tracing layer を固定した。製品6354の source432/build314/product13 に不一致はなかった。manifest は追加の ELF 候補49件、全 file pin587件を保持する。ELF候補は実行時のロード証明ではなく、native の実際の exe、PID/start、environ、map dev/inode/SHA を別に必須とする。

`unitrace` CLI/CreateConfigLog は一切起動しない。Python controller/server/wrapper に preload を渡さず、最終 native の `Popen(env=...)` にだけ設定する。`FollowChildProcess=1` で tracer.cc343 の LD_PRELOAD 書き換えを除外し、`CUDA_MODULE_LOADING=EAGER` で generate.cpp1455 の setter を除外する。Session/StartPaused/PTI/XPTI/metrics/Chrome/temporal controls は存在させない。実際の BUILD_WITH_ITT/OpenCL/OMP は0、L0/XPTIは1。初期化経路と分岐の監査は provenance.json に記録した。この限定レシピの根拠であり、全transitive runtimeの普遍的な環境不変性は主張しない。

まず現行 `sycl_prompt_native16_test --smoke` の FP64_full/finite/guard/input oracle と、native PID名のsummary、実ロードcollector/XPTI/L0 tracing layerを確認する。その最新closed PASSをMainが明示選択した場合だけ、別呼出しで vLLM BF16/QSA0 の official bench.sh を1回実行できる。自動移行・別arm・retryはない。比較条件は固定済み profile receipt `17038b32…` そのまま（39262 input/39261 batch、24 output、thinking ON、chunk8192、C/resident65536、R_GPU32768、cache128、workers5、MTP4、banks+alias、ROPE1）。追加はcollectorと既存warning validationで、coarse/QSA/CPUwait/STRATA_TRACEを保持する。

共有lock、boot/driver/dump、per-command journal cursor、所有PID/start/session/normal reap、最新health PASS+SHAを要求する。保護された同UIDサービスはcmdlineと可視性限界を記録し、未知のGPU所有者が全て不在とは主張しない。所有する子のpermission errorは免除しない。tiny全体300s/native90s、model全体1200s/nativejob900s（inner890s、bench720s）、closure45s+2MiB、native/controller/全side合計62MiB、Owner64MiB、private1GiB。全sideを台帳に残し、不正・余分・partial/truncated collectionはFAIL、原失敗を保持する。cleanupやtensor保存は実装していない。

出力は `/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-vllm-unitrace-direct-{tiny|model}-v1/record.json` と `/home/yayoi/orca/workspaces/Strata/sycl/bench/results/2026-10-15-pr6-recovery-audit/own-vllm-unitrace-direct-{tiny|model}-v1.json`。host APIとdeviceの各行、calls/ns/%/avg/min/max、全行合計、native oracle、request/samples、既存CPUwaitを保持する。device合計は非排他的で、busy union/critical path/occupancyを導けない。wrapper wallはcollector・map hashing・起動終了を含み、clean speedではない。quality/performance/default/full262144/RESTORE採用flagは全false。

再現用の独立した2呼出しを以下に示す。現在明示供給された最新public PASS `1f2293b4…` を指定している（private fixture binaryがengineと違う点は仕様どおり）。Mainにより新しいclosed PASSが得られたら、その正確なpath/SHAを明示選択する。retention時刻は引継ぎ作成時の24h後で、実行時に未来48h以内のUTCへ設定する。既存v1を上書きせず、必要なら新しいvNをMainが選ぶ。

```sh
/home/yayoi/.local/share/strata-sycl/venv/bin/python -B /home/yayoi/orca/workspaces/Strata/sycl/bench/results/2026-10-15-pr6-recovery-audit/own-vllm-unitrace-direct-v1/controller.py \
  --mode tiny --run-id v1 \
  --manifest-sha256 b23f81d1e3a02632158aca8036ed020e4744e13126f96cfe20f56db9032162e6 \
  --health-receipt /home/yayoi/orca/workspaces/Strata/sycl/bench/results/2026-10-15-pr6-recovery-audit/own-vllm-public-hook-v3.json \
  --health-sha256 1f2293b46be99f4922e7bd145e1922b530732da55f26d6ef75759c26f5c5d407 \
  --retention-review-utc 2026-10-17T04:37:55Z
```

Mainがtiny receiptのclosed PASS、fault0、oracle、actual PID/library、両表の時間行をレビューした後に限り、次を別に起動する。次のSHA計算は明示されたtiny v1だけを読む。自動latest探索や失敗の置換はしない。

```sh
/home/yayoi/.local/share/strata-sycl/venv/bin/python -B /home/yayoi/orca/workspaces/Strata/sycl/bench/results/2026-10-15-pr6-recovery-audit/own-vllm-unitrace-direct-v1/controller.py \
  --mode model --run-id v1 \
  --manifest-sha256 b23f81d1e3a02632158aca8036ed020e4744e13126f96cfe20f56db9032162e6 \
  --health-receipt /home/yayoi/orca/workspaces/Strata/sycl/bench/results/2026-10-15-pr6-recovery-audit/own-vllm-public-hook-v3.json \
  --health-sha256 1f2293b46be99f4922e7bd145e1922b530732da55f26d6ef75759c26f5c5d407 \
  --tiny-receipt /home/yayoi/orca/workspaces/Strata/sycl/bench/results/2026-10-15-pr6-recovery-audit/own-vllm-unitrace-direct-tiny-v1.json \
  --tiny-sha256 "$(/usr/bin/sha256sum /home/yayoi/orca/workspaces/Strata/sycl/bench/results/2026-10-15-pr6-recovery-audit/own-vllm-unitrace-direct-tiny-v1.json | /usr/bin/cut -d ' ' -f 1)" \
  --retention-review-utc 2026-10-17T04:37:55Z
```

残る確認はMainのソースレビュー、構文/host検査、tiny互換性、条件付き1回のmodel診断。過去buildのworktree生成 .gen は現在不在のため、公式generator template・現存header・qualified binary/build receiptを根拠にした範囲をprovenanceに明記した。未知の実ロードELFはFAILになり、poll間にload/unloadされたlibraryの完全捕捉は保証しない。R860の未完package、保護record、他engineの実装は読んでいない。

manifest SHA256: `b23f81d1e3a02632158aca8036ed020e4744e13126f96cfe20f56db9032162e6`。manifest内部はcontroller/provenanceを固定し、`SHA256SUMS` はmanifest・controller・provenance・この非実行review文書を封印する。
