R856 / own-qsa-device-events-v1 は source-only 提案です。**UNRUN**。product、既存 report/controller、Git、build product、runtime/service/driver を変更していません。ビルド、テスト、GPU/model/profiler 実行、active process 調査、agent の起動・誘導はしていません。Main がレビュー、適用、ビルド、検証を行います。

成果物は `proposal.patch`、5 個の `proposed/` ファイル、`base-hashes.json`、`candidate-hashes.json`、`provenance.json`、`qualification-plan.json` と本書だけです。patch は frozen `6354f0b4ff3731466795410e11ca9ef785c9d326` として指定された現在の4ファイルを読み、下記の完全 SHA256 snapshot を基点に生成しました。Git による commit membership 照合はしていません。Main は適用直前にこの snapshot と commit を照合し、`git apply --check` を行ってください。report-only HEAD 更新は source の同一性を変えません。

| 既存の対象ファイル | 元ファイルの SHA256 |
| --- | --- |
| `include/strata/kernels/qsa_prompt_attn.hpp` | `5c7f5488dec2f93a9d4acbbb077b431fbabefd6c99cc84f688b388719dd5be1d` |
| `src/kernels/sycl/prompt_attention.cpp` | `0d1b0be4282b4535749bbda6436d0be923d0bdafc06ae6115a0bc1688f2bd500` |
| `src/prefill/prefill.cpp` | `d01662ae6c46bfdd2556da1345c5f9edb79f91694391eee2b2b0aebfeab831ce` |
| `src/prefill/layer_major.inc` | `e2def3bbd248c06c25ff2fce7ebc0ce49aecfcb0e418115196da8bb4b99960b1` |

追加する opt-in は `STRATA_PREFILL_QSA_DEVICE_PROFILE=1` です。既存の `STRATA_PREFILL_LAYER_MAJOR=1`、`STRATA_PREFILL_LAYER_MAJOR_PROFILE=1`、`STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE=1` を必要とします。短い warmup は従来経路で動き、device sample を出しません。long layer-major の観測対象は、明示的な既存 in-order/profiling stream、48 layer/12 QSA、resident/unrotated INT8 KV、host mirror なし、prompt matrix attention です。`STRATA_PROMPT_ATTN_OLD` は値0を含め存在したら拒否します。streaming、Q4、hybrid、rotated KV、別 queue、profiling property/support なしを診断対象に広げません。

通常 OFF では collector、event 保存領域、clock 読出し、profiling property 問合せを作りません。既存の公開 FP32/BF16 attention signature と symbol は維持し、診断用 overload に null を渡す wrapper にします。既存 callsite はそのまま使えます。診断時だけ追加引数で caller-owned `Submission` を渡します。新しい global/TLS owner、owner map、Runtime、context、queue、provider、CPU 推論 fallback はありません。queue 選択、submit の個数・順序、既存 `finish(stream,event)`、async error/checked drain、memcpy は維持します。OFF の host 命令列や binary identity が旧製品と同一、という主張はしません。

`prompt_attention.cpp` の2箇所で、元の `queue.submit()` が返す実イベントを保存します。新しい marker/barrier/event submit は追加しません。CGF 本体（local_accessor 宣言、nd_range、kernel lambda 全体）は静的なテキスト比較で byte-identical です。

| CGF 本体 | bytes | 元と提案の共通 SHA256 |
| --- | ---: | --- |
| FP32 template | 11028 | `bcc030203b983f9f2eb250cc7ba93ee9c5cdda319f549170aec5dc1d63d6f8d0` |
| typed BF16 | 10156 | `fa8edd7ebaea968ccaf02ae03810217342452d3ea46516ef5d2d48a38b4c5dbf` |

この比較は compile/test ではありません。scale、barrier、local memory、geometry、accumulation、BF16 rounding、query/KV/output の storage は変更していません。`layer_major.inc` の既存「QSA/GDN internal tensors remain FP32」という stale banner も本件では変更していません。observer を dtype/mathematics の変更と混同しないでください。

所有権と退役は次の有限な経路です。

1. 最初の request work より前に、`ceil(n/chunk)<=1024`、context/position、route、matrix capability、`device.has(aspect::queue_profiling)`、既存 queue の `enable_profiling` / `in_order` を検査します。FP32 capability helper は full/tail shape と Mapping2 の既存 SLM gate も検査し、submission はしません。typed 側は既存の強い allocation/geometry validation を継続します。
2. `Profile` を Restore / checked lease より前に所有します。`unique_ptr<Submission[]>` を要求全件分確保してから GPU work に進みます。空の `optional<sycl::event>` なので slot 初期化で default event を大量生成しません。host admission は既存 owner を累積し、slot の実サイズと backend event あたり4096 bytesの明示的な見積 allowance を加えます。backend 内部 allocation のサイズは SYCL が保証しないため、この allowance を厳密な runtime memory 上限とは呼びません。
3. attention の直前に allocation-free な `reserve()` で layer/chunk/絶対 token start/count、logical key end、cap、route を確定し、retired flag を false にします。reserve 後の input validation や部分的 submit が throw しても、保守的に drain が必要な状態のままです。slot に vector push/resize はありません。
4. `ApiSpan` は公開診断 attention 関数の validation/dispatch を含む host entry/return、`SubmitSpan` は内部 `queue.submit` の host entry/return を別々に保存します。既存 `cpu_wait_profile::stamp()` の issuing-thread CPU clock を再使用します。CPU clock が利用できない場合は個別値を null とし、集計に CPU sample count を付けます。host API span は submit span を内包するので両者を加算して独立時間と見なしてはいけません。
5. `queue.submit` 成功後にその返却 event を事前確保済み optional に保持します。event copy が implementation 内で throw しても、slot/queue/source/output owner は outer lease の内側で破棄しません。throw は再送出し、host span を失敗として閉じます。submit 自体が event を返さず部分的に throw する場合も同じ outer checked drain が必要です。event retention のために新しい待機や reset を入れません。
6. 既存 wrapper `drain()` の compute/copy が両方成功した時だけ `checked_drain()` が retired flag を立てます。これは host flag 更新だけです。run_impl の callback drain や overlay retirement の順序を変えず、要求全体の event を terminal drain / `lease.retire()` まで保持します。event の status/profiling query は terminal 成功 seam の `finish()` にだけあります。
7. 最終の inference path、既存 coarse/QSA observer、件数がすべて成功してから、complete status、同一 backend、submit<=start<=end、start の順序を検査し、submit/start/end を取得します。per-event wait / barrier / 新しい stream wait / global wait はありません。取得・検査・集計が throw した場合は完成 event row を一つも出さず request を失敗扱いにします。
8. 正常時は `lease -> Restore -> device borrow -> device owner` の破棄順序で終了します。途中 return/throw でも既存 lease が checked drain を行い、成功時だけ owner を解放します。既存 fatal drain は owner が live のまま abort します。Profile destructor も未退役のままの解放を abort する防御を持ちます。失敗 destructor は event query をせず、complete=false / completed_samples_emitted=0 の1行だけを出します。

イベント上限は要求全体で **12288 = 12*1024** です。262144 context で chunk8192 は384件、chunk256 は12288件です。chunk255 以下で上限を超える要求は GPU work 前に拒否します。小さい context/chunk は同じ `ceil(n/chunk)` 判定で扱い、例として n1024/chunk1 は12288件、n1025/chunk1 は拒否です。上限を越えて一部だけ保存したり、「代表 sample」として省略したりしません。12 layer ごとに full/tail query shape の最大24 aggregate で、process environment が exec 前から不変という契約の下では同じ layer/shape の branch tuple は不変です。未知の追加 group、重複/欠落 chunk、route 変更は明示的失敗です。

出力は `strata qsa device: ` を prefix とする JSONL です。成功時のみ event rows、route/layer/query-shape/kernel ごとの aggregate rows、terminal request row を出します。request ID は host steady-clock nanoseconds によるローカル ID、queue ID は request_compute_0、実際の queue address と backend enum は request row に保持します。event ごとに family/mode/direct/native_kv/mapping も保存します。これは exact JIT binary/resource identity や完全な dependency DAG の代わりにはなりません。Main の receipt に実 source diff、binary/runtime/driver/device の identity を別途保存してください。

`event_duration_ns=end-start` と `submit_to_start_ns=start-submit`、host API/submit wall/CPU は別の量です。submit->start は queue backlog、dependency、runtime/device dispatch 等を含み、host CPU 時間ではありません。event duration は exclusive GPU service/critical path/occupancy を証明しません。request の nonexclusive duration sum と attention interval union を区別します。union は全 event の backend 一致と start 順序を検査した後、SYCL の同一 backend timebase 契約を使って計算します。host steady clock と device timestamp の生値は差し引きません。

`request_wall_ms` は既存 coarse の layer-major `started` から checked lease retirement 直後までの wrapper wall です。preflight、最終 observer query/format/emit、Restore/destructor teardown、API response 完了は含めません。完全な API request latency と取り違えないでください。個別 host API spans も runtime の内部 work/implicit waits と時計読出し cost を含み得ます。existing CPU-wait v3 は必要に応じて別途取得し、explicit wait の内外と pure computation を混同しません。

最大12288 event rows、24 aggregate rows、terminal row1件に限定し、追加テキストは要求あたり16 MiB未満を設計 budget とします。tensor dump、反復 API/poll ログは出しません。Main は全 diagnostic job の64 MiB raw budget を監督し、truncation、件数不一致、terminal row欠落を incomplete としてください。短い warmup/empty/no-QSA/失敗要求を successful completed sample として数えてはいけません。`normal_retirement=true` は checked drain の意味だけです。`numerical_correctness=not_checked_by_observer` のままにし、独立 oracle/比較の結果は Main が別 receipt に記録します。

Main の実行順序・flags・command data・停止条件は `qualification-plan.json` にあります。新しい private build root は `/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-r856-qsa-device-events-v1/build` とし、既存6354 PASS製品を保存します。順序は capacity/configuration の有限 host gate、元FP32/typedBF16 の短い traced attention と独立 oracle/event寿命 gate、selected modelの全48 layer mechanics gate、その後 official39K の各 route1要求の diagnostic pair です。clean speed は既存9 sampleと別に扱い、新診断から speed改善を主張しません。262144実モデル/RESTORE/general-qualityは本件でも UNRUN です。

直接 kernel fixture は prefill flag だけでは instrument されません。Main は private fixture variant に明示的な Submission owner/診断 overload を渡し、既存 checked drain 後にだけ event を読む有限 adapter を用意してください。数値 oracle を変更せず、元の fixture source と追加 adapter の provenance を pin します。`tests/sycl/decode_attention.cpp` は runtime 初期化後に setenv を切り替える legacy fixture なので、そのまま再利用しない計画です。`prompt_native16.cpp` / `prompt_k_cellmajor.cpp` の immutable-environment oracle variant を使用し、route は別 process に固定します。この fixture adapter、実 build/test、fault/normal-exit receipt は Main の未実施作業です。

新 helper は本 task で authored、repository MIT `LICENSE` に従います。Strata の既存 helper の契約と [Khronos event reference](https://github.khronos.org/SYCL_Reference/iface/event.html)、[queue reference](https://github.khronos.org/SYCL_Reference/iface/queue.html)、[device reference](https://github.khronos.org/SYCL_Reference/iface/device.html) の API semantics だけを使いました。外部 example/template/kernel/header を copy/adapt していません。XeStrata の code/helper/fixture/過去 proposal は参照していません。R847 は指定の行だけ、R848 は timing distinction/correction を読む範囲に限定し、knowledge-collection report body は開いていません。候補・入力の完全 hash と provenance は各 JSON を参照してください。

現時点の判定は **source proposal ready for Main review / 全 execution UNRUN / 性能主張なし** です。現在の33.9610s等は従来の非排他 queue window のままで、新しい device service 測定や bottleneck 結論はまだありません。
