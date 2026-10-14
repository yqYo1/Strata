# KV miss-copy workgroup 数の上限候補

2026-10-15。未適用、未build、CPU/GPU fixture未実行。書込は本directoryだけ。
Mainのexisting-exact actual GPU run中の製品source、binary、service、設定を変更していない。
Git、GPU、build、test、service操作、cleanupは行っていない。
プロジェクト専用 `strata-upstream-sync` のsource/caller確認と部品別評価の手順を適用した。
測定値、速度改善、7秒台の境界waitの原因、採用を認定する資料ではない。

`source.diff` が提案の実体。`files/` は適用先relative pathと同じ配置の候補source。
`pins.json` は読んだ製品sourceと先行報告のSHA256。
`generated.json` はdiffと候補sourceのhash、未実行status。
`regen.py` はpinが一致した製品sourceを読み、本directory内の候補とdiffだけを再生成する。
製品へ適用する機能は持たない。元sourceが進んだらpinを黙って更新せず、Mainが差分を再reviewする。

## 最小の製品変更

`src/kernels/sycl/kv_stream.cpp` のcopy kernelだけに別のlaunchを追加する。
`STRATA_SYCL_KV_COPY_GROUPS` がunset、空文字、`0`なら、従来の1 slot = 1 groupのkernelを選ぶ。
`128`, `256`, `512`, `1024` はopt-inの候補。その他の文字列はdevice仕事を発行する前に拒否する。
`nq <= 0` の既存early returnが先にあり、その場合は新しい設定も読まない。

候補は `G = min(n_slots, configured_cap)` groups、local size 128で発行する。
各groupは `k = group_id; k < ctl[2]; k += G` で複数のplaced pageを担当する。
`ctl[2]` をCPUへ読まず、page listとcopyの間にwait、event、queue、host taskを追加しない。
既存のresolver single_task、CLOCK、counter、overflow、flag reset、pool allocation、format、
`finish(stream,event)` とhost-memory所有者は変更しない。

32768 resident cells / page_size4 の8192 slotsなら、source上のlaunch形状は次の通り。
これは実行時間の測定ではない。

| 設定 | copy groups | workitems | placed8192時の各groupの担当page数 |
| --- | ---: | ---: | ---: |
| default0 | 8192 | 1048576 | 1 |
| 128 | 128 | 16384 | 64 |
| 256 | 256 | 32768 | 32 |
| 512 | 512 | 65536 | 16 |
| 1024 | 1024 | 131072 | 8 |

全hitでもcopy submission自体は残り、候補の全groupがdevice上の0件を見てcopyせず終了する。
小missの不要groupを減らす可能性がある一方、多missでは並列性を減らして遅くなる可能性がある。
R588の境界waitや`decode-format-service.ja.md`の7.418/7.479秒へ、この効果を帰属しない。

## 正しさの境界

resolverの実装が作る `P=ctl[2]` は `0 <= P <= n_slots`。
`0 <= k < P` に対し `g=k mod G` と `j=floor(k/G)` が一意なので、候補も全placed pageを一度ずつcopyする。
Gより大きいP、full slots、任意のvalid cell ID、非連続duplicateを切り捨てない。
CLOCKがplaced pageへ別々のslotを割り当て、`runs`/`spans` が全writable runを互いに、かつread runから
分離する既存条件を保つ。各page内のbyteは `i=lane+128*j` で分担し、追加barrierを必要とする共有領域はない。
コピー順の変更でformatの数値演算を変える案ではない。

任意IDではwidth個がwidth個の別pageを指せる。`ceil(width/page_size)` やselector固有の上限を
copyのlaunch数やcoverage証明へ使わない。headerの `nq*(cap/page_size+2)` の説明は汎用arbitrary-ID容量保証ではない。
必要なdistinct pageがslotsを越えると、既存通りctl[3]を立て、置けた有効なsubsetだけをcopyする。
overflowを消したり、そのselection全体がresidentになったと扱ったりしない。

`runs` のfmt0 FP16はK,Vの2run。fmt1 int8はK,V codeとK,V FP16 scaleの4run。
fmt2 Q4_0はscaleを含むpacked K,Vの2runをそのままcopyする。
real shapeのpageあたりpayloadは順に8192 / 4224 / 2304 B。
q4のHadamard回転とappendの量子化はこのcopyの外であり、候補はそこを変更しない。
K8V4 hybridは現sourceがstreamingを拒否し、`qsa_kv_format`もblock mover到達を拒否する。
候補がhybrid streamingを新規実装したとは扱わない。

`n_blocks`、`n_slots`、cap/nq、format/shapeの既存validationは維持する。
論理262144 cellsを小さくする分岐、host backing削減、prefix制限、末尾のpage切捨てはない。
末尾がpartial pageなら、その割当済みpageの全byteを従来通りcopyし、読者が選択したvalid cellだけを読む。
fixtureのpage_size4ではthreshold側とcontext末尾側の二つのpartial block、および262143を明示する。
これはmodelの物理256K全履歴実行やRESTORE完走の代替ではない。

## configured source resolver とcaller

root `CMakeLists.txt:186` はSYCL時に `cmake/sycl_backend.cmake` をincludeする。
同fileの `strata_kernels` は実際に `src/kernels/sycl/kv_stream.cpp` をcompileし、
`strata_engine` → `strata_prefill` → `strata` がそのlibraryを使う。CUDA root generatorの機能を根拠にしていない。

| caller | 適用条件と順序 |
| --- | --- |
| `src/core/layer.cpp:735` `qsa_kv_resolve` | `st.kv_mode==1` だけblock moverを呼ぶ共通wrapper。fmt/pools/host/mapを同じstateから渡す。mode0とmode2はreturn。 |
| `src/core/layer.cpp:954` 単token QSA | nq1、device `st.step` とselected IDs。selection後、同streamのdecode attention/gather前。 |
| `src/core/verify.cpp:754` verifier QSA | nq=`n` のselection batch。`tb`offsetのIDs/stepsを同csへ渡し、batch attention前に全queryのunionをresolve。通常MTP4のactual `n` の分布は本調査で測っていない。 |
| `tests/sycl/kv.cpp:106` | 既存小fixtureの直接呼出し。candidate group数未満の3 slotsだけなので、このtest単独ではstride処理を検証できない。 |
| 新component fixture | 実production entrypointへ直接発行、または同entrypointで記録したnon-native command graphをreplayする。component以外の推論経路を置き換えない。 |

ringの `kv_ring_table/kv_ring_restore`、promptの `kv_stage_from_host` はresolve copyではなく対象外。
append (`src/kernels/sycl/kv.cpp`) はhost USMへ常に書き、resident slotだけへ追加で書く既存動作を維持する。
rejectされたdraftを含むappend/commitの契約に変更はない。

`queue_for` はopaque streamをin-order SYCL queueとして検査する。
resolver task → copy → 後続attentionの依存は同じqueueの順序のまま。
nonnull streamではhostへ非同期returnし、map/selection/host/pool所有者は既存完了境界まで生存させる。
null streamは従来の `finish` がruntime event完了を待つ。
`kv_stream_counters` は同期debug accessorで、external queueの完了はcallerが先に保証する。
今回追加する唯一のhost読取は環境変数であり、USMのhost readやresetではない。
設定はgraph capture時に確定するので、起動前に固定する。既存graphは設定変更では変化しない。
componentのABBAだけは単一ownerで各arm完了後に環境変数を変え、同時inferenceを行わない。

## fixtureと実component測定の候補

CMake追加はEXCLUDE_FROM_ALLの二targetだけ。CTestへ自動登録しない。

- `kv_copy_groups_host_fixture`: SYCL include/runtimeなしの独立C++ fixture。Gの分割coverage、
  全miss0..513、境界127/128/129・255/256/257・511/512/513・1023/1024/1025・full8192、
  format別の全byte比較を行う。CPUの独立ordered-miss/CLOCK oracleは、page table、slot block、
  stamp/ref、16 ctl全て、miss listの未使用tailも含めて結果を定義する。
- `sycl_kv_copy_groups_bench --validate`: 実kernelのbaseline0と4候補を全state/byte oracleへ照合する。
  17slot/page4と1031slot/page1/64dimの小fixtureでmin(slots,cap)とmiss境界、1031full slots、duplicates、hit保護、eviction/second chance、
  bounded-list overflow、全slot保護時の置換不能、epoch -1/INT_MAX、無効hand、u64 carry/wrap、
  invalid IDs/width、既存overflow保持、width0とnq0を確認する。
  `--validate-exhaustive` はGPU側もmiss0..513を全列挙する。
- real shape/page4/65536 logical blocks/8192 slotsではFP16/int8/q4、任意513 pages、full8192、
  synthetic T4の各width2051・union2052 pages、二partial blockと262143を確認する。
  T4は明示したfixture形状であり、actual verifierのhit率やtransfer量を観測したものではない。
- production span validationのhost/destination alias、map出力alias、無効optionをenqueue前に拒否する
  fixture、unset/empty/zeroのlegacy選択、null-stream同期とdrain後のcounter accessorも含む。
- graph fixtureは各armを一度captureし、同じUSM addressのままIDs/width/ctlを変え、
  miss513 → hit0 → より広いselection → slots全幅をreplayする。envを変えても既存graphのgeometryは固定。
  input全byte、source全byte、全metadata/poolの前後guardを確認する。

後でMainがidle境界でreview/applyし、既存のpin済SYCL buildへ追加した場合のコマンド候補:

```sh
cmake --build "$MAIN_BUILD" --target kv_copy_groups_host_fixture sycl_kv_copy_groups_bench
"$MAIN_BUILD/kv_copy_groups_host_fixture"
"$MAIN_BUILD/sycl_kv_copy_groups_bench" --validate
"$MAIN_BUILD/sycl_kv_copy_groups_bench" --validate-exhaustive
"$MAIN_BUILD/sycl_kv_copy_groups_bench" --graph-benchmark 128 40
"$MAIN_BUILD/sycl_kv_copy_groups_bench" --graph-benchmark 256 40
"$MAIN_BUILD/sycl_kv_copy_groups_bench" --graph-benchmark 512 40
"$MAIN_BUILD/sycl_kv_copy_groups_bench" --graph-benchmark 1024 40
```

これらをこのproposal作成では実行していない。Mainはactive existing-exact runの終了を確認して直列化し、
現在のcompiler/runtime/driver・AOT・全flags・binary hash・device identityを固定する。
`decode-format-service-normal-v1.json` のfrozen env recipeを使用し、特に
`UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1`, `NEOReadDebugKeys=1`, `EnableDirectSubmission=0`,
`STRATA_SYCL_VERIFY_NATIVE_CAPTURE=0` を維持する。fixtureはこれら4項とGPU selectorを起動時に検査する。
warning/parameter-validationも同じrecipeを両armへ使う。profileだけ別のnormaldecode設定へ変えない。

component専用ownerは、各processのwall1800秒、RSS4GiB、stdout/stderr合計64MiBを候補上限とする。
通常validationを先に閉じ、exhaustive/benchmarkを一つずつ進める。予算超過はincompleteでありPASSにしない。
最大のfixtureはhostsource512MiB、device slots64MiB、初期値/期待値/読戻し等を含め1.5GiB未満の設計。
これはmodel run用の予算ではない。modelは既存の88GiB等のMain側制約を別途使う。
Mainのowned process/measurement lock、期限、fresh fault/dump/boot/終了・reap確認を既存手順で付ける。
fixture単独stdoutのPASSでGPU faultなしやowner正常終了を証明したとは扱わない。

## timingと保存

`--graph-benchmark G 40` はcapture済みの実resolve2node graphを同じqueueでABBA反復する。
各case/formatの40標本はA0,BG,BG,A0を10回、各arm20標本。
40..80の4倍数だけ受け付ける。init/copy/capture/warmup/oracleは計時外。
毎標本、同じ初期mapとpoolを復元するので、前sampleのmissが次sampleのhitに変わることを防ぐ。
これはmap上のcold/missing状態であり、CPU cacheやPCIe transactionのcoldを保証する用語ではない。

`--benchmark G 40` は同じentrypointの直接host発行比較。
`--graph-event-benchmark` / `--event-benchmark` は別processの診断として前後marker eventを足す。
計測列は次のように別scopeで保存する。

| 列 | 範囲 |
| --- | --- |
| `host_operation_submit_ns` | directならproduction呼出しのhost wall（validation/env parse/2 submissions）、graphならgraph submissionのhost wall。完了時間ではない。 |
| `host_marker_submit_ns` | event診断だけの前後marker発行host wall。cleanではnull。 |
| `host_whole_queue_ns` | そのoperationの発行開始から同queueのchecked完了まで。event診断ではmarkerも含む。CPU/device/runtime待ちを含む。 |
| `device_marker_envelope_ns` | 前markerのdevice endから後markerのdevice startまで。resolver + copy + queue gapでありcopy単独serviceではない。cleanではnull。 |

host clockとdevice timestampのabsolute時刻をjoin/加算せず、markerの値をclean timeと混ぜない。
copy単独eventは現void APIから取得していない。さらに必要ならMainのbounded collectorでactual2nodeを
識別する別診断とし、この候補に新しいproduct event readを足さない。

全標本をJSONLで出す。phase、direct/graph、format、nblocks/nslots/page/heads/dim、nq/cap/各width、
configured cap、source上のlaunched groups/workitems、実際に読戻してoracleと一致したplaced、
logical host-read payload bytes、既存counter、overflow、state/payload fingerprint、全timingを保持する。
`launched_copy_groups` はpin済sourceと設定から算出した形状で、profilerのworkgroup実測ではない。
`placed_pages` はdevice ctlのreadbackがoracleと一致した値。

payloadは `placed * block_bytes` の論理host-read byteであり、DMA量やPCIe wire bytesではない。
全hitのpayloadは0。`bandwidth_GBps` は全caseでnullにし、0 payloadの時間から帯域を作らない。
overflowのpartial copyはcorrectness caseとして保存するがbenchmarkはoverflowを拒否する。
`lookups` は各queryのpage runを数える実source契約（非連続duplicateは再計上）。
unique union/miss数とは違う。`lookups-misses` をhit数へ変換しない。

compact receiptには全個別標本、command/environment、入力case identity、source/diff/binary/toolchain、
correctness/fault/owner status、中央値・範囲とABBA組単位の差を残す。
同条件のcase/phase/formatだけを比較し、allhit改善をfullslots退行と平均して採用しない。
容量、全KV format、同一stream lifetimeを保持し、少なくとも小missとfullslots双方を見てから次段を決める。
profile traceを取るなら64MiB内で有用なnode/call/queue要約とhashを抽出し、rawを退役する判断はMainが行う。
今回はretire/cleanupするartifactはない。

## 残る未検証点

C++/CMakeのcompile、独立CPU fixture、SYCL event/graph APIの実toolchain互換性、GPU byte/state parity、
fault/owner、性能は全て未実行。source読取による論理説明と候補コードだけである。
legacy defaultにも新たなhost環境変数parseがある。直接呼出しのoverheadと未変更binaryのdefault退行は別途確認する。
graph replayではparseはcapture時だけだが、hostwallが同様に改善するとは先に決めない。

componentで改善した場合も、同条件のnormal MTP4 actual decode、各formatのstate/counter・出力、
clean公式bench.shのprefill/decode/agent、物理256KとRESTOREはMainの後続gateである。
本proposalはそれらを完了、model速度改善、品質同等、7秒waitの解消、採用済みとは報告しない。
