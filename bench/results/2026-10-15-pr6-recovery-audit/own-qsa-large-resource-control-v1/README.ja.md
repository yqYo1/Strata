# QSA 8192 query の同値資源比較: source-only 提案

状態は **SOURCE_ONLY_UNAPPLIED_UNCOMPILED_UNTESTED**。現行 source、CMake、既存 report は変更していない。
新規物はこの directory だけ。Main の測定と並行したビルド、preprocess、テスト、native/runtime/GPU、
profiler、process inspection、Git、cleanup、追加 agent は実行していない。推論最適化の採用や性能値の追加ではない。

依頼の source base は `3035e7e8b7417acadb80ddd7592f53a94b0d3625`。Git が禁止のため、その commit の
membership は独立確認していない。代わりに実際に読んだ 444 行の source 全体と comparator の exact preimage を
`base/` に保存し、SHA256 を `provenance.json` に固定した。Main の poison と三つの inspection gate が入った
現行 source hash は `3a1fb967eb0cf2241b3eba4f8d58b4d8c2671b5b4a6969de62a64a133d584edc`。
patch はこの保存済み preimage から生成した。Main は適用前に live preimage と hash を照合し、異なる場合は
完全差分を再審査する。full file を stale mirror から現行 tree に上書きしない。

成果物は次のとおり。

- `files/tests/sycl/qsa_resource_control.cpp`: full patched source。
- `base/tests/sycl/qsa_resource_control.cpp`: exact C++ preimage。
- `source.patch`: 上記 preimage に対する unified diff。対象は既存 `tests/sycl/qsa_resource_control.cpp` だけ。
- `compare_frames.py`: 独立した streaming comparator 提案。実行していない。
- `base/compare_frames.py` / `comparator.patch`: 前回 comparator と、それを基準にした完全差分。
  patch の相対名は `compare_frames.py`。既存 report directory への上書きを指示するものではない。
- `sizes.json`: shape ごとの全 buffer の element 数、payload、guard 込み device bytes、host element storage。
- `provenance.json` / `manifest.json` / `SHA256SUMS`: 出自、実際の read snapshot、出力 hash と未実行 gate。

## 比較の目的と入力

前回の 128 query では、shared/whole 11.692072 ms、replicated/whole 11.909869 ms、shared/grain8
30.700135 ms だった。これは Main の既存独立 3 process/control 結果で、新たな実測ではない。
詳しくは隣の `hardware-resource-followup-v7.ja.md` と `own-qsa-resource-comparison-v1.json`。
小さい grid を 16 回に分ける影響と、実運用でも既に使う 8192 query batch の資源感度を分けて問う。
同じ幅の人工入力を使う部品診断であり、32K 以上のモデル PP の代用にはしない。

| shape | query 数 | physical replica 数 | 許可する bench grain |
| --- | ---: | ---: | --- |
| 既存 small smoke | 3 | 3 | bench ではない。内部 1 / 2 / 3 を維持 |
| 既存 bench | 128 | 128 | 1 / 8 / 32 / whole |
| 既存 bench | 256 | 256 | 1 / 8 / 32 / whole |
| 追加 bench | 8192 | 128 | 512 / whole |

引数の順番、repeats 1..20、warmups 2..5 は同じ。8192 の grain 1、8、32、0、負数、その他の値は
runtime 作成前に拒否する。numeric 8192 も従来の numeric nq と同様に whole の別表記として受理する。
128/256 に grain512 は追加しない。

bench は全 query で cap/width2051、cells2113、Page4、padded_cells2116、24 query heads、2 KV heads、
Dim256。既存の ordered ID `c=(j*17)%2113` と query `float(sin(i*.019))` はそのまま。
replicated ID だけを `(b % nreplica)*2116+c` とする。shared ID は replica0 の `c`。
8192 query では各 physical replica を 64 query が選ぶ。query 値を replica ごとに複製する変更ではない。
旧 128/256/smoke は `b % nreplica == b` なので、既存の入力 byte を維持する設計。

128 replica 全体の page table は 67712 entries。各 replica の reverse-page mapping を全 entry 初期化し、
proof で全 entry と actual page extent を確認する。K/V と scale も全 physical row、padding、非選択 row を
初期化し、replica0 との全 byte 比較を追加した。従来の全選択 ID ごとの K/V/scale byte 比較、欠落 page の
同値性、actual physical row bounds の検証も残した。query 数と replica 数の混同を避け、bench の unique
replicated rows は `shared_unique_rows*nreplica` を要求する。8192 でも unique address cohort は 128 倍であり、
8192 倍とは主張しない。

両 address arm に全 replica、両 ID 配列、全 query、同じ出力を確保・初期化・転送する。
入力 byte、math、全 allocation は等しく、kernel に渡す ID 配列の選択だけが address arm の違い。
grain 比較では同じ query cohort を kernel command に分ける。payload scrambling、precision/layout/kernel の変更はない。

## kernel の実際の仕事量

読んだ `src/kernels/sycl/prompt_attention.cpp` の FP32 `launch` は WG=128、
`nd_range(queries*shapes.n_head_kv*WG, WG)`。group index を query と KV head に分ける。
従って 24 query heads を group 数として掛けず、query あたり 2 group。

| query cohort / grain | 全 workgroup | kernel calls/work | workgroups/call |
| --- | ---: | ---: | ---: |
| 128 / whole | 256 | 1 | 256 |
| 128 / 8 | 256 | 16 | 16 |
| 8192 / whole | 16384 | 1 | 16384 |
| 8192 / 512 | 16384 | 16 | 1024 |

8192 の C と A は総 query、width、総 group 数が同じ。小さい旧 grain8 の 16 groups/call と、追加 grain512 の
1024 groups/call を区別する。8192/512 では毎回 128 replica を 4 回ずつ選ぶため、replica cohort を一部だけ
残す分割にもならない。各 query の蓄積順序は変えない。

kernel は steps の width のみを使い、ID を page table から actual row に変換する。pos/NKv を ID の範囲制限、
RoPE、bias に使うという仮定は置かない。今回の host proof は old FP32 API が全 pool extent を自動検証する保証に
依存しない。kernel/API の production code は変更しない。

pre-exec の固定値は前回と同じ。

```text
STRATA_SYCL_PROMPT_DIRECT=1
STRATA_SYCL_PROMPT_K_CELL_MAJOR=0
STRATA_SYCL_PROMPT_NATIVE_KV_MIN=0
STRATA_SYCL_PROMPT_SOFTMAX_SUBGROUP=0
STRATA_TEST_MAIN_GATES=1  # Main の監督時
```

pools の K8/V8 により Mode1、Direct1、Nativefalse、KMapping0、Softmaxfalse。process 内で環境変数を変更しない。
Main は既に qualified な selector/provider/library/compiler/driver/その他の環境を child exec 前に固定し、その実効値を
別 receipt に保存する。この source だけで runtime を新たに qualified と扱わない。

## 正確な buffer 容量

以下は source-derived な要求 byte 数。実測 RSS、driver 内部 allocation、physical traffic ではない。
全 device buffer は前後 16 elements の guard を含む。全 10 buffers の guard 合計は 960 bytes。

| buffer | 8192 host Frame payload bytes | 8192 device allocation bytes |
| --- | ---: | ---: |
| K int8 | 138674176 | 138674208 |
| V int8 | 138674176 | 138674208 |
| K scales uint16 | 4333568 | 4333632 |
| V scales uint16 | 4333568 | 4333632 |
| table int32 | 270848 | 270976 |
| shared IDs int32 | 67207168 | 67207296 |
| replicated IDs int32 | 67207168 | 67207296 |
| steps int32 | 131072 | 131200 |
| query FP32 | 201326592 | 201326720 |
| output FP32 | Frame にはない | 201326720 |
| 合計 | 622158336 | 823485888 |

8192 の physical pages は 67712、rows は 541696。shared の異なる選択 row は 4102、replicated は 525056。
K/V+scale の unique selected payload は 2165856 / 277229568 bytes。これは選択 address の論理勘定。
8192 の reuse は shared8192、replicated64。physical DRAM bytes の計測値ではない。

| query shape | 全 device allocation bytes | Frame host element bytes | host vector element storage の最大同時量 | dump bytes |
| --- | ---: | ---: | ---: | ---: |
| smoke3 | 403752 | 329064 | 599688 | 0 |
| 128 | 294681024 | 291534336 | 436500256 | 3145752 |
| 256 | 589361088 | 583068672 | 873000224 | 6291480 |
| 8192 | 823485888 | 622158336 | 1226138496 | 201326616 |

host の最大同時 element storage は `Frame + 2*guarded_output + max(全guarded buffer)`。
reference/initial を保持したまま、final または全 input readback を 1 本ずつ持つ時を数える。
128/256 は K/V readback が output より大きい。8192 は 3 本目の guarded output が最大。
FP64 oracle の weight 配列は最大 16408 bytes だが、initial を作る前なので最大同時量を増やさない。
proof の vector<bool> は proof 内だけで破棄される。標準 library に依存するその実 capacity、vector の制御構造、
allocator metadata/丸め、stack、SYCL/driver/library の host storage は表の外。実 RSS の正確値とは呼ばない。

size_t の storage element products、guard 加算と byte products、allocation 合計、dump byte 計算を checked にした。
固定 shape 内の int index は最大 query index 8191、flattened Q index 50331647 で INT32 内。ID/page/row は
実 backing extent で別に検証する。1GiB の device budget は従来通り必要条件として維持した。

## validation、dump、comparator

全 small 3 cases、負 ID、欠落 page、明示 duplicate、不正幅、全 small FP64 oracle、2 address arms×3 grain の
18 full-output 比較、比較前 poison、initial/final 全有限、全 output byte、全入力不変と guard、Drains、
fail-stop、Main の三 gate は残した。大きい bench の FP64 oracle は query0/1/2/last のみ、全8192queryではない。
timed repeats は final frame だけを検査する従来仕様。kernel 内の新しい検査や test weakening はない。

明示的な host allocation、copy、CPU oracle、proof、dump、logging は timed enqueue/drain window の外。
inside window は既存 entrypoint への submission と wait/clock。既存 runtime/driver の内部処理の無 allocation を
保証するという意味ではない。thread CPU と host wall を分ける既存計時を保ち、wait CPU を host math と数えない。

QRCV1LE の 24-byte header と FP32 little-endian payload は変更しない。8192 は 201326616 bytes、
192MiB+24。上限は 193MiB = 202375168 bytes。旧 7MiB 予算を全 shape で無制限に広げず、comparator 側は
nq128/256/8192、cap2051、count=nq*24*256、正確な file extent を要求する。

修正した safety gap は既存 dump の `trunc` overwrite。新案は `O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC`、mode0600。
既存 path、symlink、directory、open/write/close の失敗は FAIL。短い write と EINTR を扱い、成功する最後の close も
検査する。Linux close error の再試行はしない。失敗 partial file を worker/helper が消す処理は加えていない。
fd close は fsync/crash durability の保証ではなく、Main の post-run 読み直しと hash/byte 比較が必要。

comparator は canonical と candidate を regular file の open handle で読む。最終 symlink、同一 inode の自己比較、
不正 header/extent、途中の EOF、末尾追加、読み取り中に変わった stat identity を拒否する。1MiB の bounded reads、
short-read loop、全 FP32 exponent の Inf/NaN 検査、全 payload byte 比較、全 frame SHA256、最初の異なる byte offset を
記録する。byte mismatch 後も両 frame の最後まで有限/byte/hash を検査する。hash 一致だけの PASS はない。
一度に full frame を buffer しない。最大 3 個の 1MiB read buffer が一時的に重なる設計で、Python interpreter や
receipt 等の overhead は別。複数 candidate 指定時は canonical を再読し hash を再確認する。

comparator の PASS は full initial frames の一致だけ。process の成功、後続 repeat、GPU fault、安全な所有終了は
native receipt で別に確認する。自己比較を拒否しても、異なる file が別 process 由来である保証にはならない。

## Main の残り gate と保持

1. 完全 patch と preimage/provenance、既存 feature 保持を審査し、Main が実 tree へ統合する。
   CMake の既存 `sycl_qsa_resource_control` EXCLUDE_FROM_ALL target が使えるため、CMake 変更は不要。
   qualified compiler/flags/link/runtime/ELF/hash を改めて保存する。
2. Main の所有 lock と有限 deadline の下で、まず従来 smoke を trace+parameter-validation 付きで再確認する。
   三 cases の kernel 提出は従来 39。必要な small128 と旧256も新 build の回帰確認に使い、8192 へ進む。
3. 8192 の最初の shared/whole を repeats1/warmups2 で監督し、全 allocation/input/guard、sampled FP64、full frame、
   native exit0/所有終了、新規 fault0 を確認する。次に replicated/whole、shared/512 の first checks。
   2×2 の interaction を問うなら replicated/512 も同じ gate。最初は clean 性能値として扱わない。
4. first8192 の全 cross-arm/grain frame を canonical と逐次 comparator で全有限/全 byte 比較する。
   repeats1/warmups2 の提出数は whole5、grain512では65。一般に `1+(1+warmups+repeats)*calls_per_work`。
5. 全 first gates 後、instrumentation を外した別 child processes で各 control を独立3回以上実行する。
   最小 A=shared/whole、B=replicated/whole、C=shared/512。D=replicated/512 は interaction 用の追加 control。
   2warmup/10repeatなら whole14、grain512では209提出/process。process 内 repeat を独立 n と数えない。
   順番を balance し全個別 wall/enqueue/wait/thread-CPU samples、実効 pre-exec 環境、PID/start/boot/exit、fault を保存する。
6. kernel control の時刻だけで特定 unit の帯域、占有率、XMX activity、barrier wait を断定しない。
   この診断自体は engine adoption candidate ではない。別の実装候補が生じた時に公式長入力 bench、品質/state、
   physical262144、RESTORE と short-input/decode 保持を別途 gate にする。

small first-check の環境は既存 qualified recipe から exec 前に固定する。

```text
UR_ENABLE_LAYERS=UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION
UR_LOG_TRACING=level:info;flush:info;output:stderr
```

8192 trace も有限 deadline と raw text 64MiB 上限を Main が先に割り当てる。上限/timeout 到達は不完了、pass ではない。
8192 は旧128の単純な64倍で完了すると仮定せず、host full proof/oracle と実 device/wait の進捗を別に記録する。
成功 trace の要点抽出・検証後は全履歴を保持しない。失敗は distinct mechanism の代表診断を保持する。

raw frame の owner/consumer は Main と本 directory の comparator。現在比較する shape ごとに canonical1本とcandidate1本のみ。
8192 の二本合計は 402653232 bytes、予算386MiB。候補の native/process/fault receipt と full比較/hash を検証した後、
run 終了と所有不在を確認し、path/size/hash/reason を deletion manifest に記録して候補を退役させてから次の process。
canonical の next review point は全 cross-process cohort の比較と判断の完了。unresolved numerical failure では必要な一例を
残して再審査する。多数の 192MiB frame を同時蓄積しない。この worker は capture も deletion も作っていない。

未実行 edge cases: CLI不正値全組合せ、close/write/ENOSPC/EINTR failure injection、existing path/symlink、
truncated/oversize/mixed-shape/mutating/nonfinite/1byte mismatch frame、same-inode rejection、32bit host、異 endian。
code は source reasoning のみ。comparator 自身の qualification も Main に残る。

出自は現行 own MIT fixture、現行 own kernel/runtime API、今回の original host control/comparator。
前提の `LICENSE` と既存 own provenance/source reviews を読んだ。project-only upstream-sync skill は出自と
scope 制限の参照に用い、upstream check/fetch/移植は行っていない。XeStrata の実装、helper、test、archive、
外部実装の copy/adaptation はない。research-index の knowledge report body は読まず、前回測定の必要な文脈だけ使った。
全成果物の実 hash は `SHA256SUMS`、source/compare/patch の postimage hash と未実行状況は `manifest.json` にある。
