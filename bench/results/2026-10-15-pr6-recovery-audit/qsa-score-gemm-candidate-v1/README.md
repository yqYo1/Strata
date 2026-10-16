# QSA prefill score GEMM candidate v1 — 未適用・未ビルド・未実行

2026-10-15 JST。Main が実行を直列管理するための実装 proposal。製品 tree、Git、branch、build、test、GPU、profiler、service、cleanup は変更・実行していない。このディレクトリだけに README、最小 patch と regenerator、その追加ソース、source pins、独立 CPU fixture を置く。project-only `strata-upstream-sync` と `ARTIFACT_RETENTION.md` を適用した。品質・速度・採用はいずれも未評価。

候補由来は R589 の形状分析。R563 の MoE descriptor、R564/568 の prompt attention、R581 の既存発散、R583 の oneMKL 内部 dispatch 仮説からの実装 import ではない。実装は現 fork の prefill 専用の新規 code。既存 `qsa_block_scores_tc()` は SYCL では常時 false のまま、prefill の指定 callsite だけが候補を選ぶ。decode caller、quant、GDN、MoE、既存 Gemm、default math、attention kernel は変更しない。

## 設定と patch 境界

`STRATA_SYCL_PREFILL_QSA_SCORE_GEMM` を**プロセス起動前**に `fp16` または `bf16` に設定する。unset が OFF。空文字、`off`、`0`、大文字、前後空白、部分一致などは明示エラー。最初の startup `bytes_needed/init` 時の thread-safe static に固定し、その後 getenv を読み直さない。enabled と既存 `STRATA_SELECT_OLD` の同時指定はエラー。

OFF では candidate scratch allocation、convert/GEMM/post submit、追加 wait が全てゼロ。存在するのは小さい non-owning host view と frozen enum。既存 scorer→top-k のコードをそのまま通る。enabled の unsupported geometry / null stream / insufficient borrowed arena はエラーにし、黙って別 precision へ切り替えない。

変更ファイルは `prefill.cpp` の include、Impl view、QSA count/carve、init/sizing admission、selection prepare/dispatch/invalidate と `cmake/sycl_backend.cmake` の source/manual target 登録、新規専用 header/source/component のみ。`layer_major.inc` の変更は不要で、その既存 `run_impl` 呼出しから同じ prefill dispatch を通る。

`pins.json` は生成時に読んだ live source の SHA256、各 anchor、このartifactの追加source/fixture/README/regeneratorのSHA256を記録する。`prefill.cpp/layer_major.inc` は Main の phase patch で変わり得る。`patch/regenerate_patch.py` は現在の source を読み、各局所 anchor がちょうど一つである場合だけ unified patch をこの artifact に書く。whole-file overlay は作らず、製品へ apply しない。anchor mismatch は停止するので Main が当該箇所を再確認する。無関係な phase 診断の追加は regeneration で保持する。

Main のみが行う最初の手順（実行していない）:

```sh
python3 bench/results/2026-10-15-pr6-recovery-audit/qsa-score-gemm-candidate-v1/patch/regenerate_patch.py
# Main: inspect pins + localized patch against finished phase-audit source,
# then apply through the authorized normal source-change workflow.
```

## 行列と因果/tail 契約

FP32 query は token stride 512、head offset `h*128`。FP32 pooled は block stride 128。各 batch は 1..256 query。各 head の row-major GEMM は `A[nq,128] * B[live+1,128]^T -> C[nq,live+1]`、`lda=512, ldb=128, ldc=max_blocks+1`、FP16/BF16 input・FP32 alpha/beta/output。既存 `Gemm::f16/bf16` の K 固定 lda に通さない。score 専用の明示 strided product を新 source のみに置く。

`live=floor(last_n_in_chunk/4)`。完成済み pooled rows `[0,live)` と dead 1 row を converted key に詰める。dead の列は `live`。`pooled[live]` 以上は読まない。最後の query が最大 logical 262144 の場合、live=65536、GEMM N=65537、score の物理 row stride は既存どおり65538。reserved GEMM row stride は65539。maxlogical と physical residency は別条件。

各 batch の raw query を一度変換し、head=0,1,2,3 の GEMM と post を**同じ順序**で enqueue。各 head ごとに `total += dot > 0 ? dot : 0`。最後の head のみ tail の `n%4 != 0` に既存の `+1e9f` を加算する。`ReLU(sum(head))` へ変更しない。

query 自身の `nb=n/4` ごとに `b<nb` は pooled column b、`b==nb` は dead column live を読む。`b>nb` は既存 scorer と同様書かない。`n%4==0` の dead score 自体は算出してよいが既存 top-k の weight は0、partial tail は1/2/3。future scores が未書き込みでも top-k は `nb+1` までしか読まない。元の radix top-k、同点は cell ID の小さい順、出力 ID は昇順という契約をそのまま使う。+1e9 は加算であり、他の score がそれ以上なら tail の無条件選択を意味しない。

## owner、寿命、メモリ admission

scratch は `Prefill::Impl::region` の QSA 部分へ一個だけ追加する。count の `qsa_set_bytes` と `carve` の末尾が同じ `take<uint8_t>` を使い、`bytes_needed` にも同じ region max を反映する。借用時は caller-owned expert-cache arena、非借用時は既存 `m.owned` の region allocation が owner。view は USM の所有権を持たず、free、queue 作成、destructor wait、host tensor allocation はしない。

最大の明示 scratch:

| 部分 | bytes |
|---|---:|
| keys `(65538+1)*128*2` | 16,777,984 |
| queries `256*512*2` | 262,144 |
| reusable head output `256*(65538+1)*4` | 67,111,936 |
| 合計 | **84,152,064 (約80.25 MiB)** |

既存 `sel_scores` 67,110,912 B は別の既存 allocation として存続する。新規 buffer の offsets は4-byte alignmentを満たす。raw scratch 全体のサイズは256-byte multiple。既存 Alloc は各 take にさらに256 Bを足して256-byte roundingするので、この一個の追加takeの予約は最大84,152,320 B。これは count/carve 共通で region maxへ含む。candidate 専用 host/RAM tensor bytes は0、viewは固定サイズ。1 layer/chunk 内で key conversion を batch ごと繰り返さず、query conversion のみ batch ごと。

既存 arena は `max(gdn,qsa,moe)`。実 VRAM 増分は `max(gdn,qsa_new,moe)-max(gdn,qsa_old,moe)` と alignment の差で、raw scratch と同一とは限らない。この scratch を別 cudaMalloc して見積りの外へ逃がさない。512 MiB headroom、auto expert-cache slot/resident prefix、borrowed prefill budget、layer-major residual allocation は Main が実 source の算術と allocation receipts で一緒に確認する。candidate を追加した binary/config で cache sizing を再実行し、旧 budget/prefix を強制再利用しない。

**oneMKL 内部 workspace の上限はこの80.25 MiBに含められない。** 現 API は既存 `Gemm::workspace_` も渡さない。同じ library の hidden allocation/residency は source から固定できないので、Main は初回と warmed peak/実 free の差を別採取し、内部割当を含む admission を実証するまで production qualification を保留する。新 caller-owned scratch の owner/lifetime は確定しているが、全 library peak が80.25 MiB以内という主張はしない。scratch を小さく見せるため driver/oneMKL pool を無料と数えない。

寿命の順序は既存非 null **in-order compute queue** 上で `append full chunk -> prepare pooled/dead -> query pack -> head0 GEMM/post -> ... -> head3 GEMM/post -> topk -> 次 batch -> invalidate -> attention -> GDN/MoE shared-region reuse`。既存 async submissions の順序で bytes を保護する。追加の queue、event wait、host readback、per-head/per-batch sync はない。release/relayout は既存 completion 境界を維持する。`invalidate` は host flag のみで queue を drain せず、後続の reuse は同一 queue 上の順序で安全になる。

pooled は完成後安定するが append は同じ chunk の初めに live rows/dead を更新する。従って converted keys は**layer/chunk ごと必ず prepare**し、その区間の batch 間でのみ再利用する。layer、chunk、request、RESTORE、relayout、MoE arena overwrite をまたぐ cache はない。最大256 query以外の残りbatchを含めても key 構築の位置は変わらない。score helper は stream identity を保存して異なる queue の再利用を拒否する。

## nonfinite と precision 契約

FP32 dot から FP16/BF16入力・FP32 GEMMへの変更は**意図した score precision algorithm 変更**。Main の同 dtype・同算法 bitwise 診断を緩めるものではない。既存 FP32 scorer を reference とし、score差、top-k差、routing品質、model品質を別 gate とする。

GEMM の各 C cell を**ReLUの前**に `isfinite` で検査する。非finite の場合、その `(query,block,head)` の元 FP32 Q/K を同じ32-lane・4要素局所積和・xor 16→1 reductionで再計算する。flag は C cell由来で全lane共通、reduction後lane0値を broadcastして後続 branch を uniform にする。4 head を足した最終 total が非finite なら元 FP32 の全4headを再計算する。元の NaN dot は `dot>0?dot:0` により0、元の正Infは正Infのまま。根拠のない clamp、epsilon、NaN finite化は加えない。

これは**観測された GEMM output 非finite**の数値 fallback。half overflow/Inf/NaN、または finite products の accumulation overflowを host sync無しで扱う。有限な underflow/subnormal flush/roundingや近同点のroute差はfallbackしない。oneMKLが非finite inputを有限outputへ変換する実装挙動を、この機構が検出する保証はない。そのため fixture に raw NaN/Inf を含め、実 library の behaviorを gate する。非finite input全体の独立事前検出や完全補償をclaimしない。finite Cでも不正結果になる library/backend は不合格として候補を無効化する。

同期/非同期 library exception、device fault、allocation failure は数値 fallback の対象外。既存 error/wait 境界で失敗を記録する。submit途中の失敗から host が buffer を再利用して old scorerを投げる安全性は保証しないため、catchして続行する経路を追加していない。失敗 arm は失敗のまま停止する。

## Main の独立 component gate とコマンド案

`fixtures/tail_causal_fixture.py` は標準 library の CPU専用 oracle。個々の cell を score順・cellID順で選び、block radix実装に依存しない。手書きexpected IDsで `n%4=0/1/2/3`、dead置換、future poison、同点途中切断、identity、+1e9加算を確認する。さらに全 Tbatch=1..256 の stride/最大logical端点、half overflow/subnormal/BF16 tie を確認する。ここでは実行していない。`fixtures/config_shape_fixture.cpp` は新headerそのもののexact parse、範囲/overflow拒否、全Tbatchのbyte算術をCPUだけで検査する別source。startup freezeは製品sourceのstatic経路とMainの新プロセスarmで確認する。

patch 内の `tools/sycl/qsa_score_gemm_component.cpp` は製品と同じ新 helper と既存 FP32 scorer/top-k を呼ぶ。`sycl_qsa_score_gemm_component` は `EXCLUDE_FROM_ALL` manual CMake target、CTestに追加しない。精度 mode は製品と同じ startup parse を使い、unset/不正値を拒否する。独立cell-ranking oracle、simple exact score oracle、全buffer redzones、future score/unused ID sentinel、両armの個別初回/warmed selection time、score差・ID差・query差・選択cell集合一致数を出す。score difference は品質合格にしない。

実行時の明示 device+host は512 MiB以内へ admissionし、独立expanded-cell oracleの最大262144cellsとsort temporaryも予算化する。guard含む device allocationとhost budgetを最初に出す。oneMKL内部 memory はこのcomponentにも別途加算する。診断の timing 境界にだけ既存 runtime wait を置き、製品helperへのwait追加はない。

以下は**Main 用の未実行例**。既存 build が閉じた後に専用 build identity・artifact lock・deadline・ログ予算を固定して使う。`BUILD` は Main が選んだ候補 build directory に置換する。

```sh
python3 bench/results/2026-10-15-pr6-recovery-audit/qsa-score-gemm-candidate-v1/fixtures/tail_causal_fixture.py
cmake --build BUILD --target sycl_qsa_score_gemm_component -j2
STRATA_SYCL_PREFILL_QSA_SCORE_GEMM=fp16 BUILD/sycl_qsa_score_gemm_component --validate 7 4099 1 exact
STRATA_SYCL_PREFILL_QSA_SCORE_GEMM=bf16 BUILD/sycl_qsa_score_gemm_component --validate 7 4099 1 exact
```

Main が同じ source/binary/toolchain/device/libraryで各dtypeを新プロセスで比較する。component sourceはshape可変なので、少なくとも `(nq,last_n)=(1,1),(1,4),(3,7),(7,2051),(7,2052),(31,4095),(32,4096),(33,4097),(127,262141),(128,262142),(129,262143),(255,262144),(256,262144)` と、Tbatch=1..256をsmallshapeで確認する。`kind` は `exact,near-tie,overflow,subnormal,nonfinite,tie`。BF16を range controlにし、FP16を第一性能候補としても測定前にwinnerを決めない。partial selection batchとmaximum65538score strideは常に維持する。`--validate` は exactな別epochで同じ prepareを二つのsub-batchから再利用し、invalidate後の拒否、同じsource pointerのpooled/deadを書き換えた再prepareの反映も検査する（nq=1は一batch）。

`--benchmark nq last_n samples kind` は1..9の個別 sample、first-callはsample=-1、warmed順はAB/BA交代。選択subtotalはconversion、4 GEMM、4 post、top-k全て含む。ここではcold/warmを出す小比較で、全prefillやexclusivekernel timeの代用ではない。queue/setup/validationcopy/CPU oracleはtiming外。複数batch再利用・mutationは `--validate` 専用で timing外に置く。全runtime gate は Mainの実行まで未完。

## Main の model quality / cost / 採否 gate

1. 最初に phase診断とは別に source resolver と generated/actual entrypoint、新 source のcompile flags、startupmode、nq/active/stride/maxlogical、同一in-orderqueue、defaultOFF allocation/submitが不変である証拠を固定する。OFF の既存 bitwise control を維持する。
2. FP32/FP16/BF16それぞれの input identityとQ/K値域を固定し、変換overflow、raw非finite、subnormal、zeroへの丸め、fallback発動を観測する。componentはfallback countを保持しないため、Mainの観測armでは bounded counter/capture を別設計する。score abs/rel/ULPと非finiteclass、top-k selected-ID集合一致率・flipquery層/位置・FP32境界margin・tie/near-tieを保存する。IDが変わった代表だけを必要tensor consumerとして保持し、失敗instanceを全量複製しない。
3. first/final logits、teacher-forced target NLL/PPL、KL、既存long-context retrieval/task metric、route/attention/residualの必要比較、RESTOREを**別**に評価する。34K/長文/fulllogical256K/実physicalresidencyの条件を区別する。token一致や小score差だけを合格にしない。候補はprecision変更なので採用前に許容quality変化の基準をMainが明示し、結果を見て閾値を後付けしない。
4. clean processで転送と変換込みQSA selection、fullprefill、cold decode/restored decodeを独立個別sampleで測る。mode以外同じcache/residencyを固定した比較と、同じVRAM/headroombudgetでauto cacheを再計算した実用比較を区別する。actualexpertprefix、queue、driver/oneMKL allocations、RAMpeak、VRAMpeak、arena old/new/deltaと初回librarywarmupを同じreceiptに残す。nonexclusive 49.455秒phase intervalからspeedupや上限利得を推定しない。
5. shape/memory/faultが通り、独立quality基準を満たし、変換/top-k/失ったcacheを含めてもfullprefillが改善し、独立decode/RESTOREが退行しない部品だけを採用対象にする。失敗・差なし・未評価はそのまま記録する。現在の状態は**全arm未評価、defaultOFFのproposal**。

次のconsumerは Main の phase診断完了後のlocalpatch review。rawtrace/tensorの生成・削除はない。今後の測定は `ARTIFACT_RETENTION.md` の64MiB rawlog予算、個別sample・fault結果保持、代表failureと必要fixtureだけを残す規則に従う。pinsが変わった過去結果は書き換えず、新patchidentityとの対応を記録する。
