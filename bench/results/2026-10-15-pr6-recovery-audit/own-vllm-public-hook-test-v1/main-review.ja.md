# R833 公開 Gemm 経由 vLLM native16 テスト提案

状態は **SOURCE-ONLY / UNRUN / UNAPPLIED**。このパッケージの4ファイルだけを作成した。
production、既存 tests、CMake、header は変更していない。既存の
`tests/sycl/vllm_native16.cpp` と own の回帰テストはそのまま残す。

基点は coordinator 指定の `5020c05e`。Git コマンドによる commit 同定はしていない。
`manifest.json` の `base_snapshot_sha256` が読んだ source の完全な SHA256 を固定する。
特に CMake preimage を一度読んだ同じ文字列から patch を生成した。Main の tree が進んだ場合、
preimage と公共 API/実装の hashes を照合してから全差分をレビューする。stale tree のファイルで
production を上書きしない。

## 何を通るか

提案ソースは `tests/sycl/vllm_engine_native16.cpp` の1 translation unit。
include は公開 `strata/prefill/gemm.hpp` と既存 `strata/sycl/runtime.hpp`。
`runtime_for(ordinal)` で得た同一 Runtime の `compute()` の実アドレスを
`Gemm::init_external(&q, nullptr, 0, nullptr, 0, err)` に渡す。
`stream()==&q` も確認する。全 allocation はその Runtime の既存 Device registry に登録される。
新 Runtime、default context、独立 queue、独立 registry 実装を加えない。

唯一の GEMM submit wrapper `invoke()` は公開 `f16_out16` / `bf16_out16` を呼ぶ。
adapter の include、型への cast、直接 `beta0` 呼び出しはない。`x::Storage` は元の oracle を
変えずに再利用するための **テスト内 enum** で、adapter 型ではない。
input FP16/BF16 と output FP16/BF16 の全4組合せを実行する設計。

子環境は Main が exec 前に `STRATA_SYCL_PREFILL_NATIVE16_BACKEND=vllm` と
`STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE=compact` または `wide` を固定する。
起動時の getenv 検査は Runtime 初期化より前。setenv/unsetenv はない。
backend=own、tile=auto/未指定のまま、この fixture を走らせて成功とはしない。
この環境検査と vLLM 固有拒否は公開 dispatch の経路検証であり、loaded image / ISA の証明ではない。

## 有界なケース数

各 suite は tile ごとに別プロセス。chain は1ケースで公開 GEMM を2回呼ぶ。

| 1プロセス | dyadic | strict edges | K跨ぎ | BF16 chain | 正例ケース | 成功 public calls | 拒否 public calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| --smoke | 32 | 8 | 4 | 4 | 48 | 52 | 160 |
| --full | 36 | 8 | 4 | 4 | 52 | 56 | 160 |

共有 dyadic は各4形式で T=1,3,17,160、それぞれ (N,K,ldy引数)=(64,32,0) と
(192,96,200)。後者は8要素 padding と wide の N=128 tile に対する64列の端を含む。
前者の beta は +0、後者は -0。full は各形式に **同じ1形状**
(T,N,K,実ld)=(8192,128,64,136) を1件追加し、巨大な全形状直積を避ける。
strict edges は元の T17/N64/ld72 で K32 と K64。
新しい cancellation は T17/N192/K96/ld200、非零項を k=0,32,64 に置き、
各列・行の符号を変えて正確な ±0.25 を確認する。2048+.25 は FP32 で正確であり、
第2 tile 後の蓄積を FP16/BF16 に狭めると .25 を失う。これは reduction 順序用の誤差許容ではない。
chain の既知の BF16 中間値と最終値は元の式を維持し、2つの public call の間に
wait/download/cast/FP32 tensor を挟まない。

テスト所有 Device payload は RAII ledger が実 allocation より長く reservation を保持し、
**134217728 bytes 未満**を各 allocation 前に強制する。手計算の peak は smoke=132352 bytes、
full=3293952 bytes。raw unregistered Device USM と short-span Device も ledger に入る。
Runtime/driver の内部領域を含む総 VRAM 測定ではない。入力を含む全 guard 領域を upload/readback し、
X/W の完全不変性、Y の prefix/suffix と全行 padding を検査する。
正常データ基点は64Bに合わせた登録 Device allocation 内の位置で、全 tensor span も事前確認する。

## Oracle と変更箇所

再利用元はこの repo の MIT `tests/sycl/vllm_native16.cpp` のみ。
外部数値 helper、Torch、TLA oracle、fixture 外の tensor ファイルに依存しない。
独立 FP64/dyadic 値生成と IEEE encode/decode は元の文字列を保持する。
有限非零値は RNE のビット完全一致、NaN は分類、zero は符号を問わない元の `matches()` のまま。
新たな epsilon/ULP 許容はない。input/output min-normal/min-subnormal、subnormal midpoint、
RNE midpoint の正負、Inf/NaN、FP16 最大有限値、BF16 大値から FP16 への overflow を含む
元の17 edge 行は無改変。FTZ が出た場合に通す分岐はなく、strict gradual-underflow の失敗とする。

以下は両ソース内の **inclusive 行番号**。各範囲の同一文字列 SHA256 は manifest に記録した。

| 無変更領域 | 元 fixture | 提案 |
| --- | --- | --- |
| require/product/name/decode/ties_even/encode/classification/matches | 23–95 | 28–100 |
| CheckedDrain comments and implementation | 96–121 | 101–126 |
| dyadic row/basis functions | 152–155 | 204–207 |
| edge data and independent expected construction | 194–225 | 248–279 |
| edge exact comparisons and error gate | 234–245 | 288–299 |
| chain data and independent expected construction | 251–259 | 306–314 |
| chain exact comparisons | 270–278 | 325–333 |
| reject helper and alignment helpers | 283–298 | 339–354 |

全テキスト変更範囲は次表。表記は **1始まりの半開区間 [開始,終了)**。
insert の元区間は空。これは元テストを書き換える patch ではなく、新 translation unit との比較である。
元の数式・判定を変えた箇所はなく、差分は公開呼び出し、queue、ledger、case数、拒否/CLIにある。
`manifest.json.complete_text_change_map_from_original_fixture` に同じ全範囲を構造化保存した。

| 操作 | 元 [start,end) | 提案 [start,end) |
| --- | --- | --- |
| replace | [3,5) | [3,7) |
| replace | [19,20) | [21,25) |
| insert | [122,122) | [127,149) |
| insert | [123,123) | [150,151) |
| replace | [127,129) | [155,157) |
| insert | [133,133) | [161,163) |
| insert | [152,152) | [182,204) |
| replace | [156,157) | [208,209) |
| replace | [158,159) | [210,211) |
| replace | [170,171) | [222,223) |
| replace | [173,176) | [225,229) |
| insert | [185,185) | [238,239) |
| replace | [186,187) | [240,241) |
| replace | [192,194) | [246,248) |
| replace | [226,227) | [280,281) |
| replace | [229,231) | [283,285) |
| insert | [246,246) | [300,301) |
| replace | [249,251) | [304,306) |
| replace | [260,261) | [315,316) |
| replace | [263,265) | [318,320) |
| replace | [266,268) | [321,323) |
| insert | [279,279) | [334,335) |
| insert | [300,300) | [356,357) |
| replace | [303,304) | [360,361) |
| replace | [309,311) | [366,372) |
| replace | [312,322) | [373,384) |
| replace | [324,326) | [386,421) |
| replace | [328,330) | [423,427) |
| replace | [331,341) | [428,455) |
| replace | [342,343) | [456,457) |
| replace | [344,365) | [458,482) |
| replace | [368,369) | [485,500) |
| replace | [374,376) | [505,508) |
| replace | [379,403) | [511,532) |
| replace | [404,406) | [533,554) |
| replace | [407,408) | [555,559) |
| insert | [410,410) | [561,564) |

## 拒否と所有権

各形式で40件、計160件の `invalid_argument` を要求する。未初期化 Gemm、null、X/W/Y の64B基点、
出力と入力の重なり、address overflow、登録 short X/W/Y、host/unregistered Device/ordinary host、
零・負・過大形状、surface/coordinate limits、N/K tail、stride、beta を通る。

own の広い契約を vLLM に誤って当てはめない。特に2Bだけずらした基点、N80/K16、
beta 1/.5 は own の許容に関わるが、選択された vLLM では拒否が必要。
N80/ldy0 の W/Y は own なら実際に収まるよう余裕を確保し、short-span が拒否理由を代行しない。
完全 Y 矩形の拒否は T1/N64、64Bに合わせた short_y 基点から193〜256 bytesを残し、
128 bytesの live row は収めつつ、16B刻みの declared pitch だけが登録末尾を超える構成。
own の final-live-element 範囲検査と区別できる。

別 context/Runtime/queue を作る wrong-context/out-of-order 拒否は本 fixture では行わない。
adapter-private Storage/Tile の不正 enum も公開 API の bool に存在しないため移植しない。
これらは無変更の直接 adapter fixture 側の責務。unsupported device/option/config は
public init/call の明示失敗として fixture が非零終了する。異種 device を作って成功した拒否件数に加えない。

全 async copy/submit の前に、使用する allocation と upload/readback vector の **後** に
元と同じ CheckedDrain を宣言する。途中の enqueue、public call、readback が throw しても、
queue.wait_and_throw と Runtime::check を先に行い、その後で owner を破棄する。
drain 失敗は `_Exit(86)` で owner unwind を抑止し、失敗 wait を解放許可として使わない。
main でも Gemm より後に lifetime drain を置き、Gemm の状態を破棄する前に完了検査する。
completion marker は Gemm の実際の destructor の後。Main は marker と正常 exit の両方を確認する。

## Main の次の作業

`proposal.patch` は新 test 追加と `cmake/sycl_backend.cmake` の小さな opt-in 部分だけ。
`STRATA_SYCL_VLLM_PUBLIC_HOOK_TEST=ON` は既定 OFF、vLLM build option が必要。
新 target `sycl_vllm_engine_native16_test` は EXCLUDE_FROM_ALL、strata_prefill にリンクし、
余分な runtime.cpp/core/adapter を再コンパイルしない。CTest 登録はない。
既存エンジンの mixed-image AOT/JIT 設定、既存 fixture の target は変えない。

Main は現在の whole build が閉じてから全 patch と preimage を確認し、独占した通常の手順で
configure/build と compact/wide の smoke/full を直列実行する。具体的な環境/CLI テンプレートは
manifest の `commands_for_Main_only`。device/provider とその他 SYCL/Level Zero 環境も exec 前に固定する。
process ごとの source/binary/toolchain/env、個別ケース結果、exit、fresh fault/cleanup 記録を残す。
失敗したときに tolerance や strict-underflow を緩めない。成功しても既存回帰、実モデル品質、
262144 context、loaded-image/entry、性能のゲートを代替しない。bench や速度採用の処理は含まない。

保持物はこの小さな source package。実行ログ・tensor は未生成であり、cleanup 対象はない。
Main が将来実行する場合は docs/ARTIFACT_RETENTION.md に従い、有界な process deadline と
64 MiB以下の raw text-log 予算を事前設定し、閉じた run の有用な結果を抽出して保持する。

AGENTS.md、artifact retention とプロジェクト専用
[適用ガイド](../../../../.claude/skills/strata-upstream-sync/SKILL.md)を読み、出自固定・実行分離を適用した。
index と最新 scope review の関連 metadata を参照し、無関係な knowledge-collection 本文は読んでいない。
公式外部 source のコピーや XeStrata 実装/helper/test のアクセスはない。
保護された record、他 worker への接触、Git、build/test/syntaxcheck/sourcecheck executable、
GPU/profiler/library load、process inspection、install、cleanup は行っていない。
生成時には Python の文字列編集・diff・SHA256のみを用い、C++ を実行/構文検査していない。

## 提案の fingerprint

- `tests/sycl/vllm_engine_native16.cpp` SHA256 `7e1d5a45526eb367e1062d46135c2e99a8fa3a12f4e3db9684de9f526b204716` (27144 bytes)
- `proposal.patch` SHA256 `bb714c55c7a043aa468e006fa61490716412043dc6bd7b7c94d9156131ade3f6` (28905 bytes)

report 自身の SHA256 は manifest.artifacts に、manifest 自身の SHA256 は完了通知に記録する。
