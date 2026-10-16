# Layer-major periodic checkpoint の整合境界案

状態は **提案のみ・未適用・未build・未実行**。Main の同じ 92acf71e binary による Direct0/Direct1 各 full 3 runs を妨げる操作はしていない。Git、build、test、GPU、server、profiler、cleanup は実施していない。書込みはこの新規ディレクトリ内だけ。`generate_proposal.py` の固定文字列変換と SHA256 計算だけを実行した。スキルの通常の commit/build 手順より今回の限定指示を優先した。

`source.diff` は製品へ適用する候補、`proposed/src/program/generate.cpp` と `proposed/include/strata/program/prompt_checkpoint.hpp` は適用後の全ファイル。入力と出力の byte 数・SHA256 は `sha256-ledger.json`。Main の source が進んだ場合は台帳の入力 SHA を先に照合し、差分を盲目的に適用しない。Niko1221/Strata 基盤のこの fork が対象で、CUDA/HIP の実装移植は含まない。

## Source finding

Main の仮説を覆す保護は見つからなかった。

- `src/prefill/layer_major.inc:300-380` は layer ごとに segment 全体を `run_impl` し、次の layer に進む。最終 layer の `on_chunk` は全 chunk を元の caller に転送する。先行 GDN layer の state は既に segment 末尾にある。
- 同ファイルの `on_stage_chunk` は最終 layer、whole-segment の末尾だけを渡す。`generate.cpp:4979` の別経路 `on_chunk -> checkpoint_at(done)` を保護しない。
- `src/core/conversation_state.cpp:55-71,142-169` は token 数の容量、geometry、running-state buffer の存在・サイズを確認する。`gdn_state` / PLE / QSA metadata が **その IDs prefix と同じ位置で計算されたか** は検証しない。`checkpoint_save` の成功はこの問題の正しさ証明にならない。
- `Prefill::run` は `n > actual chunk` のときだけ LM を使う。短い segment は既存 `run_impl`。`run_prompt_with_mtp_experts` は LM owners の退役後に MTP experts を復元して戻る。
- 既存 outer loop の `reread_to, root_at, turn_at, n-1` は順に見て `to <= at` を飛ばす。短い read は outer segment の長さで windows を選び、window commit と MTP prefill を終えてから戻る。既存 periodic checkpoint は windows 自体では作らない。

R608 の現行 cold bench は prompt-cache0 なので `checkpoint_at` は no-op。この案を bench 中に適用したり、flag だけで cache を有効にしてよいという結論ではない。

## 提案する最小変更

変更は既存 `generate.cpp` と既存の純粋 CPU header `prompt_checkpoint.hpp` の二つ。追加 Prefill API、kernel、CMake、既存 fixture の変更は不要。

`STRATA_PREFILL_LAYER_MAJOR` が exact `1`、single GPU、prompt-cache > 0、interval > 0 の全条件を満たす場合だけ `coherent_periodic` を立てる。非 LM / multi-GPU の callback と stage-part 組立ては既存のまま。cache0 / interval0 は outer boundary ごとの現行 cold 処理を保つ。唯一共通の変更は既存三か所の `position + interval` を飽和加算にすること。通常の範囲では同じ値、overflow の場合は INT64_MAX。

対象モードでは `on_chunk` の MTP callback と progress をそのまま行い、そこでの periodic capture を抑止する。caller の batched branch が interval に到達する chunk end で分割する。各 `Prefill::run` が全 layer、全 MTP callbacks、owner の退役と MTP expert 復元を終えて戻り、さらに `refill` が prompt loan を返却してから `checkpoint_at(end)` する。loan は次の segment の前に再度 lend する。slot restore 失敗 / checkpoint 保存失敗は、同時に cancel が来ても単なる cancellation として飲み込まずエラーにする。

outer の root/turn/reread 境界を残し、その内側だけを分割する。root/turn の保存も残す。同じ L の periodic と root/turn が重なった場合は既存 `checkpoint_at` の de-duplication が同じ prefix を再使用する。window を使うかどうかは元の outer segment 長で一度だけ決めるため、分割で短くなった batched segment が windows に変わることはない。window が threshold を越しても periodic を作らない既存仕様を維持し、次の batched chunk end で保存する。

`STRATA_CKPT_REREAD` により `reread_to > 0` になった request は既存 INT64_MAX sentinel と一回の batched reread を維持し、periodic を作らない。初回の cold request は reread_to がないため通常の分割保存になる。この diagnostic が root/turn を含む過去の read 分割を厳密に再現するわけではないことは既存にもある。数値差が出た場合は同じ IDs・chunk・outer/periodic schedule の controlled reread と分けて調査する。reread の差を自動的に checkpoint 破損や修正成功と判定しない。

## interval と chunk の仕様

従来は resume + interval を最初の threshold とし、callback の `done >= threshold` で保存後、**実際に保存した done + interval** に進める。interval の倍数へ遡って保存したり、一 callback で飛び越えた複数 checkpoint を作る仕様ではない。root/turn 保存は threshold を進めない。

`lend` 後の実 `sp.chunk()` を使い、その segment 開始位置から見た最初の chunk end が threshold に到達するよう切る。例: chunk8192、resume0、interval16384 は 16384、32768。interval16385 は 24576、以後はそこから16385先を最初に越す chunk。threshold が短い windows のため既に過ぎている場合も、次の batched chunk の末尾を使う。outer boundary が先ならそこで一旦止まり、threshold 未満なら periodic 保存せず、root/turn 保存だけを行う。

`request_chunk` は borrowed-buffer path の要求長を256へ切上げ、configured max で cap する。今回の planner はその前に推測せず、lend/relayout 後の actual chunk を参照する。次の segment でも remaining outer length で lend するため、長い途中では同じ chunk、最後の短い tail では request-sized capacity となる。自前 buffer の path は既存の実 chunk を使う。CPU fixture の fixed chunk model は relayout 自体を検証しないので、actual engine の境界記録が必要。

新規の全境界加算は end の差によって上限を確認する。round-up に `distance + chunk - 1` や chunk 数の乗算を使わない。`checkpoint_next` は0を無効 sentinel、overflow を INT64_MAX にする。正の segment は必ず end > begin、end <= outer_end で一段進み、interval1 / interval0 / INT64_MAX でも同じ位置への loop は作らない。物理 engine は既存 session/context bound がさらに小さい。CPU の INT64_MAX 算術ケースを物理 context の検証と数えない。

## Cancellation / error / LIVE / RESTORE

途中の layer/final-layer callback が cancel/error で false を返した segment からは保存しない。既に全 layer と refill が成功した以前の segment の checkpoint は残る。次 segment の開始前に stop flag を検出したら新しい run をせず既存 cancellation path に戻り、残る loan を既存 outer cleanup で refill する。`pp_reached` は従来どおり最終 layer callback / window progress で更新し、cancelled request を LIVE として公開しない。

成功した boundary と同時に cancel が到着しても、その boundary の checkpoint は全層で整合した prefix なので保存可能。以後の停止は次 segment の冒頭または既存 decode path で扱う。checkpoint/refill 自体のエラーは `boundary_failed` により fatal として扱う。wrapper の MTP experts 復元失敗は既存 fail-stop が先に処理する。run 自体の false と stop flag の扱いは従来のまま。

`c.ids = cur[0,end)` を維持し、処理対象も `ids[a,end)` と同じ位置で渡す。MTP next IDs は従来どおり `cur[p0+t+1]`。decode の最新 sampled token はまだ state に commit されていないため、今回も `live.swap(consumed)` を変更しない。RESTORE 時の PLE previous IDs 再構築、QSA metadata restore、MTP kv_restore と checkpoint prefix selection / cvec / image filtering / eviction を変更しない。これらを「未変更だから検証不要」とは扱わない。

## 性能・メモリの代償

LM の expert 再利用単位が whole outer segment から periodic segment になる。root/turn なしの34602 batched tokens、chunk8192、interval16384なら 16384 + 16384 + 1834 の三回になり、二回の長い LM pass がそれぞれ全48層512 expertsをロードする。最後の1834は短い run_impl。全長一回の LM より expert upload、残差 buffer admission/割当、同期、MTP expert release/restore、loan refill/re-lend が増え得る。性能数値は未測定。

interval <= chunk では periodic segment が一 chunk になり `Prefill::run` の短文 guard により run_impl を使うため LM の複数 chunk 再利用を失う。MTP release scope も n <= chunk では detach しない。機能を維持した結果としての明示的 tradeoff であり、interval を勝手に0へ変更していない。cache0 / interval0 の既存性能にはこの分割を持ち込まない。periodic snapshot の112 MiB程度/個の GDN payload と eviction policy も別に host-RAM admission を確認する。

## Main が測定終了後に行う検証計画

全項目は **NOT_RUN**。Direct0/Direct1 が両方終端してから Main が一つずつ実施する。プロセス起動前に SYCL/Level Zero/driver/experiment 環境を固定し、実行中に setenv/unsetenv しない。

1. 台帳の source SHA を照合して patch を review/apply。独立 CPU fixture は `c++ -std=c++20 -I <this-dir>/proposed/include <this-dir>/boundary_model_test.cpp -o <owned-temporary-output>` で build し実行する。実際の統合済 header を検査する際は include を製品 include に置換する。small exhaustive token-event oracle と算術 planner、overdue window、partial outer boundary、resume、interval0/1/16384/16385/INT64_MAX、near INT64_MAX、cancel/error/refill failure を確認する。既存 prompt_checkpoint fixture と source resolver の actual SYCL build も Main が直列実行する。fixture は本エージェント未起動。
2. Main の現行候補と修正候補を別 owned process で、同じ pack / input IDs / binary flags / MTP4 / expert placement / sampling / chunk8192 / FP32 saved R=4096 で比較。cache0 と interval0 は output IDs、first logits、PP callbacks、loan lifecycle、MTP path、cold PP/decode を照合する。Direct0/1 の既測定 binary を修正した binary と混ぜて集計しない。
3. cache6、periodic16384 の first prompt を root/turnなしで32768超、非倍数tail34602等にし、保存位置と prefix IDs hash、各 layer GDN、PLE history/prev、全12 QSA tail/dead/block position、pooled dead-row復元、target/draft KV の位置・hashを記録。選択された periodic checkpoint から同じ suffix を RESTORE し、同じ boundary schedule の full reread と first logits、残差、実出力 IDs、MTP draft offered/acceptedを照合。GPU counters/protocol成功だけでpassにしない。
4. 別 owned process の STRATA_CKPT_REREAD を同じ fixtures に適用して独立照合する。元 diagnostic と同じ「全prefix一回」schedule で差があれば、scheduleを揃えたcontrolled rereadを追加し、原因を分離する。復元・reread双方の個別サンプル、どのcheckpointを選んだか、RESUME/REUSED/fresh lengths を残す。
5. tokenizer-verified turn ID248045、root threshold2048、tail-role boundaryを含む multi-turn / branching / second chat shared-root / same-prefix LIVE / shorter rewritten prefix miss / conversation cache off の各ケース。root/turn と periodic が同一点・前後・interval跨ぎになる入力を用意する。short-read0 と64を別 process で検査し、63/64/65-token outer suffix と periodic thresholdをwindowsで跨ぐ場合も確認。thinking on/off の実際の serialized IDs を使う。MTP4 prompt draft_kv/fallbackと後続decode整合を検査する。
6. prompt cancel を early layer、final-layer chunk途中、successful segment直後、window途中で発生させ、completed checkpointだけが残り、partial LIVEが使われず、次requestが正常にresume/rereadできることを確認。run/copy/checkpoint/refill の fault injection はまず CPU/controller scopeで、既存 fail-stop契約に従う。実GPU faultを無秩序に注入しない。error を cancel成功へ変えない。
7. LM+multi-GPUは現行未対応のため有効化を試して採用しない。nonLM multiGPUのstage parts callbackとcheckpoint assemblyが今回のcoherent flagで抑止されないことをCPU/mockまたは既存対応機で検証する。CUDA/HIP engineの実装・実機検証はこの作業範囲外として記録する。
8. **実際の `--max-context 262144` を割り当てた SYCL engine** と host-backed/streamed QSA KV fallback で、periodic checkpointから長prefixを復元し、context末尾まで到達する長いpromptと合法な後続decodeを行う。入力はactual tokenizer IDs/hash、n-1 batched / final input token契約、物理allocation、KV slot/page modeとfallback証跡を記録する。261K相当suffix等の位置は実max/contextとdecode枠から正確に決定し、容量超過の入力は成功条件にしない。小さい論理capやCPU plannerの262144定数だけではfull physical256K合格にしない。これが未実施ならfull256K未検証のまま。
9. 数値・restore・cancel・fault gatesが通ってから、別 clean process で >=32K と実agent multi-turn workloadの各armを反復測定。TTFT、fresh PP、checkpoint copy/refill、cold/restored decode、MTP通常本文を分け、全個別サンプルを保存。interval16384のreload増をphaseで説明し、interval変更の最適化は別比較にする。

CPU fixture は純粋 helper と状態位置の模型を検査する案で、実 GDN/QSA/PLE/MTP の数値や実API cancel pathを走らせない。fixture内のタグ一致を実 tensor一致と呼ばない。現時点の採用判断は未評価で、functionality を維持する具体的 source 案までが今回の到達点。

## 保持

consumer は Main の source review / 後日の適用とCPU fixture / lifecycle gates。proposal一式の上限は2 MiB。見直し点はMainのreviewと統合完了時。必要なdiffとcompact report/台帳/fixtureをGitへ残す判断はMainが行う。本エージェントは既存artifactを削除していないためdeletion manifestは空である。新規measurementsやlarge captureは存在しない。
