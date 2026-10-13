<!-- SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# 実験用の phase decode cache

`STRATA_PHASE_DECODE_CACHE=1` は通常の `--serve` の直列 GEN に限定した実験です。
値は起動時に固定し、省略、`0`、`01` は従来の経路を使います。
通常の stock HTTP client の ckpt 省略は、実際の checkpoint、live session、
parked conversation、slot、再利用状態が無い場合に受け入れます。
prompt cache と conversation cache の容量はゼロに保ちます。
GENI、SAVE、RESTORE、batch、multi-GPU、MTP、pipeline、vision、remote helpers は受け入れません。
元の 262144 context admission と生成用 reserve はそのまま使います。

各 GEN は prefill 成功後に Prefill と Gemm scratch を退役し、固定 virtual address の
追加 expert mapping を fill、readback、全 residency 比較してから decode します。
次の GEN は前の verifier graph を退役し、追加 residency を無効化して unmap、drain し、
同じ session buffer を使う Prefill を再初期化します。通常の fresh session zero はその後です。
KV、GDN、PLE、QSA の所有 buffer を phase change で交換しません。
短い warm-up も同じ手順です。cancel、nonfinite、完了が不明な操作は fail-stop し、
HTTP server の自動 restart と reload は抑止します。次の実験は server を明示的に起動します。

VMM aspect、device recommended/minimum と context minimum granularity、最大 allocation、
実際の空きメモリを SYCL から取得します。開始 address、reservation、map extent の
alignment、divisibility、overflow を検査し、不確かな ownership を継続しません。
3 expert/layer の基底に実験用 quota の 16 expert/layer を追加します。
これは測定済みの最適値という意味ではありません。容量不足で quota を黙って下げません。
非対応 device は既定の OFF 経路を使います。

attach または detach が失敗した範囲は、mapped / unmapped のどちらとも確定しません。
SYCL unmap は mapped と確認できる範囲にだけ使えます。この不明状態では owner を
poison し、追加 request と reader 公開を拒否します。map rollback はその場で止まり、
繰り返しの close も unmap、physical 破棄、virtual reservation 解放を行いません。
既に成功した mapping を含め、physical handle、virtual range と backend/context の
所有権を process 終了まで保持します。不明状態の owner を破棄すると既存の
std::terminate 方針に従います。自動 retry、暗黙の回復、直接 Level Zero による回収は
ありません。physical 作成前の失敗など、mapping 状態が確定している失敗の cleanup は
従来通りです。provider の map/unmap 失敗時の atomicity は仮定しません。

Linux の CPU fixture は不明 map、失敗 unmap と rollback 中の unmap 失敗を子 process で
検査します。注入を解除しても繰り返し close が所有権を保持し、unmap/free/destroy と
reader 公開を行わないことを確認した後、destructor を呼ばず _exit します。
mapped と unmapped の両方の模擬失敗後状態を使いますが、実 provider の挙動を証明する
ものではありません。

CPU fixture は `phase_cache_owner_test`、小さな実 VMM fixture は
`phase_cache_vmm_test --small-vmm` です。後者は CTest に登録せず、GPU 作業を
直列管理する担当者が実行します。source、compiler、runtime/provider、モデル入力を固定し、
二つの fresh GEN と正常 QUIT を確認してから性能を比較します。
slot byte と residency の一致は必要です。生成文の完全一致は品質の必須条件にしません。

phase ON の `prompt_ms` は、GEN admission 成功後、前の GEN の退役処理を始める直前から、
当該 prompt の完了後の map、fill、readback、memcmp、residency publish と host refresh が
完了するまでの実経過時間です。前 GEN の quiesce、graph 退役、invalidate、unmap、
Prefill 再初期化と、当該 GEN の quiesce、Prefill reset も含みます。最初の GEN では
前 GEN の退役処理はありません。短い warm-up も同じ扱いです。起動時の base fill、
GEN admission 前、decode、QUIT は含みません。OFF の時計位置と意味は従来通りです。
`PrefillStats` の `ms_total` は、転送と同期を含む `Prefill::run` の経過時間です。
この集計は別に保持し、phase 処理を加算しません。

phase 専用ログは `begin_gen` / `successful_prompt` の owner counter 差分と、実際の
cycle operation / transition の host monotonic elapsed span を記録します。
quiesce、graph 退役、Prefill reset、再初期化、invalidate、native map/unmap、physical
作成/破棄、drain、publish、checked close を区別します。expert payload の H2D/D2H と
memcmp は residency table の H2D/D2H と比較から分離します。`calls`、`success`、`failed`、
`attempt_bytes`、`success_bytes` は実際に呼んだ操作と戻り値から集計し、期待 slot 数を
コピー件数として扱いません。失敗した操作の elapsed と試行 bytes も残します。
`host_elapsed_ms` と `nested=1` の span は入れ子を含むため、足して wall time と呼べません。
GPU event timing や転送だけの device time ではありません。全 request wall は upstream
client の値と別に比較します。


## Native head の prefill lease と固定 12288 容量

`STRATA_PHASE_HEAD_LEASE=1` は既定 OFF です。最初の GEN まで Prefill の初期化を延期し、
各 GEN で quiesce と reader graph 退役後に native head の weights と scratch を解放します。
最初の upload より前に、対応する type と shape から検証した byte 数だけ、元の raw GGUF
payload を CPU RAM にコピーして保持します。Prefill の全実行と reset が終わってから、
この固定した元の bytes を同じ upload と変換で head に戻します。変換後は既存の Runtime の
checked drain を使い、保存された非同期エラーと watchdog も確認します。ファイルを再度開かず、
同じ metadata のままファイル内容が変わっても reload に使いません。追加の CPU payload は
厳密に `STRATA_PHASE_HEAD_LEASE=1` の場合だけ保持し、head の D2H コピーはしません。
Verifier は同じ NativeHead object を保持し、reload 後に新しい graph を capture します。
upload または変換の submission / completion が不明な失敗では、weights、scratch と元の
CPU source の owner、pointer、byte 数を保持し、retry / run / release / reload を拒否します。
不明な完了を持つ owner の destructor は、DMA source を解放する前に fail-stop します。
submission 前の allocation 失敗だけは安全に解放でき、再試行できます。解放 API が失敗した場合も
解放済みか未解放かを仮定せず owner を poison し、解放と再試行を拒否します。成功した pointer は
個別に消し、後続の解放失敗で二重解放しません。
失敗、cancel、nonfinite、不明な完了では fail-stop し、reload を回復手段にしません。
head の退役と復元時間も prompt_ms に含まれます。

`STRATA_PREFILL_SCRATCH_ALIAS=1` も既定 OFF です。in-order compute queue 上で、最後の reader が
完了した後の emb / bo / ple_emb と qkv / FP32 y を共有します。byte count、owned / borrowed carve、
relayout は同じ起動時の設定を使い、FP32 y と normalizer の計算はそのままです。

`STRATA_PREFILL_GATED_ALIAS=1` は別の既定 OFF の opt-in です。起動時に正確な `1` だけを
受け入れ、未指定、`0`、`01` は OFF です。allocation 前に上の scratch alias と in-order compute
queue を必須とします。両方の HC half で FP32 gated の up GEMM、`gr_mix_r` の全要素 read、
最初の GDN / QSA / MoE scratch writer が同じ queue 上で順に完了するため、gated を shared
region の先頭に置きます。R、xn16、injection、PLE / draft の local view と batching はそのままです。
計算、dtype、precision、cache、session は変更しません。

count と carve は同じ host-only layout helper と固定した設定を使います。region の容量は
block 最大 extent と gate extent の大きい方です。trace は block 最大、gate extent、実際の region
allocation の requested bytes、削減 requested bytes、winner、pointer equality を表示します。
stock 12288 の block extent が gate より大きい設定では、ソース上の削減は
503316736 bytes（480 MiB + 256 bytes）です。`Alloc::take` の padding を含む requested count
であり、physical VRAM の削減、fit、速度の測定ではありません。12K の head lease と scratch
alias の必須条件はそのままで、gated alias は明示的な A/B のための任意の設定です。
reserve と容量を自動で下げません。

固定 12288 容量は `STRATA_PHASE_PREFILL_12K=1` と `--prefill 12288`、上の lease と alias の両方を
明示した場合だけ受け入れます。その他の phase の制約はそのままです。省略時は固定 8192 です。
auto、別の容量、BF16X2、別途所有する KV stage は拒否します。初期化と全 fresh GEN 再初期化に
同じ容量を使い、不足時に 8192 に下げません。full context は 262144 のままです。

head 解放後の新しい device free と、現在の設定から数え直した workspace、token buffer、
mode1 identity table、条件付き GDN chooser と lazy GEMM の上限、既存の `--vram-reserve-mib` を
比較します。KV stage は workspace 内として別表示し、二重加算しません。reserve は既存の
operator 設定（既定 700 MiB、正の値が必要）で、JIT、backend、provider、fragmentation の
未確定分を残すためのものです。GPU 型番では選びません。ログは requested byte の推定、
shared region の実際の最大 count、設定容量、実行 chunk の T を区別します。
この guard は physical peak の測定を代替しません。容量の実機 qualification には、最初の
chooser / GEMM peak、繰り返し GEN、4095 tail と final token、logit / persistent state 比較、
full 262144 context と継続 / restore の確認が必要です。

## serve の first logit row を保存する診断

`STRATA_SERVE_FIRST_ROW_CAPTURE=1` と、既存の私有出力ディレクトリを指定する
`STRATA_SERVE_FIRST_ROW_DIR` の両方で有効になります。省略、`0`、`01`、`true` は
OFF です。OFF の場合、この診断の row allocation、copy、ファイル IO は行いません。
速度測定には使いません。

この経路は既存の phase-cache と head lease の契約を必要とします。
単一 GPU の通常の直列 `GEN 1`、`--suffix-draft 0 --lookup-chain 0`、
prompt/conversation cache と保存、adaptive swaps、batch、MTP、pipeline、
`--dump-logits` を無効にした設定に限定します。起動設定だけでなく、実際の fresh state、
再利用 token 数 0、最初の Verifier window の T=1 と位置 P-1 も確認します。
raw `--spec` は Verifier の容量であり、既存の容量範囲を受け入れます。
通常の `--spec 4` でも最初の実効 T=1 を必要とします。
不適合の request は拒否し、不確実な copy や phase 処理の失敗時は所有者を解放せず終了します。

通常の prefill、head reload、phase publish を通り、最初の正常な非 NaN/Inf window の
`Verifier::copy_logits(0)` から全 vocabulary の FP32 row を読みます。
request と最終 commit、phase completion が成功してから `request-N.bin` を公開します。
layout は int32 vocabulary、int32 rows=1、vocabulary 個の FP32 です。
stderr の COPIED は未完成、COMPLETE は request index、実際の prompt 長 P、
予測位置 P-1、source row 0、実際の再利用 0、vocabulary、byte 数と完成ファイルを示します。

1 process あたり最大 64 request、合計 64 MiB を予約します。作成、write、flush、close、
byte 数、有限値を確認します。`.partial` を排他的に作成し、完成したファイルだけを
hard link で排他的に公開するため、出力 filesystem は hard link をサポートする必要があります。
既存の名前を上書きしません。失敗した partial を完成ファイルとして扱いません。
成功した partial link も残すので、Main が閉じた run の hash と必要な証拠を確認してから
[artifact retention](ARTIFACT_RETENTION.md) に従って整理します。

固定履歴比較では各 request の prefix を `prompt + follow[:i]` にします。
生成した argmax token を次の履歴に戻す必要はありません。外部 controller が各 prefix、
target、完成 row、source/binary identity を結び付けます。1 row の捕捉は full sequence の
teacher forcing、NLL、full quality、256K、RESTORE、速度の pass を意味しません。
