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
client の値と別に比較します。これらの source 変更の build、runtime、性能は未検証です。

## Native head の prefill lease

`STRATA_PHASE_HEAD_LEASE=1` は起動時に固定され、既定は OFF です。既存の
phase-cache 条件と native head が必要です。chunk は 8192 のままです。
MTP、複数 GPU、再利用、short-read verifier は対象外です。

ON では startup の Prefill 初期化を最初の GEN まで遅らせます。Verifier
初期化は head がロードされた状態で行い、head の kernel graph はまだ記録しません。
各 GEN で checked quiescence、全 reader graph の破棄、head weights と scratch
の解放、Prefill 初期化と run、成功後の Prefill reset の順に進みます。
その後、同じ GGUF の元の型、形状、payload サイズ、shard、offset を再確認し、
同じ upload と既存の row conversion を行ってから cache の公開と decode に進みます。
Verifier は同じ NativeHead オブジェクトを参照し、新しい graph は再ロード後に記録します。
GGUF の内容の同一性は呼び出し元の固定されたモデル契約が保証します。
GEN ごとの巨大なファイル hash や保持された RAM tensor はありません。

head の quiescence、release、再 upload と conversion は prompt_ms に含まれます。
失敗、cancel、nonfinite、完了状態が不明な場合は fail-stop で、再ロードや再試行を
回復として行いません。QUIT は checked drain と graph retirement の後で終了します。
ログには元の GGML 型と conversion 後の型、weights/scratch bytes、形状、
release 前後の actual free を記録します。要求 bytes の算術は物理的な節約や
12K/16K の fit を証明しません。CPU lifetime fixture も GPU の完了証明にはなりません。
