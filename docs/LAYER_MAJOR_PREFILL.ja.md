<!-- SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# 実験用 layer-major prefill の復元

`STRATA_PREFILL_LAYER_MAJOR=1` は layer ごとに全 chunk を読む経路を選びます。
起動時に固定し、省略、`0`、`01`、`true` は従来の chunk-major 経路を使います。
旧 `STRATA_PREFILL_COMPACT=2` は復元しません。現在の kernels、MMQ/FP16 fallback、
batching、scratch/gated alias は既存の設定を使います。

単一 GPU の fresh standalone と通常の `--serve` の GEN を対象にします。
全 KV resident と、現在の context262144/resident32768 streamed-KV 経路を受け入れます。
streamed KV は各 chunk の既存 host-prefix seed、append、identity stage、attention を
維持し、layer ごとの永続 stage cache を追加しません。KV の mode/page/context は全 QSA
で一致する必要があります。物理容量、数値一致、速度は別途検証する契約です。

prompt/conversation cache と保存を 0/OFF にし、fresh request ごとに通常の checked
session zero を実行します。実際の resume/live/checkpoint/parked/slot 状態も確認します。
前 request の非再利用 metadata は終了時に捨てます。warm-up と後続 GEN で同じ設定を
使えます。1-token prompt は prefill を行わず通常の token 経路へ進みます。
途中 checkpoint、public stage callback、MTP、split/helper、pipeline、vision、
borrowed workspace、adaptive swaps、SAVE/RESTORE、batch、VRAM command は拒否します。
phase-cache/head-lease との統合は後の段階です。phase0 の通常 serve を使用します。

`--no-prefill-borrow --prefill N --prompt-cache 0 --conversation-cache-mib 0` と、
明示的な正の `--vram-reserve-mib M` を必要とします。serve では `--short-read 0` も
明示して、request の途中を Verifier window へ切り替えません。auto chunk と自動 reserve
減額は使いません。raw `--spec 4` は許可し、通常の最初の decode window は実効 T=1 です。

`STRATA_PREFILL_LAYER_MAJOR_R_GPU` は GPU prefix の最大 token budget です。
明示的な符号なし十進整数 `0` から `262144` を受け入れ、負数、空白、overflow、後続文字を
拒否します。request の実際の n に clamp してから chunk 境界へ下方向に丸めます。
通常の conditioning は prompt P の最終 token を除いた n=P-1 です。全 n が入る場合は
最後の短い chunk も GPU に含めます。これは request 長に合わせる budget の意味であり、
空き容量が不足した際に reserve や prefix を自動で減らす fallback は行いません。
例えば n=17/chunk8/budget32768 は GPU17 rows、n=33/budget17 は GPU16/RAM17 rows です。
trace は要求 budget と実効 GPU/RAM rows を区別します。

入力、pos0=0、context<=262144、shape、byte 積、store 境界を割当前に検査します。
FP32 の全 residual rows を GPU prefix と RAM tail に保存し、chunk は store をまたぎません。
既存 model、workspace、head、session、通常 expert cache を確保した後の fresh 空き
VRAM/RAM を調べ、新しい residual store、one-layer expert cache、source staging、
residency metadata と明示 reserve が入らなければ拒否します。lazy kernel/provider の
一時確保は reserve で扱い、物理 high-water の保証とはしません。デバイス名で分岐しません。

layer-major の count/carve は通常の device expert staging ring を割り当てません。
全 active-layer expert が private cache に resident であること、slot 数、offset と実際の
blob byte 容量を各 chunk の前に検査します。host Stager、pinned source、events と
MMQ gather ring/group buffers は残します。cache miss 用 ring がない経路で missing expert
を処理しません。streamed-KV stage と現在の alias allocation は維持します。

各 layer の全 expert blob を一度だけ新しい cache に upload し、その layer の全 chunk
が cache を再利用します。source の blob pointer は checked synchronous copy が終わるまで
保持し、その後に次の blob 呼び出しで再利用できます。transient source は現在の
`copy_blob` を使います。非同期 upload の最適化は別の変更です。

QSA/GDN の global ordinal、位置と状態、PLE history は現在の chunk 実装で進めます。
PLE の二つの直前 token は各 layer で request 開始時の同じ値へ戻します。decode cache は
変更しません。内部 callback は中間 layer の residual bridge を書くだけで、公開 on_chunk
は最終 layer の checked compute completion 後にだけ residual rows/output/progress を
渡します。途中の全 layer 同一位置 snapshot は作りません。callback から checkpoint を
保存してはいけません。tokens/chunks は最終 layer で一度だけ加算し、total wall は外側の
一回分です。one-layer 境界で既存 inter-layer norm fusion が使われないことがあるため、
数値一致を推定せず全 head/state で比較します。

成功、エラー、cancel の全経路で checked compute/copy drain を確認してから callback、
cache、stack residency、residual pointer を復元します。不明な copy/drain completion は
unwind/free より先に fail-stop します。失敗した session は rollback せず、非finite/不健全な
request を再利用しません。

`STRATA_TRACE` は layer、役割、実際の uploaded bytes、slots、computed chunks、
要求/実効 prefix と GPU/RAM rows、checked completion を示します。速度測定では trace と
transfer/timing 診断を無効にし、同じ source/binary、shape、KV、prefix budget、MMQ 方針で
交互比較します。full vocabulary/output、session state、後続 decode、repeated fresh GEN、
normal teardown と fault 結果を別に確認します。PR #6 の旧測定値はこの経路の測定値ではありません。
[artifact retention](ARTIFACT_RETENTION.md) に従って commands、環境、individual samples と結果を残します。
