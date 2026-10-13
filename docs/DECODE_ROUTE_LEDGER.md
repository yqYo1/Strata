<!-- SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors -->
<!-- SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# デコード経路の集計 / Decode route ledger

`STRATA_DECODE_ROUTE_LEDGER=1` の完全一致だけで有効になります。既定は無効です。
通常の単一 GPU `--serve` GEN、native pack、MTP なし、batch なし、
`--suffix-draft 0 --lookup-chain 0` が対象です。prefill は含めません。
`--spec` は verifier の確保容量であり、実際の NT とは別です。通常の
no-MTP 要求では raw `--spec 4` でも実際の dispatch NT は 1 です。
verifier は容量 2 以上を必要とするため、`--spec 0` は要求しません。
共通の host policy は容量 2 / 4 を許可し、dispatch 時に実際の NT=1 を
確認します。engine の容量、数値計算、routing は変更しません。

未対応要求は `eligible=0 incomplete=1` を出します。通常終了は既存の
`wait_commit` 後に一つの出力ブロックを出します。早期終了の
`error_or_early_exit`、invalid、overflow は不完全な診断です。
64 layer × 4 owner × 実際の NT 1..8 の最大 2048 bucket、状態 128 KiB、
終了時テキスト 512 KiB が上限です。固定配列範囲、pointer、count、type、
blob offset を配列アクセスと extent 計算前に検証します。

CPU groups は完了した native pool jobs、entries は実際の job NT の合計です。
GPU groups は同一 dispatch 内の一意 expert、NT はその expert の実際の行数です。
`gpu_vram` は provider plan の kind=0、`gpu_pcie_staged` と
`gpu_pcie_mapped` は kind=1 と実際の pcie_mode で区別します。PCIe ownership
はコピー件数ではありません。GU extent は [0, down_off)、Down extent は
[down_off, bytes)、blob はその合計で、group ごとに一度加算します。
物理 DRAM、H2D bytes、request 全体の expert union の測定値ではありません。
`queued_h2d_bytes=unavailable` は fetch callback が実際の queue 成功 byte 数を
返さないことを示します。時間計測は追加しません。既存 decode timing と
pool phase_ms は重複し得る host interval であり、合計を exclusive critical
path として扱わないでください。

通常の serve は n-1 conditioning tokens を prefill し、最後の prompt token を
最初の T=1 verifier window で処理して最初の出力を生成します。この window は
集計範囲に含みます。layer_calls は実際に観測した完了 dispatch 数です。
正常終了した通常の NT=1 要求では、全 callback が観測できれば
layer_calls = layer 数 × emitted です。終了時に総数と各 layer の回数を照合し、
正常終了、invalid/overflow なし、両者一致のときだけ coverage_complete=1 とします。
callback を省略する GPU shortcut、cancel、失敗 window、算術 overflow などでは
coverage_complete=0 です。期待値は coverage 検査専用で、観測していない経路を
補いません。分母には実際の layer_calls を使ってください。この集計だけでは
全処理の bottleneck や exclusive critical path は判断できません。
Main が CPU fixture、env の完全一致、guard、早期終了、request reset、candidate
build、bounded 32K GEN の parity / fault を検証します。
速度比較は変更していない upstream bench.sh のみです。この診断の速度は
性能比較に使いません。

Exact `STRATA_DECODE_ROUTE_LEDGER=1` enables this default-off observer. Scope is
ordinary single-GPU --serve GEN with native packs, no MTP or batch, and
--suffix-draft 0 --lookup-chain 0. Prefill is excluded. Raw --spec describes
verifier allocation capacity, separately from actual NT. Ordinary no-MTP
requests with raw --spec 4 still dispatch actual NT=1. The verifier requires
capacity at least 2, so --spec 0 is not required. The shared host policy accepts
capacities 2 and 4; dispatch additionally enforces actual NT=1. Engine capacity,
numerical computation and routing are unchanged.

Unsupported requests emit eligible=0 incomplete=1. Normal close emits one block
after existing wait_commit. error_or_early_exit, invalid or overflow means an
incomplete diagnostic. Bounds are 2048 buckets (64 layers, four owners, actual
NT 1..8), 128 KiB state and 512 KiB closing text. Fixed array bounds, pointers,
counts, types and blob offsets are validated before array access and extent math.

CPU groups count completed native pool jobs; entries sum actual job NT. GPU
groups count distinct experts within each dispatch; NT counts actual routed rows.
gpu_vram is provider-plan kind=0. gpu_pcie_staged and gpu_pcie_mapped use kind=1
and actual pcie_mode. PCIe ownership is not a copy count. Logical GU extent
[0, down_off), Down extent [down_off, bytes), and full blob extent are added once
per group. They do not measure physical DRAM, H2D traffic or a request-wide
expert union. queued_h2d_bytes=unavailable means the fetch callback does not
acknowledge actual queued bytes. No timing is added. Existing decode timers and
pool phase_ms may overlap and must not be summed as an exclusive critical path.

Ordinary serve prefills n-1 conditioning tokens, then processes the final prompt
token in its first T=1 verifier window to generate the first emitted token.
This window is within ledger scope. layer_calls counts observed completed
dispatches. For normal ordinary NT=1 close with all callbacks observed,
layer_calls equals layer count times emitted. Closing coverage checks both the
total and each layer's count. coverage_complete=1 requires normal close, no
invalid/overflow, and matching counts. A GPU shortcut omitting a callback,
cancellation, failed window or arithmetic overflow yields coverage_complete=0.
Expected counts serve only to check coverage; they never fill unobserved routes.
Use observed layer_calls as the denominator. These counters alone do not identify
a whole-path bottleneck or an exclusive critical path.
Main owns CPU fixtures, exact environment matching, guards, early close, request
reset, candidate build, bounded 32K diagnostic GEN and parity/fault checks. Only
unchanged upstream bench.sh supplies speed comparisons. This diagnostic is
performance-ineligible.
