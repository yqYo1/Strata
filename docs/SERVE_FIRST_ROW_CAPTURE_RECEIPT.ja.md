# 実際の SERVE GEN1 先頭行の取り込み

このアダプターは、Main が実機で確認した、正常終了済みの一行だけを
`tools/fixed_history_capture_receipt.py` から取り込む。比較は
`tools/fixed_history_logits.py` の別スキーマで行う。既存 standalone の
follow dump、二行以上の producer admission、空の `REVIEWED_PRODUCERS`
の意味は変更しない。[English](SERVE_FIRST_ROW_CAPTURE_RECEIPT.md)

現時点の登録は `phase-gated-gen1-b4f7b5ec-32k` のみ。元の source revision
`6035c4f92bf87362d7e55051ae1375b9befba89e` と五つの変更済みファイルは、
producer commit `b4f7b5ec1d7583de52f0e0dbcf181d461591cf17` に固定される。
対象は保存済み `request-2.bin`、32768 トークンの fresh GEN1、位置 32767、
248320 語彙の IEEE FP32 little endian、一行 993288 bytes。
SHA256 は `d3903d0a0eefc77dbb3493ed413b007104eb365944e67df641684f00d9297c54`。
B570 10 GiB / Ryzen 5600X / RAM 128 GiB の元の実機結果を取り込む。
新しい実機測定や品質結果ではない。

## 契約

- `strata.serve-first-row-sidecar.v1` は producer、receipt、publication、retention、
  capture、fixture、source_root、source_commit、request_index、model_files、
  consumer、retire_after を必須とする。例は
  `tools/serve_first_row_example.json`。pin は path と SHA256 のみ。
- 登録 JSON 自体の hash はアダプターの source に固定される。登録ごとに
  Main がレビューした実際の receipt、publication、retirement の hash、
  commit/tree、283 個の src/include/cmake/build entry の hash、五つの source
  pin、model/provider/binary の identity、保存行と request index を束縛する。
  新しい producer を呼び出し時に登録するオプションはない。P の layer-major
  candidate は未登録。将来の追加は実機 qualification 後の明示的な Main の
  source review と registry/hash の変更が必要。
- 元の immutable receipt の PASS、complete、inactive、正常 QUIT、signal 不使用、
  owner exit 0、reaped/session empty、READY の PID/start、完全な前後 health と
  kernel 観測、同一 boot、fault filter と件数を検証する。埋め込み stdout/stderr
  は元 owner log hash と照合する。成功 raw log を再要求しない。
- raw spec=4、effective T=1、prefill capacity 8192、phase/head/scratch/gated、
  source_row=0、reuse=0、MTP/lookup/batch 無効を元の実際の証拠に束縛する。
  COPIED の後に COMPLETE があり、出力一個と DONE があることを検証する。
- 全 conditioning fixture の SHA256 は
  `5284d8fa53d1ef28f6ba916a629faf901c890bc0c669bf986ee04ed72aa2c3ff`。
  65536 個の token IDs の prefix から、実際の `GEN 1 temperature=0 ...` packet
  の hash を再構成する。比較 target は固定 fixture[P]。生成された文字列や
  自己 argmax は品質の合否基準にしない。
- native row format に magic はない。LE int32 `(vocab, 1)` のヘッダー、正確な
  byte count、全語彙の finite 値、hash を検証する。fixture と row は各 2 MiB、
  JSON は 16 MiB の上限。一行のみを読む。model tensor は読まない。
- 退役済み request-1 と `.partial` alias は要求しない。選択行は retirement の
  deletion に含まれていてはならず、現在の consumer と次の review 点が必要。
  退役を失敗に変更しない。検証したファイルを最後に再 hash する。

## Main が順番に行う検証

実装 handoff 時点で以下は **未実行**。

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_serve_first_row_capture_receipt.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_fixed_history_capture_receipt.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_fixed_history_logits.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/fixed_history_capture_receipt.py tools/serve_first_row_example.json
PYTHONDONTWRITEBYTECODE=1 python3 tools/fixed_history_logits.py tools/serve_first_row_self_comparison.json
```

CPU fixture は bad header/endian/shape/truncated/nonfinite、request position/packet、
model/provenance、active/unreaped owner、fault/missing coverage/boot、consumer、
retired capture、未登録 producer と不正な authoritative hash の拒否を検証する。
実機 row の取り込みは Main が独占枠で別途行う。

比較 manifest は `strata.serve-first-row-comparison.v1`、baseline と candidate に
sidecar の path/hash、consumer と retire_after を持つ。両 arm を実際に再取り込みし、
全 conditioning identity、numeric argv、effective environment を一致させる。
現在は同じ K 行の self comparison だけで経路確認できる。別 candidate の実機
qualification はこれから必要。同じ行でゼロ delta が出ても候補の品質証拠ではない。

出力は target log probability/NLL、rank、top1/top2 margin、双方向 KL の一行の
数値だけ。corpus quality、zero ability loss、256K physical extent、RESTORE、
速度、採用を認定しない。quality/performance/full-context/restore flags は false。

## 現在の pairing 制限

現実装は numeric argv 全体と effective environment 全体（capture directory を除く）の
完全一致を要求する。これは現在の K 同士の確認の契約であり、将来の K 対 P の
layer-major 比較を既に受け付ける契約ではない。例えば source が定める暗黙の既定値
700 と明示的な 700 は、effective value が同じなら数値設定の変更ではない。
現在の文字列の完全一致だけでは、この同値性を正規化できず拒否し得る。

将来の candidate 比較には、Main が source をレビューした treatment normalization と
許可差分の契約が必要。既定値は各 arm の固定 source と実際の effective 証拠から
解決し、LM と RGPU の差は、事前に定めた比較対象の処置として明示的に管理する。
比較対象以外の数値設定、conditioning、model/provider の統制を保つ必要がある。
呼び出し側が任意の argv/env を除外する仕組みでは代用しない。

この契約変更、P の登録、K 対 P の比較は今回実装していない。P の actual model
経路は未 qualification で、P の保存 row は存在しない。

## Main の実行証拠

上の「未実行」は最初の source handoff 時点の記録。以後、Main の CPU v1 receipt は
fixture 成功と actual import の `qualified effective route` 拒否を記録し、失敗のまま
保持されている。原因は誤った `STRATA_PHASE_PREFILL_GATED_ALIAS` の名前で、実際は
`STRATA_PREFILL_GATED_ALIAS`。Main が名前を修正し、実際の reviewed receipt を使う
route/wrong-name regression を追加した。Main の v2 receipt
`main-serve-first-row-adapter-cpu-root-v2/record.json` は、三つの CPU suite、actual K 行の
取り込み、K 同士の比較の PASS/complete/inactive を記録する。この文書更新の担当は
これらを実行していない。v2 の成功は P の qualification や品質評価を意味しない。
