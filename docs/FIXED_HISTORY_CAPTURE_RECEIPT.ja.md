<!-- SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# 固定履歴 capture receipt

`tools/fixed_history_capture_receipt.py sidecar.json` は、終了した standalone
native-pack の `--spec-follow --dump-logits` 証拠を読み、既存の
`tools/fixed_history_logits.py` が受け取る receipt JSON を標準出力へ返す。
実行、GPU 操作、モデル tensor の読み込み、速度測定は行わない。
receipt をファイルに保存する場合、証拠 pin は sidecar と同じディレクトリを
基準にする。比較 manifest の各 arm の receipt pin に保存結果を指定する。
比較側の `common` と capture/binary/source/runtime pin は同一にする。

現在、登録済みの実 production producer はない。adapter は整合した fixture
でも `production_capture_qualified: false` を返す。これは意図した拒否境界で、
手入力の `passed: true`、owner ソースの hash、command recipe は実行の真正性を
証明しない。実 controller の生成経路と collector を監査して登録する別の変更が
必要である。`adapter_schema_validated: true` は schema と pin の整合性だけを表す。
producer 名や `evidence_kind: production` の変更でこの境界を越えられない。

## 入力 schema

sidecar は `strata.fixed-history-production-sidecar.v1`。以下の全項目が必要で、
記載されていない項目は拒否する。pin は `{path, sha256}` のみで、SHA256 は
小文字 64 桁、相対 path は sidecar のディレクトリ基準。JSON 重複キーと
非有限値は拒否する。JSON は各 16 MiB、log は各 64 MiB 以下。

- `schema`, `producer`（空でない文字列）, `evidence_kind`（`production` または
  `synthetic-fixture`）, `consumer`, `retire_after`（空でない文字列）。
- `common`: 既存比較契約の `P N vocab max_context endian stride
  position_convention policy prompt_ids follow_ids prompt_ids_sha256
  follow_ids_sha256 tokenizer model`。位置は `P-1+i predicts follow[i]`、stride は 1。
  policy は `bos: prompt-explicit`, `eos: score-all-stop-off`, `mask: all-follow`。
  `tokenizer` は pin、`model` は tensor ではなく下記の小さな identity manifest pin。
- `budget`: `max_vocab max_rows max_capture_bytes` の正整数。capture は
  `8+4*vocab*N` byte。adapter の上限は 1 GiB、vocab は 2097152。
- `owner_source`, `owner_receipt`: pin。owner source SHA は
  `a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f`。
  receipt は実 owner の `commands` 配列を含む保存 object。`command_index` は
  対象 command の非負整数 index。元 receipt の追加項目はそのまま残す。
- `artifacts`: `capture binary source runtime generator verifier prompt follow
  input_manifest stdout stderr provider_evidence` の各 pin。
  source/runtime は source/build/runtime 記録の小さな envelope、generator/verifier は
  実ソース、binary は実行ファイル。model manifest と source の範囲は別。
- `environment`, `identities`, `fault`, `close`: 下記の JSON pin。

`environment` は owner が subprocess に渡した **全有効環境** の文字列 map。
owner receipt と完全一致し、`STRATA_VERIFY_SEGMENTED=1` と
`STRATA_VERIFY_DEVICE_PLAN=0` が必要。`LD_PRELOAD` と probe/bench override は拒否。
環境の他の値と build/runtime envelope の意味は、将来 producer 監査の対象になる。

`identities` は `schema: strata.capture-identities.v1` と
`command_sha256 environment_sha256 artifacts model_files pack_path native_path
profile_path boot_before boot_after started_utc finished_utc`。
command SHA は argv 配列を UTF-8 の compact JSON（区切り `,` と `:`）にした hash。
環境 SHA は environment pin の SHA。時刻と argv/environment は owner receipt に結合する。
`artifacts` は sidecar の全 artifact 名を持ち、各値は
`{pre_sha256, post_sha256}`。capture/stdout/stderr の pre は `null`（未存在）、
その他は post と一致。post は pin SHA と一致する。
`model_files` は空でない配列。各要素は
`{path, bytes, pre_sha256, post_sha256}`、path は重複しない絶対 path、bytes は正整数、
両 hash は一致する。native/profile path が含まれる必要がある。
model manifest は `{schema: strata.model-identity.v1, files: model_files}`。
全 tensor/shard、pack metadata、profile を列挙する責任は実 producer にある。
adapter は巨大 tensor を再 hash せず、pre/post 同一性証拠の結合を検査する。

`provider_evidence` は `schema: strata.capture-provider.v1` と
`provider generator_sha256 verifier_sha256 binary_sha256 runtime_sha256 source_sha256
native_pack device_plan segmented clip_scope`。
provider は `standalone-native-segmented-spec-follow-dump-logits`、native_pack/segmented は
`true`、device_plan は整数 0、clip_scope は
`full-vocabulary-all-follow-no-position-clipping`。各 SHA は artifact pin と一致する。
これは source/native-pack/runtime 範囲と logits の位置選択範囲を分ける記録である。

`fault` は `schema: strata.capture-fault-health.v1` と
`boot_before boot_after before_utc after_utc command_sha256 kernel_complete health_complete
records health_records kernel_evidence health_evidence`。
同じ boot の、command を包含する UTC interval が必要。開始前・終了後の余白は
それぞれ 60 秒以内。complete は `true`。
records/health_records は空配列で、kernel_evidence/health_evidence pin の JSON 配列と
完全一致する。実 collector は interval の kernel fault と device health を漏れなく
分類し、取得失敗・cursor 欠落・boot 変更・fault を拒否する必要がある。
空配列と complete の自己申告だけでは collector の真正性を証明しない。

`close` は `schema: strata.capture-close.v1` と
`command_sha256 capture_sha256 bytes closed_after_utc close_error forced canceled
 deadline_hit budget_hit`。close_error は `null`、最後の四つは `false`、
時刻は command 完了以降、bytes/hash は実 capture と一致する。

## 検証と実 producer に残る仕事

owner command の observation/reap/session-empty/session-anchor は `true`、
ownership は observed、survivors/errors/cleanup は空配列、exit は整数 0 が必要。
PID/start-ticks/session と登録 owner identity が一致する。欠落値を成功で埋めない。

argv は `binary generate` から始まり、明示の tokens-file/spec-follow/dump-logits、
max-new N、max-context、logits-stride 1、spec >=2、native/pack/profile/cache/prefill、
suffix-draft 0、lookup-chain 0、adapt-every 0 が必要。N >=2 とする。
サポートする option の許可リスト外、重複、serve/MTP/EOS/early-stop/prefill-until は拒否。
新しい option は監査した変更で追加する。prompt/follow file の全 ID と入力 manifest を
common に照合する。

stderr の実 segmented-init 行、stdout の prompt/output 全 ID、N 個の decode、
logits-dumped 完了行と owner の log hash/size を照合する。capture は int32 vocab/rows
の 8 byte header と row-major 全 FP32、exact N/bytes、全値有限を検査する。
位置と target は履歴から導出し、入力された成功位置を信用しない。
全 pin を検査後に再 hash する。fixture は模擬 owner hash を test 内だけで差し替える。

実 controller は owner receipt を保存し、command の前後 hash、未存在 output、
完全な model-file inventory、実有効環境、同一 boot、fresh fault/health interval、
close 観測を収集する必要がある。build の source-to-binary 対応、runtime 解決、
collector の完全性、改ざん耐性をレビューしてから producer 登録を行う。
現在の schema 検査はそれらの独立した真正性監査を代替しない。

production capture の閉鎖/route provenance、logits 数値比較、persistent GDN/QSA/PLE
state RESTORE 比較（未実装）、held-out task quality（未実施）は別の結果である。
`quality_qualified` は false のまま、自動採用はしない。生成 prose の完全一致を
quality gate にしない。速度比較は変更していない upstream `bench.sh` だけを使う。
raw capture は指定 consumer、byte budget、retire_after の期間だけ保持する。

schema fixture の検査コマンドは
`python3 -m unittest discover -s tools -p test_fixed_history_capture_receipt.py`。
fixture の合格は GPU capture、production qualification、品質合格を意味しない。
