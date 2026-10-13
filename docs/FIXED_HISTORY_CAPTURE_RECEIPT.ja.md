<!-- SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# 固定履歴 capture receipt

`tools/fixed_history_capture_receipt.py sidecar.json` は、完了した standalone
native-pack の `--spec-follow --dump-logits` capture の証拠を読み、
`tools/fixed_history_logits.py` が受け取る receipt JSON を出力する。推論プロセスの
起動、GPU 操作、model tensor の読み取り、速度測定は行わない。証拠 pin の
相対パスの基点を維持するため、receipt は sidecar と同じディレクトリに保存する。
比較 manifest の各 arm の receipt pin はその出力を指す。common と
capture/binary/source/runtime の pin は一致しなければならない。
[English](FIXED_HISTORY_CAPTURE_RECEIPT.md)

**この standalone schema では**、実際の production producer はまだ登録されていない。
別の SERVE GEN1 schema には Main が確認した K の保存行が登録されている。
この文書の standalone 境界と混同しない。standalone の整合した fixture でも
`production_capture_qualified: false` を返す。手入力の `passed: true`、owner source
hash、command recipe だけでは実行の真正性を証明できない。実際の controller と
collector の監査と登録は、別のレビュー済み変更で行う必要がある。
`adapter_schema_validated: true` は schema と pin の整合性だけを意味する。
producer 名や evidence_kind を production に変えても、この境界は越えられない。

## 入力 schema

sidecar schema は `strata.fixed-history-production-sidecar.v1`。以下の全 field が
必須で、列挙されていない field は拒否する。pin は `{path, sha256}` のみを持つ。
SHA256 は小文字の十六進数 64 桁。相対パスの基点は sidecar のディレクトリ。
重複 JSON key と非有限の JSON 数値は拒否する。各 JSON は 16 MiB、各 log は
64 MiB を上限とする。

- `schema`、空でない `producer`、`evidence_kind` (`production` または
  `synthetic-fixture`)、空でない `consumer` と `retire_after`。
- `common`: 既存比較契約の `P N vocab max_context endian stride position_convention
  policy prompt_ids follow_ids prompt_ids_sha256 follow_ids_sha256 tokenizer model`。
  位置は `P-1+i predicts follow[i]`、stride は 1。
  policy は `bos: prompt-explicit`、`eos: score-all-stop-off`、`mask: all-follow`。
  tokenizer は pin。model は後述の小さな identity manifest を pin し、tensor を
  指してはならない。
- `budget`: 正の整数 `max_vocab max_rows max_capture_bytes`。capture のサイズは
  `8+4*vocab*N` bytes。アダプターの上限は 1 GiB、語彙数は 2097152。
- `owner_source`、`owner_receipt`: pin。owner source SHA は
  `a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f`。
  receipt は実際の owner が保存した commands 配列を含む object。
  `command_index` は対象 command の非負 index。元の owner receipt の追加 field
  も保持する。
- `artifacts`: `capture binary source runtime generator verifier prompt follow
  input_manifest stdout stderr provider_evidence` の pin。source/runtime は小さな
  source/build/runtime 記録 envelope。generator/verifier は実際の source、binary
  は実行ファイル。model manifest の範囲は source の範囲とは別。
- `environment`、`identities`、`fault`、`close`: 後述の JSON pin。

environment は owner が subprocess に渡した **effective environment 全体** の
文字列 map。owner receipt と完全に一致し、STRATA_VERIFY_SEGMENTED=1、
STRATA_VERIFY_DEVICE_PLAN=0 を含む必要がある。LD_PRELOAD と probe/bench override
は拒否する。他の environment 値と build/runtime envelope の意味は、将来の
producer review で確認する必要がある。

identities は `schema: strata.capture-identities.v1` と
`command_sha256 environment_sha256 artifacts model_files pack_path native_path
profile_path boot_before boot_after started_utc finished_utc` を持つ。
command SHA は argv 配列を UTF-8 の compact JSON、separator `,` と `:` で直列化
した hash。environment SHA は environment pin の hash。時刻と argv/environment
を owner receipt に束縛する。artifacts は sidecar の全 artifact 名を含み、それぞれ
`{pre_sha256, post_sha256}` を持つ。capture/stdout/stderr の pre hash は null
（起動前に不存在）、その他の pre hash は post と同一。post hash は artifact pin
と同一。model_files は `{path, bytes, pre_sha256, post_sha256}` の空でない配列で、
path は重複しない絶対パス、bytes は正の整数、前後 hash は同一。native/profile
path を含む必要がある。model manifest は
`{schema: strata.model-identity.v1, files: model_files}`。実際の producer は全
 tensor/shard、pack metadata、profile を一覧化する必要がある。アダプターは大きな
 tensor を再読せず、前後の identity 証拠を束縛する。

provider_evidence は `schema: strata.capture-provider.v1` と
`provider generator_sha256 verifier_sha256 binary_sha256 runtime_sha256 source_sha256
native_pack device_plan segmented clip_scope` を持つ。provider は
`standalone-native-segmented-spec-follow-dump-logits`、native_pack/segmented は true、
device_plan は整数 0、clip_scope は
`full-vocabulary-all-follow-no-position-clipping`。hash は artifact pin と一致する。
これにより source/native-pack/runtime の範囲と、logits の位置選択の範囲を区別する。

fault は `schema: strata.capture-fault-health.v1` と
`boot_before boot_after before_utc after_utc command_sha256 kernel_complete health_complete
records health_records kernel_evidence health_evidence` を持つ。同じ boot と、command
を含む UTC 区間が必要。起動前と完了後の余裕はそれぞれ最大 60 秒。
complete flag は true。records と health_records は空の配列で、kernel_evidence
と health_evidence が pin する JSON 配列と一致する。実際の collector は区間全体の
kernel fault と device health を分類し、収集失敗、cursor 欠落、boot 変更、fault を
拒否する必要がある。空の配列と自己申告の complete flag だけでは collector の
真正性を証明できない。

close は `schema: strata.capture-close.v1` と
`command_sha256 capture_sha256 bytes closed_after_utc close_error forced canceled
 deadline_hit budget_hit` を持つ。close_error は null、末尾四つの flag は false。
時刻は command 完了以後で、bytes/hash は実際の capture と一致する。

## 検証と、実際の standalone producer に残る作業

owner の observation/reap/session-empty/session-anchor は true、ownership は
observed、survivors/errors/cleanup は空の配列、exit は整数 0。PID/start-ticks/session
は登録済み owner identity と一致する必要がある。欠けた値を成功で補ってはならない。

argv は binary generate で始まり、tokens-file/spec-follow/dump-logits、max-new N、
max-context、logits-stride 1、spec >=2、native/pack/profile/cache/prefill、suffix-draft 0、
lookup-chain 0、adapt-every 0 を明示する。N は >=2。許可一覧外の option、重複、
serve/MTP/EOS/early-stop/prefill-until は拒否する。option の追加にはレビュー済み変更
が必要。prompt/follow token file 全体と input manifest を common に照合する。

実際の segmented-init stderr 行、stdout の完全な prompt/output IDs、N decode tokens、
logits-dumped 完了行、owner log の hash/size を照合する。capture は 8-byte int32
vocab/rows header に続く row-major の全語彙 FP32。正確な N/bytes と、全値が finite
であることが必要。位置と target は提示された成功位置を信用せず、履歴から導く。
検証後に全 pin を再 hash する。fixture が mock owner hash を使うのは test 内だけ。

実際の controller は owner receipt を保存し、command の前後 hash、起動前の出力
不存在、完全な model file 一覧、実際の effective environment、同一 boot、新しい
fault/health 区間と close 観測を収集する必要がある。producer 登録前に、source と
binary の build binding、runtime 解決、collector の完全性、改変への耐性をレビュー
する。schema check はこの独立した監査の代わりにならない。

production capture の closure/route provenance、logits 数値比較、永続 GDN/QSA/PLE
state の RESTORE 比較（未実装）、held-out task quality（未評価）は別々の結果。
quality_qualified は false のままで、自動採用しない。生成 prose の完全一致は品質
 gate ではない。速度比較には変更していない upstream bench.sh だけを使う。
raw capture は宣言した consumer、byte budget、retire_after 期間のためだけに保持する。

schema fixture command:
`python3 -m unittest discover -s tools -p test_fixed_history_capture_receipt.py`。
fixture 成功は GPU capture、production qualification、quality を証明しない。

実際の正常終了済み SERVE GEN1 行には、別のレビュー済み一行 schema を使う。
[SERVE first-row adapter](SERVE_FIRST_ROW_CAPTURE_RECEIPT.ja.md) と
[English](SERVE_FIRST_ROW_CAPTURE_RECEIPT.md) を参照。standalone の意味は変更しない。
