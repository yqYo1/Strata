# Layer 26 phase audit v1（非適用実装案）

本ディレクトリだけを書いた。製品 source は変更せず、compile/test/GPU/Git/service/cleanup は
行っていない。`source.diff` は既存2ファイルへの局所追加と新しい診断 header だけである。
`regen.py` は固定した source hash と一意な anchor を確認して diff を再生成する。
source の新 snapshot を overlay して置き換えない。Main の測定が閉じた後に review・適用・検証する。
本作業では project-only strata-upstream-sync と git-workflow の指示を読み、今回の書込範囲と
実行禁止を優先した。外部ソースや私用 Codex データ、envscan、禁止 record は読んでいない。

既存の新実測は b6838d286a6ec00b0d92c7ce8f8480c81511c4ab823e400273c52b2585c55aa6 binary、
34,603 actual input / 34,602 batched、FP32、chunk 4,096、fixed GPU rows 4,096 対 auto 24,576。
19 selected rows の layer 0–25 は一致し、layer 26 / row 24,575 が first selected difference、
後半6 selected rows が不同。各 run の保存・読み戻し memcmp は一致し GPU fault 0 だった。
19 selected 外の入力と GDN state は一致未証明であり、GDN 原因確定ではない。
この実装案には新しい実行結果がない。旧 FAIL は変更しない。

現在の resolver は root `CMakeLists.txt` → `cmake/sycl_backend.cmake:103` の
`strata_prefill` → root `src/prefill/prefill.cpp` → `layer_major.inc`。
既存 build receipt の compile proof もこの absolute source と SYCL / no-fast-math /
ffp-contract=off を示す。root CUDA generator から機能や推論経路を推定していない。

## 採取する3組

常に layer 26 のみ。CSV の最大6行のみを採取し、他 layer と非選択 chunk はコピーしない。
既定は relative rows `16384,24575,24576,32767,32768,34601`。
16,384 を前側 control、34,601 を末尾として残すため、既知不同6行全ては選んでいない。
CSV を変える場合も両 arm で同じ relative row 順を指定する。絶対 row は pos0 を足して記録する。

| 組 | 採取する raw | 実際の採取位置 |
| --- | --- | --- |
| 1 GDN | `m.mixed_h` FP16、`m.bo` FP32 | attention HC mix 後・GDN projection 前 / ssm_out projection 後 |
| 2 HC / route | attention write 後 `m.R` FP32、FFN `m.mixed_h` FP16、`m.mixed_bf` BF16、存在時 low BF16、router logits 512 FP32、IDs 10 int32、weights 10 FP32、host grouping 10 entries | attention GRwrite（fused norm を含む）後 / FFN mix 後 / route 後・shared expert 前 / host grouping と order 完成後 |
| 3 MoE / write | combine 後 `m.bo` FP32、最終 `m.R` FP32 | MoE combine 直後 / FFN GRwrite と適用される cvec 後 |

`mixed_h` は FP32 モデル内の既存 projection 入力形式であり、FP16 saved residual storage の
採用ではない。saved storage は FP32 だけを受け付け、FP16 指定では実行前に明示拒否する。
数学 kernel、weight、演算順の source、FFN/router/grouping の構築を変えない。
qkv / gate / hbuf / y_h、expert ごとの GU/DOWN、全 row route、GDN state の追加はない。

grouping は各 selected token の10 entriesを rank 順に、
`{expert, slot, src, cnt, off, order_index, group_local}` の int32×7 として保存する。
`src` は chunk 内 token index、slot/off は grouped row index、order_index は既存 expert 昇順
リストの index、group_local は slot-off。expert ID 範囲、slot/cnt/off 範囲、src 対応、
selected top10 の一意性、order 内存在を確認する。全512 expert の cnt/off/order は dump しない。
これは host grouping の観測であり、device slot/src 全体との追加比較ではない。

各テンソルは producer の直後に D2H し、raw に書いてから同じ owner の scratch を再利用する。
特に GDN bo は次半の上書き前、FFN mixed は次 layer の mix 前、router は MoE scratch の
利用・次 chunk の上書き前に捕る。`m.region` の GDN 内容を MoE 後に読む案ではない。

## 所有と有限予算

run_layer_major の issuer が `unique_ptr<Layer26PhaseAudit>` を所有する。
D2H destination は header 内の 40,960 B scratch member。grouping の280 Bも member に置き、
stack 一時領域を非同期コピー先にしない。owner を Restore と既存 checked queue lease より前に
宣言し、既存境界 audit と同じ Copy callback を使用する。Copy は m.cs へ D2H を submit し、
submit エラーでも同じ queue の checked drain を実行する。早期 return/exception では先に
lease が drain し、退役が証明できなければ owner が生きたまま abort する。次に borrowed
field/callback を復元し、最後に owner を破棄する。worker writer・別 copy queue は追加しない。

| 追加 capture の契約 | 上限 / 算術 |
| --- | --- |
| raw disk | 8,388,608 B / run |
| text disk | 1,048,576 B / run |
| host admission | 1,048,576 B / run を別 reservation として追加 |
| 実 raw、最大6行、BF16 low 有 | 125,288 B/行 ×6 = 751,728 B |
| 実 raw、最大6行、BF16 low 無 | 120,168 B/行 ×6 = 721,008 B |
| record coverage | low 有72 / 無66 records、行数が6未満なら比例 |
| added D2H/check-drain | low 有66 / 無60回、行数が6未満なら比例 |
| extra explicit GPU allocation | 0 |

raw を全て RAM に保持せず、stream へ書く。1MiB host reservation は scratch、固定 metadata、
FILE buffers と有界4095文字 path を含む。この reservation は model/upload/residual の
host_need に累積し、既存 audit も有効ならその4MiB reservation も加算して admission を行う。
`conversation_memory_admit(..., host_need + optional_boundary_4MiB + phase_1MiB, 8GiB)`。
先に既存 owner を確保した場合の再 admission は保守的に再度数える。8GiB floor を減らさない。
既存 runtime が D2H を扱うための内部 staging 確保量はこの source から測定できず、
既存 VRAM/RAM headroom と Main の実行監視も必要である。

既存全層 boundary audit の raw48MiB / text8MiB はそのままで、共用予算にしない。
両方有効時は各 run の raw56MiB / text9MiB が独立上限の和、追加 host reservation の和は5MiB。
2-arm 合計上限は raw112MiB / text18MiB。wrapper のその他 stdout/stderr 予算も別に定める。
出力は別名とし、log と raw を各々 O_EXCL / O_CLOEXEC / mode0600 で作る。
一方の create だけ成功した場合も自動削除しないので、次回は新しい path を使う。
raw 書込・log flush のエラー/上限超過は incomplete。完了時 raw fflush を確認する。

switch 未指定または `0` は default OFF。新しい host allocation、D2H、wait、ファイル作成は
発生しない（既存経路に元からある wait は残る）。追加は env 読み取りと null pointer 分岐だけ。
OFF時は row/path も parse しない。実 defaultOFF の検証は Main の build/model gate で行う。

## metadata と完了判定

log は各 raw span の `tensor/layer/row/absolute/chunk/start/count/dtype/elements/bytes/offset/
source_offset/source` を保持する。offset は `.raw` 内の byte offset、source_offset は元 row tensor
内の byte offset。host_grouping のみ source_offset は元 host IDs 配列の当該10-entry開始位置で、
他の grouping fields はそのIDs/slotからの selected gather である。header は host byteorder を記録する。

各 selected row は上表の順に全テンソルを1回だけ必要とする。欠落・重複・順序逆転・不正 chunk
は拒否する。low 無しの skip だけを明示的に許す。全 run の final drain と callback coverage
確認後、`status=complete coverage=complete memcmp=not_compared adoption=unqualified` を書く。
途中で layer 26 の全値が捕れても、後続 layer が失敗した run は complete にならない。
`complete()` の二重呼出しも拒否する。完了は実行 coverage であり、数値一致ではない。

`compare.py FIXED_LOG AUTO_LOG` は2つの metadata/coverage/extent を検証し、raw byte strings を
実比較する。first differing byte と各 row/tensor の不同を返す。exit0=selected raw equal、
exit1=selected raw different、exit2=incomplete/invalid。SHA256 は fixture locator のみで、
hash一致を同値判定に使わない。raw の符号付き0、NaN payloadなども区別する。
外側の Main receipt に source/diff/binary/compiler/runtime/input hash、各 arm env、実 prefix/
chunk、PID/start/boot、deadline、exit、fault window、出力の hash を別途結び付ける。

## Main の適用・検証手順（未実行）

1. 現在の actual-format GPU run を終了し、既存 measurement lock を確保する。
2. `python3 -B .../layer26-phase-audit-v1/regen.py`。pins が違えば自動更新せず局所 rebase review。
   `source.diff` を Main が review/適用する。本 generator 自身は適用も Git 操作もしない。
3. CPU fixture と comparator negative cases を実施する。手順は `cpu-fixture-plan.ja.md`。
   次に実 SYCL build、既存 CPU boundary/lease fixtures。全て Main が直列に実行する。
4. 新 binary で短めの layer-major input を使い OFF / phase-only ON / boundary+phase ON を比較。
   n>chunk、rows は小入力内へ明示変更。FP32出力 bitwise、row coverage、fault0、OFF 出力なし、
   FP16 storage と非layer-major/短入力の明示拒否を確認する。performance gate と混ぜない。
5. 新 binary 同士で既存34K recipeの FP32 fixed4096 / auto24576 を同条件比較する。
   STRATA_PREFILL_PHASE_AUDIT=1、上記共通 CSV、各 arm 別の新しい PHASE_PATH を追加する。
   旧 b6838... の capture と新 binary の数値を同一 gate としない。既存 boundary audit は
   同時に有効化可能だが、独立予算と intrusion を receipt に記録する。
6. `python3 -B .../compare.py FIXED_PHASE_PATH AUTO_PHASE_PATH`。
   coverage と equality、exit/fault、既存 boundary copy 結果、採否を別項目で保存する。
   調査が閉じたら retention 規則に従い path/hash/reason を記録し、必要な異常 fixture を残す。

実 command は既存 `check_boundary_audit_v1.py` の exact input/環境/監督を引き継いで Main が
新 recipe として記録する。本案からサービス停止・外向け公開・既存出力上書きを指示しない。

## 大枠の読み方と限界

| 最初に selected raw が変わる境界 | 次に絞る範囲 |
| --- | --- |
| GDN input mixed_h | layer26 の attention HC read、または前段入力 |
| GDN out bo（selected mixed_h は一致） | GDN projection/conv/recurrence/output の大枠。未捕捉 BF16 alpha/beta 入力、全先行 row、carried state を含む |
| attention write R（bo は一致） | attention HC write/inject の大枠 |
| FFN mixed（R は一致） | HC FFN read/mix の大枠 |
| router logits（selected mixed high/low は一致） | router projection の大枠 |
| IDs/weights（logits は一致） | softmax/top10/weight の大枠 |
| grouping（selected IDs/weights は一致） | 全 row route に依存する grouping。未選択 route 差でも cnt/off/slot が変わり得る |
| MoE bo（上の selected 値は一致） | routed/shared expert と combine の大枠。個別 GU/DOWN 数値の一致までは不明 |
| final R（MoE bo は一致） | FFN HC write/inject/cvec の大枠 |

どのケースも selected 観測の切分けであり、機構の原因確定や全入力一致ではない。
全 token の residual/GDN入力、recurrent state、GDN alpha/beta の入力、expert中間は未観測。
一方の selected GDN input が一致しても、それ以前の非選択行が state を変えた可能性が残る。
追加 D2H/check-drain は in-order queue の実行をさらに直列化し、race を隠し得る。
診断 ON で差が消えても元の FAIL を解消としない。3組は最大11個の device row tensorを
各別に drain するもので「追加 synchronization が3回だけ」という意味ではない。

GDN state capture は本案に実装しない。後段で必要と判断した場合だけ別の明示 opt-in を設計し、
matched actual chunk boundary で1 snapshot/run 以下（3,268,608 B）とし、別の host reservation/
raw増分/所有者/位置・next review を先に記録する。row24576直後は multirow recurrence 内に
あり、本 patch に state checkpoint を隠して加えない。通常 phase 案に常時混ぜない。

raw の consumer は Main の layer26 selected coarse paired comparator。owner は Main の本比較。
next review は最初の paired comparison 完了時。性能・品質・physical262144・RESTORE・修正採用を
本案の完成や selected raw一致から認定しない。
