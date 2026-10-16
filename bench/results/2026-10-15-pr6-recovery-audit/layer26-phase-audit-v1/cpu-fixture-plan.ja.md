# CPU fixture plan（全て Main 未実行）

`cpu_fixture.cpp` は GPU/header依存を持たない stand-alone 提案 fixture。
新 header template を直接 include する。製品へ差分適用する前でも Main が測定 lock 下で
compile/run できる。現在は compile も実行も行っていない。assert を有効にする。

Main は既存 actual-format run の終了後に専用出力 directory を用意し、次を直列に実行する。
`proposal` と `cpu_root` は task 専用変数の例で、出力は新規 private directory を指定する。

```sh
c++ -std=c++20 -O2 -UNDEBUG -Wall -Wextra -Werror -I "$proposal" "$proposal/cpu_fixture.cpp" -o "$cpu_root/phase-fixture"
"$cpu_root/phase-fixture"
c++ -std=c++20 -O1 -g -UNDEBUG -Wall -Wextra -Werror -fsanitize=address,undefined -fno-omit-frame-pointer -I "$proposal" "$proposal/cpu_fixture.cpp" -o "$cpu_root/phase-fixture-asan-ubsan"
"$cpu_root/phase-fixture-asan-ubsan"
```

fixture は mkdtemp0700 下へ新規 log/raw を生成し、最後にその directory を表示する。
ファイルを自動削除しない。成功 evidence の抽出後、Main が GPU/build/test と直列の cleanup
境界で path/hash/reason を記録して退役させる。

| 条件 | fixture code に含む観測 | 実行結果 |
| --- | --- | --- |
| low有/無、2 selected rows、2 chunks、pos0=100 | 全11/12 tensor 順、raw を独立組立てした期待 byte列と実比較、コピー回数20/22、metadata完了 | 未実行 |
| FP32 raw bit patterns | +0/-0/subnormal/±Inf/NaN payload を未変換で保存 | 未実行 |
| output所有 | log/raw mode0600、logとraw各々O_EXCL衝突、raw sentinel非上書き | 未実行 |
| CSV/shape | 空、trailing comma、negative、範囲外、duplicate、7行、非数、int64 overflow、n/pos/chunk上限 | 未実行 |
| sequence | skipped tensor、duplicate tensor、incomplete complete、double complete、不正 chunk | 未実行 |
| 採取範囲 | layer25 と非選択 chunk は copy callback0回 | 未実行 |
| failure owner | callbackがpending dstを保存してfalse後、owner存続中にfake retirement書込ができる、complete不可 | 未実行 |

fake retirement は helper内scratchの寿命だけを試す。実 D2H の完了、実異常時の outer lease、
same-queue retirement は証明しない。実 source の Restore/lease 順 review と既存
`tests/cpu/draft_uploads.cpp` を併用し、その後 Main の実 SYCL gate を別に記録する。

適用前に Main が追加する negative fixture 案：

- `capture_grouping` の selected e/slot/src/cnt/off/order不正、top10 duplicateを各別に拒否する。
- selected nullptr/empty Copy、raw/log書込失敗（専用fault-injectionまたはRLIMIT_FSIZEとSIGXFSZ処理）で
  完了行を出さず、最後の必要fixtureを壊さない。`/dev/full` など既存pathをO_EXCLで開く方法は使わない。
- `STRATA_PREFILL_PHASE_AUDIT=0` で不正ROWS/PATHを指定しても新診断のparse/I/Oがない。
  これは helper constructor単独ではなく dispatcher/統合側で確認する。
- phase ON と FP16 saved storage、非layer-major、n<=chunkは明示拒否する。
- 最大6行と末尾partial chunkのcaptureで751,728 B/721,008 Bの式、全record coverageを確認する。

`compare.py` の Main fixture 案（GPU不要、未実行）：

1. `case-0` を自分自身と比較：coverage_complete=true / raw_memcmp=equal / exit0。
   low有り `case-1` も自分自身で同様。これは同値判定pathのfixtureでありarm間の測定ではない。
2. 新しい private directory にlog/rawを複製し、rawの1 byteを反転：coverage_complete=true /
   raw_memcmp=different / そのtensorのfirst_different_byte / exit1。SHAを同値判定に使っていないことを確認。
3. raw末尾切断、余分byte、record欠落/重複、offset/shape/dtype/row改変、footer欠落、low mismatch、
   異なるselection/schema、byte budget超過はcoverage_complete=false / raw_memcmp=not_compared / exit2。
4. copy-failure fixtureの未完了logは数値FAILに分類せず、coverage incompleteとして拒否。

1MiB/8MiB上限のI/O failure injectionはproposalの固定正常geometryから自然には到達しない。
最大選択数でもraw約0.72MiBに収まるためで、coverage検証とbudget-check source reviewを分ける。
上限到達のため実tensor数やshapeを緩める案にはしない。

CPU gate と actual model gate の receipt、追加待機の侵襲、original-auto FAIL保持、採否を
区別して Main が保存する。fixture sourceの存在やregen成功をPASS結果にしない。
