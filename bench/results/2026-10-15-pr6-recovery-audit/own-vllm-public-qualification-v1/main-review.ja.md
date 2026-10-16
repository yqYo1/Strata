# R846 Main 専用 PUBLIC Gemm qualification — UNRUN

このディレクトリの新規3ファイルだけを作成した。コンパイラー、CMake/Ninja、ビルド、テスト、Python syntax check、candidate import、GPU、profiler、library load、process inspection、Git は実行していない。worker が読んだのは既存の規約、review、controller、fixture、receipt、build recipe と provenance のためのファイル識別情報である。Main は controller.py 全文を読み、active な quality job が正常に閉じた後で排他的に実行する。Main の実行レビュー・承認済みという意味ではない。

R833 fixture は SHA `7e1d5a45526eb367e1062d46135c2e99a8fa3a12f4e3db9684de9f526b204716` の既存ファイルを直接コンパイルする。変換、oracle のコピー・書き換え、tolerance 変更、case 削減はない。独立 FP64/dyadic oracle、4 入出力形式、K96 cancellation、厳密 gradual-underflow、T/N tail、pitch/full Y rectangle、registered span、checked drain、Gemm destructor 後の最終 marker が元のまま残る。既存 component/68-case/model test の代替ではない。

compiled base は `6354f0b4ff3731466795410e11ca9ef785c9d326`、build-v2 receipt は SHA `990635731be1198706abeb682817fdab247f8f51d31bab971d3d4f4db53cb639`。この receipt の build_path が v1 名のディレクトリである点は意図どおりであり、B/main-vllm-prefill-build-v1/build の現行 archives と final native image/host slice を使う。古い9c ABIの archives はリンクしない。build receipt 内の314 historical/provenance pins も照合するが、リンク候補とは区別して manifest に残した。現行13 products の SHA と432 source pins、header tree6組、toolchain/runtime/library、モデルmetadata/stat identity、CMakeCache/build.ninja/rules.ninja/compile_commands、prelink receipt を固定する。model shard の同一性は既存どおり bytes+mtime_ns であり full shard hash ではない。

compile/link argv は現行 build.ninja の sycl_vllm_native16_test stanza と rules.ninja を起点に固定した。新しい fixture/object/output/dependency path だけを置換し、`-fno-fast-math -ffp-contract=off -Werror -fsycl`、defines/includes、generic 最終リンク、native final image+host slice の順序、system library の元の順序を保持する。コンパイル・リンクの cwd、TMPDIR、cache と出力は新規 B/main-vllm-public-hook-v1 内。元 object/binary/source を変更するコマンドはない。official Intel LLVM v711 install は歴史的に XeStrata 配下に置かれているが、その install 外の engine/helper/test は利用しない。外部実装コードの追加・コピーはない。新規 controller は当 repository LICENSE (MIT) に従い独立に作成し、既存 Strata direct_owner と R816 の pin/tree/receipt/driver/model-owner helpers のみ Main 時に hash を確認して import する。R816 の9c用 build/qualification workflow は呼ばない。

CLI は最新 normally-closed same-boot PASS GPU receipt と SHA を必須とする。最新の選択は Main の責任で、active/private record の探索、FAIL 続行 override、quality failure override はない。通常の binary_sha256 と新しい four-arm screen の binding.binary の両方について、同じ build/source/binary を検証する。quality 未採用という false フィールドを quality 合格に読み替えず、component 用の healthy prior としてのみ使う。別 queue の既存 FAIL や build-v1 FAIL は保持され、この controller の PASS から上書きしない。

Main の実行コマンドは次のとおり。`R846_MANIFEST_SHA256` は今回の納品 digest を全文照合して設定し、`LATEST_PASS_RECEIPT` と `LATEST_PASS_SHA256` は active quality job 閉鎖後の実際の最新 PASS を明示する。`RETENTION_REVIEW_UTC` は実行時から48時間以内の UTC deadline。これらが未設定なら shell が停止する。manifest/report の相互 hash cycle を避けるため、manifest SHA の値はこのファイルに複製しない。

```sh
/usr/bin/python3 /home/yayoi/orca/workspaces/Strata/sycl/bench/results/2026-10-15-pr6-recovery-audit/own-vllm-public-qualification-v1/controller.py \
  --manifest-sha256 "${R846_MANIFEST_SHA256:?set reviewed delivery manifest SHA}" \
  --prior-receipt "${LATEST_PASS_RECEIPT:?set actual latest closed PASS receipt}" \
  --prior-sha256 "${LATEST_PASS_SHA256:?set its exact SHA256}" \
  --root /home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-vllm-public-hook-v1 \
  --retention-review-utc "${RETENTION_REVIEW_UTC:?set explicit UTC within48h}"
```

canonical root 以外、既存 root/result、symlink root/parent、使用中 lock は拒否する。共有 `B/owned-v0141-measurement.lock` を全工程で保持し、同一 uid の model/fixture cmdline と実 executable identity を確認する。fresh boot、xe、Intel PCI、devcoredump、各操作の前後 kernel journal を検証し、fault は累積する。各 native child の環境は exec 前に別の immutable map として設定する。compact smoke → wide smoke → compact full → wide full の4独立プロセス。最初だけ flushed UR tracing+parameter validation、残りは warning validation。実際の route banner と UR launch trace、PUBLIC_HOOK_BEGIN/COUNTS/最終 SUITE_PASS、全形式の dyadic/edge/chain/cancellation/reject evidence を検証する。GPU selector と pinned public dispatch の無 fallback 契約を使い、CPU GEMM 経路は追加しない。

whole900秒のうち855秒を通常作業、45秒を閉鎖用に確保する。各 compile/link180秒、native120秒を残時間で短縮し、compile/link RSS8GiB、native4GiB、全 text64MiB、private512MiB。期限・memory・log 超過、unsupported、skip、非zero終了、cleanup signal、残存 owner、失敗 numerical case は PASS にならない。Owner は PID/start/session/pidfd を記録して自身の子だけを管理し、normal reap/session empty を要求する。全 pin は操作前後に再照合し、最終失敗時にも確認する。guard/pin/journal 自体の失敗も記録する。900秒上限直前で閉鎖・記録が完了できない場合は private FAIL/incomplete を保持し、完了と称さない。

Main 実行後の compact receipt は A/own-vllm-public-hook-v1.json、新規 private record は B/main-vllm-public-hook-v1/record.json。各 argv/env/binary/hash、各 case の実 stdout と strict edge bits、positive48/52、public calls52/56、160拒否を保持する。失敗 native stdout/stderr の位置と bounded failure tail は後続 journal の結果と分離する。public receipt は exclusive create で1回だけ保存し、既存 receipt は書き換えない。

保持 consumer は Main R846 の実 public Gemm component qualification review。first compact trace はその API/完了の抽出証拠を Main が照合するまで必要で、総text64MiB以内。qualification閉鎖時か次のGPU/build/retention境界の早い方、かつ指定UTC期限までに見直す。成功full trace等の削除は今回行わず、後日の Main が所有権閉鎖・抽出証拠・path/hash/reason manifest を確認して直列に実施する。performance、general quality、physical262144、RESTORE、当 private executable の actual ISA/occupancy は全て未合格で、default OFF を維持する。
