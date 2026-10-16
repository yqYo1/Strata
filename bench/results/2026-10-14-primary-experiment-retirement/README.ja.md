# 検証ブランチの保存と整理

本家はNiko1221/Strata、主作業は `feature/sycl`。XeStrata由来の枝を主実装として扱っていた
運用を修正した。終了した8本のローカル枝・worktreeと、対応する3本のremote枝を削除した。
forkの枝はローカル・remoteとも `feature/sycl`、`main`、比較用のbulk実験1本だけとなった。
必要なcompiler、ggml、fixtureを持つ別cloneの依存worktreeは今回の削除対象にしていない。

## ソース・レポートの復元

5つの未コミット実装案、計73ファイルをsourceのみのcommitに保存した。
8本全ての元HEADと保存commit、元レポート、以前保存した104本のtipは、
主の保存merge `4526800257fe49c0db02a414f21b38dac2586e54` から到達できる。
これは未採用のXe由来実験の履歴保存であり、主の推論コードの採用ではない。
保存前後で主のengine treeは一致した。

各枝の元HEAD、保存commit、ファイルhashは [preservation.json](preservation.json)。
例えばCPUサービス診断は次で復元できる。

```sh
git show 1cd442981faf137e4370fffdc9d73ab91cce8c7a:include/strata/core/decode_service_budget.hpp
```

元レポートの復元例と削除後のsource照合、Git接続性検査も保存した。
古い調査や失敗のstatusを書き換えていない。新しい調査とMainの訂正は
[主実装点検](../2026-10-14-primary-strata-source-audit/README.ja.md) を参照する。

## 削除した成果物と保持する比較先

終了した再build可能な出力と重複metadataなど、論理容量750,271,455 bytesを退役した。
各path、hash、理由は `artifact-deletion-manifest.jsonl.gz` に格納した。圧縮前後のhashは
[retirement.json](retirement.json)。symlinkの参照先は削除していない。
元ソース・レポートはGitに保存し、重複するbuild出力の全コピーは保持しない。
削除計画を先にcommit/pushし、測定・build・testと共通lockで直列化した。
対象worktreeを参照する生存processがないことを確認してから削除した。

残す比較枝 `feat/layer-major-bulk-upload-20261013` は、部品のsource donorと既存測定の
参照先であり、新しい実装の分岐元ではない。最初の主側lease/診断移植と同条件比較の区切りで
再評価する。新しい実装枝が必要なら、その時の `feature/sycl` から作る。
旧スキル用枝のmetadata試行は主側の修正に置き換え、比較worktreeもcleanに戻した。

## まだ終えていないこと

`main` の0.1.39機能を、主の0.1.38系推論へ適合させる作業と、fork defaultの更新は未完了。
mainの履歴も保存mergeに含まれるため、通常mergeの「already up to date」は機能移植の証拠に
ならない。保存前の両tree、元共通祖先と7ファイルのmerge競合を点検レポートに固定した。
機能の適合ではその固定treeを使い、主の最適化とmainのCLI/server機能を双方維持する。
その検証後に自身のforkのPRでdefaultを更新し、続いて本家のremote更新確認・評価を行う。
この区切りでは新しい上流を確認したとも、推論採用や性能目標を達成したとも報告しない。
