# PR #6 忠実復元と安全性監査

この最初のコミットは source-only unsafe anchor。歴史ソース197ファイルをそのまま復元した。
build・test・GPU実行は未実施、>600 tok/sは再測定していない。
復元後の監査・修正・現行比較・Main検証計画を後続コミットで追記する。

元ソースとdependencyの全blob/SHA256、resolver選択は `historical-closure.json`。
外側CMakeの互換変更とcommit pinは `manifest.json` と `validation/pr6-restore/README.md`。

初回 `git diff --cached --check` は歴史ソース由来の trailing whitespace 6件を検出した。
blob一致を優先し、元バイト列を変更していない。新規wrapper/記録には該当なし。
