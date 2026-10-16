# PR #6 忠実復元と安全性監査

この最初のコミットは source-only unsafe anchor。歴史ソース197ファイルをそのまま復元した。
build・test・GPU実行は未実施、>600 tok/sは再測定していない。
復元後の監査・修正・現行比較・Main検証計画を後続コミットで追記する。

元ソースとdependencyの全blob/SHA256、resolver選択は `historical-closure.json`。
外側CMakeの互換変更とcommit pinは `manifest.json` と `validation/pr6-restore/README.md`。

初回 `git diff --cached --check` は歴史ソース由来の trailing whitespace 6件を検出した。
blob一致を優先し、元バイト列を変更していない。新規wrapper/記録には該当なし。

## 修正A: draft upload のhost source寿命

歴史 `prefill.cpp:1321-1343` はループ内の `bh[4]` と次の反復で書き換える
`tk` を非同期memcpyへ渡し、完了waitはループの後だけだった。複数batchでtokenを
早く上書きし、boundsは単一batchでもscope終了後に読む可能性がある。in-order queueは
CPU側の書換え・scope終了を待たない。歴史の timed >600 tok/s はMTPなしのため、
この欠陥と当時のGPU faultの関係は未証明。

63326f98の `include/strata/prefill/draft_uploads.hpp` と `tests/cpu/draft_uploads.cpp` を
明示的に移植した（独立した当forkの修正、MIT repository LICENSE）。全batchのsourceを
immutableにし、正常時は元々の最終waitだけ、早期return/例外ではownerより先にdrain。
drainがthrowすればsourceを解放せずnative processを終了する。kernels、精度、batch順、
layer-major schedulerは変更しない。fixtureはdeferred read、partial/full/262144、
early return/exception、failed-drain fail-stopを対象にする。すべて未実行。
歴史DPCT async handlerが例外を表示して握り潰す問題は本修正で解消したとは扱わない。
