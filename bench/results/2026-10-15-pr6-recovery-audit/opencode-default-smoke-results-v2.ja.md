# Orca経由のOpenCode既定モデル再確認

2026-10-18（日本時間）、Orca 1.4.219 / OpenCode 1.19.2で成功。
既存の `feature/sycl` に `worker-start --agent opencode` で起動し、
モデルと推論努力の引数を省略した。画面表示は **DeepSeek V4.1 Flash Free**。
モデル設定と恒久的な権限設定は変更していない。

workerは `AGENTS.md` と `LICENSE` を読み、LICENSEの先頭行・SHA256、
23×29=667、主作業ブランチと上流を367バイトのJSONに保存した。
Mainがnonceと全項目を独立確認した。正しいTask/Dispatchからの
`succeeded` 完了通知を受信し、通知確認とworkerの解放まで完了した。
Task `task_47ec0c97059e` / Dispatch `ctx_a832eb4d1581`。

シェルの実行前に計4回の許可待ちが発生し、内容を確認して各回
`Allow once` を選択した。この設定で無人完走できるとは確認していない。
Orcaのターン開始観測はunsupported、出力取得は端末へのfallbackである。
生成物と受理された完了通知を別々に検証した。
モデル名の根拠は端末表示とworker報告であり、provider内部のルーティングは未検証。
長時間の調査品質や実装能力は今回の評価対象に含まない。
GPU実行、ビルド、実装コード変更、新規ブランチ作成は行っていない。

[依頼文](opencode-default-smoke-spec-v2.txt)、
[worker生成物](opencode-default-smoke-worker-v2.json)、
[コマンド・検証・完了と解放の根拠](opencode-default-smoke-result-v2.json)を保存した。
重複する成功応答の原本は、必要な証拠を集約・照合した後に
[削除記録](opencode-default-smoke-retention-v2.json)を残して退役した。
