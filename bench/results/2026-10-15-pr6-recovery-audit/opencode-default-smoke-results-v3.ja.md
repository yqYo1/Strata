# Orca経由のOpenCode既定モデル確認 v3

2026-10-18（日本時間）、Orca CLI 1.4.227／実行中runtime 1.4.219、
OpenCode 1.19.2で成功した。`worker-start --worktree current --agent opencode`
で既存の `feature/sycl` に起動し、モデルと推論努力の指定を省略した。
画面表示は **DeepSeek V4.1 Flash Free**。モデル設定は変更していない。

workerは実際のツールで `AGENTS.md` と `LICENSE` を読み、LICENSEの先頭行と
SHA256、23×29、作業ブランチと上流を367バイトのJSONに保存した。
Mainはnonceと各項目を独立照合し、正しいTask／Dispatchからの
`worker_done`（`succeeded`）を受信・確認・ackした。
Taskは `task_fc752aa4ca10`、Dispatchは `ctx_0d9c332167a1`。

シェル操作では5回の権限待ちがあり、コマンドを確認して各回 `Allow once`
で進めた。恒久的な権限設定は変更していない。この設定での無人実行は未確認。
Orcaのターン開始観測はunsupported、出力は端末へのfallbackであり、
実際の生成物とTask固有の完了通知を実行の根拠としている。
モデル名は画面表示の観測で、provider内部のルーティングは未検証。

完了前にOrcaが端末を `user_owned`／`retained`、理由 `user_takeover` と記録した。
その所有権を尊重して端末を閉じていない。Run内のreclaimable workerは0件。
GPU、ビルド、推論実装、追加ブランチ作成は今回のテストに含めていない。
長時間の調査品質や実装能力は評価していない。

[依頼文](opencode-default-smoke-spec-v3.txt)、
[実際のworker生成物](opencode-default-smoke-worker-v3.json)、
[起動・権限・照合・完了・端末所有権の根拠](opencode-default-smoke-result-v3.json)を保存した。
重複する一時応答は結果への集約を確認し、
[削除記録](opencode-default-smoke-retention-v3.json)を残して退役した。
