# OrcaからOpenCodeを既定モデルで起動するテスト

2026-10-17に実行し、成功した。Orca 1.4.219 / OpenCode 1.19.2。
`worker-start --agent opencode` を既存の `feature/sycl` 作業場所で使用し、
`--model` と `--effort` は渡していない。新しいブランチは作成していない。

画面のモデル表示は **DeepSeek V4.1 Flash Free**。worker自身の記録は
`opencode/deepseek-v4.1-flash-free`。Orcaの構造化provider情報ではmodelが
nullなので、モデル名の根拠は画面表示とworker報告であり、provider側の
ルーティングを別途検証したという意味ではない。

workerは `LICENSE` と `AGENTS.md` を読み、シェルでチェックサムを取得し、
568バイトのJSONを保存した。Mainがnonce、LICENSE先頭行とSHA-256、
17×19=323、作業ブランチの説明を確認し、全て一致した。
Orcaへの `worker_done --outcome succeeded` も受信・検証済み。
Task `task_e622f3b011e8` / Dispatch `ctx_34e9232e7fa5`。
起動から完了通知までは約98秒（許可待ちを含む）。性能評価ではない。

現在のOpenCode設定ではシェル操作に許可が必要だった。Mainが内容を確認し、
チェックサム取得、レポート読戻し、worker自身のメール確認、完了通知の4件を
それぞれ「Allow once」で許可した。モデル設定や恒久的な許可設定は変更していない。
したがって基本的な委任は利用可能だが、無人で完了する設定の検証にはなっていない。
実装能力や長時間調査の品質もこのテストの評価対象ではない。

Orcaの入力受付は確認できたが、OpenCodeの構造化turn-start通知はunsupported、
出力確認はterminalへのfallbackだった。実際の生成物と完了通知を別途確認している。
background terminalをUIに表示できないという起動時warningはあったが、
workerの実行、読み取り、完了通知、終了処理は成功した。

完了後に `worker-release` し、Orcaが `closed_agent_terminal` とアーカイブ完了を
確認した。完了メールもack済み。コード変更やGPU実行はしていない。

コマンドの全引数、起動・完了・終了の根拠、入力と出力のハッシュ、観測限界は
[構造化結果](opencode-default-smoke-main-v1.json)に保存した。
[依頼文](opencode-default-smoke-spec-v1.txt)と
[workerの生成物](opencode-default-smoke-worker-v1.json)も保存した。
成功した重複画面とポーリングログは、根拠を抽出・検証した後に削除し、
[削除記録](opencode-default-smoke-retention-v1.json)へパス・サイズ・ハッシュを残す。
