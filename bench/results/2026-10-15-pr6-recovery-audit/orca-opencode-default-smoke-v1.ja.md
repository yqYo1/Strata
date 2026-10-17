# Orca経由のOpenCode既定モデル起動テスト

2026-10-17、Orca 1.4.219とOpenCode 1.19.2で成功。モデル・推論努力の指定を省略し、既存の`feature/sycl`作業ディレクトリに1ワーカーを起動した。実際の端末には既定モデルとして **DeepSeek V4.1 Flash Free** と表示された。Orcaの構造化モデル欄は`null`なので、内部モデルIDまでは確認していない。

ワーカーは`AGENTS.md`と`CMakeLists.txt`を読み、サイズとSHA256、内容の説明を1065バイトのJSONに保存した。メイン側でも現在のファイル、起動前のハッシュ、ワーカーの報告が一致することを確認した。正しいTask/Dispatchによる`succeeded`の`worker_done`を受信し、ワーカーのリソース解放と通知の確認処理まで完了した。

## 実際の起動方法

既存のRunとコーディネータに対し、次の形式で起動した。モデル関連オプションは渡していない。起動時の完全なタスクと入力ハッシュは[request](orca-opencode-default-smoke-request-v1.json)、起動応答は[launch](orca-opencode-default-smoke-launch-v1.json)に保存した。

```sh
/home/yayoi/.nix-profile/bin/orca orchestration worker-start \
  --spec '<対象・制約・成果物・完了条件を含むタスク>' \
  --worktree path:/home/yayoi/orca/workspaces/Strata/sycl \
  --agent opencode \
  --run run_c8f478972ec8 \
  --from term_6fbfd718-e3a2-46d8-ab62-73c36d2851a6 \
  --timeout-ms 45000 --json
```

## 運用上の制約

- 現在の権限設定では、ハッシュ計算、レポート確認、受信箱確認、完了通知に計4回の許可待ちが発生した。メインが対象コマンドを読んで各回`Allow once`を選択した。永続的な権限設定やモデル設定は変更していない。この設定のまま無人で完走できるとは確認していない。
- Orcaの起動応答は`input_accepted`、ターン開始確認は`unsupported`だった。出力取得も`provider_unsupported`による端末へのフォールバックを使用した。成果物と有効な完了通知により、実際に処理が完了したことを確認した。
- 端末の表示登録に警告が出たが、出力取得、単発許可、完了通知、リソース解放は成功した。
- ワーカーは依頼で禁止したGitコマンドのうち、追加の読み取り専用ブランチ照会を要求した。メインが確認して単発許可した。Gitの変更操作は行われていない。厳密な指示遵守や長い調査の品質まで合格したという評価ではない。
- GPU処理、ビルド、推論性能の測定はこのテストでは行っていない。

独立検証と終了処理の記録は[result](orca-opencode-default-smoke-result-v1.json)、実際の成果物は[worker report](orca-opencode-default-smoke-worker-v1.json)、個別許可は[permission actions](orca-opencode-default-smoke-permission-actions-v1.json)を参照。
