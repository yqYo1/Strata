<!-- SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors -->
<!-- SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# HTTP コミュニティ計測

[English](COMMUNITY_BENCHMARKS.md)

専用サーバーへ HTTP リクエストを送り、環境、個別サンプル、失敗を記録します。

## コミュニティの報告

報告は `bench/results/` にあります。ソース、機器、モデル、条件、成功・失敗の範囲を確認してください。
条件の異なる報告を並べただけでは、変更による高速化とは判断できません。

## ベンチマークの実行

先に専用サーバーを起動し、モデルの読み込み完了と、ほかのリクエストがないことを確認します。
このツールは `--url` で接続し、サーバーの起動・停止・設定変更は行いません。
リクエストは会話キャッシュを使います。普段のアプリが利用するサーバーと共有しないでください。

```sh
./bench.sh --url http://127.0.0.1:8095 --out ./bench/results
# ランチャーを使わない場合:
.venv/bin/python tools/strata_bench.py --url http://127.0.0.1:8095 --out ./bench/results
```

認証には `STRATA_KEY` 環境変数、または `--config FILE` の `api_key` を使えます。
`--config` は報告の注記とキーの読み込み用です。その設定をサーバーへ適用しません。
対話的な setup は、希望した場合に別ターミナル用のコマンドを表示します。既定は「いいえ」です。
`xestrata --bench` は別のネイティブエンジン計測です。

| ケース | リクエスト | 記録 |
| --- | --- | --- |
| chat | 固定の短い質問、最大 100 新規トークン | decode tok/s、最初のトークンまでの時間 |
| 4K / 16K | 校正した新しい単語列 | prompt tok/s、最初のトークンまでの時間 |
| agent | system、4 tools、5 turns。各 turn に約 1,500 語を追加 | 会話再利用、再読込みトークン、各 turn の時間 |
| `--clients N` | N 件を同時送信。slot 数を超えた分は待機 | 合計 tok/s、p50/p95 の時間、失敗件数 |
| `--long` | 約 32,768 トークンの新しい prompt | prompt tok/s、最初のトークンまでの時間 |

実トークン数は回答の usage で確認します。既定は各ケース 3 回です。
`--runs` は 1–16、`--clients` は 0–32 です。長入力・並列実行は、そのエンジンで確認した範囲を使います。
所要時間はサーバーによります。送受信する HTTP テキストは共有 64 MiB、保存する報告も合計 64 MiB 以下です。
各生成リクエストは 3,600 秒の期限と最大 30 秒の socket 待機を使います。実行全体の期限は別途管理してください。
`[DONE]` のない切断、リクエスト失敗、予算超過を成功に数えず、コマンドは非ゼロで終了します。

## 記録する条件

GPU・VRAM・CPU・RAM はサーバーの HTTP メタデータから取得し、取得できない値は不明とします。
クライアントの OS・Python・checkout と、サーバーの engine version を区別します。
注記した config と実際の設定が一致するか確認してください。

- ソースとバイナリ、モデル・tokenizer・profile の識別子またはハッシュ
- context、KV、prefill、spec/MTP、並列 slot、環境設定
- 実 prompt / generated / reused トークン数、個別サンプル、集計、失敗
- ツールが取得しない電力制限、背景負荷、ストレージなどの条件

## 測定方法

ウォームアップと校正を集計から除き、その失敗は記録します。新しい prompt の各回に固有のタグを付けます。
agent ケースは過去の回答を含む会話を継続します。速度は engine の timings、待ち時間は HTTP クライアントの時計です。
比較ではモデル・入力・設定・実トークン数をそろえ、必要なら実行順を入れ替えて反復します。

```sh
./bench.sh --url http://127.0.0.1:8095 --compare ./bench/results/OLD
.venv/bin/python tools/strata_bench.py --compare ./bench/results/OLD --compare-only ./bench/results/NEW
```

## 正しさの検査

このツールは回答の正しさや full-vocabulary logit parity を検証しません。
固定入力での token/logit 比較や最大コンテキストの fit は別に検証し、結果を記録してください。

## 報告の共有

共有前に `README.md` と `results.json` のパス、キー、利用者名などを確認します。
置換は既知の形式だけです。環境固有の情報は手動で確認し、失敗・省略は残してください。
サーバーログはその所有者が保持方針に従って管理します。未解決障害の最後の診断は残します。
成功した raw capture は、有用な証拠を記録・確認してから直列化して退役させます。

## 報告の書式

実測していない値は `not measured` / `not reported` と書き、以下の節を使います。

### 機器とソフトウェア

サーバーの GPU / VRAM / CPU / RAM、クライアントの OS / Python / checkout、engine と source/binary の識別。

### モデルと設定

入力ハッシュ、実際の context / KV / prefill / spec / slots / env。注記 config との一致。

### 方法

コマンド、サーバー所有者、専用・idle の確認、ウォームアップ、fixture、反復数、時計、集計方法。

### 結果

実トークン数、再利用、engine timing、client latency、中央値・範囲、失敗・省略・終了ステータス。

### 正しさと制限

別途の correctness / fit の証拠と、背景負荷・電力・ストレージなどの不明な条件。
