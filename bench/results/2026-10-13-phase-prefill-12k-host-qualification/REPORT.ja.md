<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Phase prefill 12K 候補のビルドと CPU 検証

source-built DPC++ 7.1.1 で free と contrib-llvm をビルドしました。
変更した 11 ファイルは clang-tidy、cppcheck、文書、秘密情報、ライセンスの
検査を実際の検証済み親ソースと比較し、新規指摘はゼロでした。
両モードの所有権 CTest、別の Release と ASan / UBSan、サーバーの phase
寿命テストが通りました。空の catch の新規指摘を修正して fixture を再検査し、
親の全件検査から変わらない他のファイルの結果を引き継ぎました。元の失敗は保持しています。
管理スクリプトの事前確認が想定した exit 2 を誤って拒否した記録も残しています。
その停止時にはビルドやテストは開始していません。GPU とモデルは実行していません。

head lease、FP32 scratch alias、明示 12288 容量は既定 OFF の実験です。
元の GGUF payload を初回 H2D 前に CPU RAM に固定し、reload はその bytes を
使います。未知の copy / conversion と解放状態は owner と source を保持し、
再試行や解放を拒否します。成功した pointer は個別に消します。
lease の変換は Runtime の checked drain を使い、保存された非同期エラーと
watchdog を確認します。通常の --native の初期ガードも修正しました。

CPU fixture は allocation 1 / 2 の false / throw、copy と変換の未知状態、
同じサイズの元データ変更、destructor の fail-stop、解放前 / 後の失敗を
確認しました。両 binary で各 12 件、計 24 件の初期ガード確認も通りました。
正しい native / explicit head は意図的な普通の入力エラーまで進み、pipeline、
auto、混在容量、欠けた opt-in、01、オーバーフローは予想した位置で停止しました。
この fixture は Runtime とモデルの構築前で止まり、物理容量や DMA を証明しません。

R386、R387 と実装報告を registry v191 に追加しました。R386 の gated overlay は
まだ実装していません。現在の Alloc の padding を含めると stock 12K で別の gated
allocation を取り除くソース上の差は 503316736 bytes（480 MiB + 256）です。
物理 VRAM と速度の測定ではありません。Main は実装報告から、解放失敗の扱いと
Runtime drain を追加修正しました。具体的な指摘と元の報告は保存しています。

速度比較は引き続き変更していない上流 bench.sh --long --runs 3 で行います。
今回、新しい速度試行はありません。既定 OFF を維持し、実機の head / alias / 12K
admission、固定履歴の logit / state、全物理 262144 と restore を確認してから採否を
判断します。過去の生成文の完全一致を品質の一般条件にはしません。
