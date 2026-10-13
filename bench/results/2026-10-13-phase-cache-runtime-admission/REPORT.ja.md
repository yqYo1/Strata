<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Phase cache の実行前検査

既定 OFF の phase expert cache を実装しました。prefill の workspace を
退役してから追加 expert を map、fill、readback、residency 比較し、decode に
使います。次の fresh GEN では graph reader を退役して追加 mapping を外します。
KV、GDN、PLE の所有 buffer と数値形式は交換しません。不明な map/unmap
状態は fail-stop し、所有権を process 終了まで保持します。

Arc B570 / Ryzen 5600X の環境で、source-built DPC++ 7.1.1 による free と
contrib-llvm のビルド、CPU 所有権テスト、別ビルドの ASan/UBSan、サーバーの
phase 寿命 2 テストが通りました。直前の全サーバー 249 テストも記録しています。
全変更ファイルの lint は変更前の実コードと照合し、新規指摘は 0 件です。
既存の mypy、CMake、解析指摘があるため lint 単体の exit は 1 です。
初回失敗や型比較で止まった v4 ビルドの結果を成功へ書き換えていません。

小さな実 VMM 検査の結果は summary.json と receipts に保存しています。
この検査は model、verifier graph、速度、品質、全物理 262144 入力を認定しません。
実 provider の map/unmap エラー注入も行っていません。次は二つの fresh GEN と
正常 QUIT を確認してから、変更していない上流 bench.sh で性能を比較します。
生成文の完全一致を一般の品質合格条件にはしません。

並行調査は registry v186 に保存しました。誤りのあった元報告と修正版を
併記し、Main の訂正を summary.json に残しています。head lease と full-context
probe は source 差分だけを保存し、ビルド、実行、物理メモリ効果は unverified です。
閉じた処理の重複ログだけを、証拠と hash を照合した削除 manifest に従って退役します。
元の pinned receipt は変更せず保持します。
