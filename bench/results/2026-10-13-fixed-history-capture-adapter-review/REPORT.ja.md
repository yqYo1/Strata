<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Fixed-history capture receipt の CPU 検査

同じ履歴からの full-vocabulary FP32 logits を比較する既存の道具に、capture の
出所と終了状態を照合する companion を追加しました。owner の終了・reap、
source・command・環境・provider の同一性、全行の寸法と有限値を検査します。
不完全な capture を合格にせず、生成文の完全一致を品質の条件にしません。

Main が新規 schema／異常入力 9 テストと既存 comparator 11 テストを実行し、
通りました。新規ファイルの全 lint も通りました。初回 lint の型注釈と改行の
指摘を修正し、初回の失敗記録を保存しています。モデルや GPU は実行していません。
実 producer の登録は空のままで、production と品質の認定値は false です。
実 capture、GDN/QSA/PLE 状態の RESTORE 比較、held-out 品質は unverified です。

並行調査の修正版は、stock client の ckpt 省略が実際の fresh state と容量 0 の
条件で受容できることと、既存 native copy stream の production 呼び出し先が
既に prefill だけであることを確認しました。decode の queue が generic であることと、
共通 CPU guard の負荷まで同一であることを区別します。追加 selector は実装しません。
速度の比較には引き続き変更していない上流 bench.sh を使います。

元の報告と修正版、実行 command と環境、失敗・成功の結果は registry と receipts に
保存しています。閉じた CPU 検査の重複ログだけを照合後に削除し、pinned receipt は
そのまま保持します。次は Main 管理の実モデル phase 確認と production capture です。
