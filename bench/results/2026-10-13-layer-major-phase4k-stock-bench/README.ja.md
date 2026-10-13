<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# 公式bench.shによる4Kレイヤー順処理の測定

変更していない上流bench.sh --long --runs3を使用。実入力32049、出力24、再利用0。
quiet prefillは654.7/667.6/669.8 token/s、中央値667.6。decodeは8.7/10.4/11.1、中央値10.4。
debugの675.4/9.6中央値は動作診断のみで、性能比較には使用しない。
両経路は正常QUIT、前後GPU検査・同一boot・source/model/provider確認を完了し、新しい障害はない。
既存より速い観測だけでは、4K単独等への改善の帰属やdecode採用を決めない。
256K物理末尾・RESTORE・モデル全体の品質は未検証で、正式採用は保留。
測定時quiet sourceはdb088ff24e67e6a4566566dfc4762c63c55bf343。
次は同期コピー回数を24576から48にする別候補の検証とdecode処理時間の分離。
R428によるとresident累積はexpert unionではなく、転送対象削減には別の正確なunion診断が必要。
成功rawログはmanifestに従い個別結果・コマンド・環境・進捗・無障害証拠へ集約。
