<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# bulk転送と非bulk再測定の公式bench結果

変更していない上流bench.sh --long --runs3。実入力32049、出力24、再利用0。
bulk prefillは693.5/691.1/701.1 token/s、中央値693.5、decodeは10.1/7.9/8.1、中央値8.1。
非bulk再測定のprefillは666.5/666.2/668.0、中央値666.5、decodeは7.4/9.5/8.5、中央値8.5。
最初の非bulk decodeは8.7/10.4/11.1。非bulkでも低下し、bulk固有の低下とは確定していない。
全サンプルを残し、prefillの改善だけでdecode採用を決めない。正式採用は保留する。
正常QUIT、前後GPU検査、同じboot、source/model/provider確認が完了し、新しい障害はない。
最初の全buildと依存再buildでABI変更を使うtranslation unitが再コンパイルされ、測定binary hashが一致した。
次はMainがCPUサービス時間の診断実装を検証し、GU・量子化・Downを分けて測る。
256Kの物理末尾、RESTORE、モデル全体の品質は未検証。設定上の262144だけでは検証済みとしない。
ブランチ固有の履歴は先の保存mergeから元のcommit指定で復元できる。保存mergeは推論コード採用を意味しない。
