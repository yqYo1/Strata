<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# decode cache全物理leaseとGPU中間表現の実機検証

decode cacheの全物理extentをprefill中に解放し、終了後にRAMから埋め直して公開する候補を検証した。
2層・2回の小さな実機fixtureと、同一プロセスのfresh GEN1024/32768、通常QUITを完了した。
全RAM配置と先頭GPU16384行＋残りRAM配置を別々に確認。32KではGPU671088640 bytes、RAM671047680 bytesを実際に確保した。
context262144、int8 KV resident32768、8Kチャンク、FP32、明示700 MiB reserve、max_new1、reuseなし。
毎層全512 expertを一度アップロードし、48層・4チャンクを処理した。

従来の処理順と各候補の実際に残した32K行、全248320 logitsはバイト単位で一致した。
固定履歴の位置32767だけの比較であり、モデル能力全体や256K・RESTOREの合格ではない。
GPU前後検査、同一boot、12モデル入力全SHA、provider、無障害ログ、正常所有物終了を個別receiptに残した。
free/contrib-llvm、4構成CPU・ASan/UBSan、サーバーlifetimeと親比較の静的検査を通過した。
静的検査の初回4件は失敗のまま保存。型拡張前の乗算を修正し、到達可能な2つのlease後失敗もfailstopにした。

速度は今回のGEN診断時間から判定しない。次は4K作業領域の検証と、変更しない上流bench.sh --long --runs3。
全物理262144入力、RESTORE、数値能力、正式採用は未成立。新しい4K案とdecode ledgerは別作業ツリーで検証する。
R412は親だけを調査したため、実際の18 source pinsを監査したR413を採否の根拠にする。
R413の「5ファイル」は列挙すると6ファイル。R415の台帳不在は探索誤りで、実際のL台帳をhash検証した。
R416の8 expert転送量上限はNT1 decode対象で、32K prefillのroute集合には適用しない。source+copyと入れ子のphase時間を純PCIe時間やcritical pathとして合計しない。
成功重複ログはmanifestのpath/hash/reasonと構造化証拠に集約し、現controllerが参照する元receiptは変更せず残す。
スキルはこのプロジェクトの.agents/skillsでのみ検出され、ユーザー共通にはない。採用時には親の旧root配置も修正版へ移す。
