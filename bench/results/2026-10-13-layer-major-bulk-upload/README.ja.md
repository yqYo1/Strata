<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# レイヤー単位一括expert転送の実機検証

既定OFFの一括転送を、free/contrib-llvmのビルド、4構成CPU・ASan/UBSan、サーバーlifetime、静的検査で確認。
実GPUの小試験は両モードで全512expert、7幅、物理容量、条件不成立時の個別コピーを確認した。
実モデルfresh GEN1024/32768は全48層のowned Host-USM転送を各1回で完了し、正常QUIT。
各GENの論理payload50292326400 bytes、コピー48回、fallback0、最大1回1363148800 bytes。
従来4K処理の同一固定履歴2位置で全248320 FP32logitをバイト比較し一致。
これは2位置の証拠で、能力全体・全256K・RESTORE・速度の合格ではない。

初回検査の改行比較ミスと次の11静的指摘は失敗のまま保存。
Mainは桁あふれとゼロを防ぐcacheサイズ計算、実配列長でのoffset書込み、テスト型・副作用順序・docs空行を修正。
修正後の全ビルド・CPU検査と5ファイル再検査を通過。残る17不変ファイルの完了済検査を再利用し、抑制なし。
前後model hash、同一boot/provider、GPU health、無障害を確認。転送wallはPCIe単独の時間ではない。
次は変更しない公式bench.shのdebug経路閉鎖とquiet3回比較。正式採用は保留。
