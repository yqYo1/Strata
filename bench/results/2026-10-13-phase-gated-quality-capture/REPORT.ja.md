<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# 実サーバーの最初の logit 取得

free・contrib-llvm の全ビルド、CPU Release と ASan/UBSan、サーバーの phase lifetime
検査が通りました。変更した 5 ファイルを実際の親 6035c4f と照合し、新規 lint 指摘はゼロです。

B570 10 GiB / Ryzen 5600X / 128 GiB RAM で、fresh 1024・32768 token、出力各 1 token、
8192 chunk の実サーバーを確認しました。phase/head lease/scratch/gated を使い、全語彙
248320 の有限 FP32 logit を取得しました。予測位置は P-1、実際の T=1、reuse=0 です。
取得した argmax と出力 token は一致し、通常 QUIT、前後の GPU 確認、モデル・provider
ハッシュ、障害検査が通りました。これは producer と lifecycle の確認であり、固定履歴の
分布同等性、NLL、全物理 262144、RESTORE、速度向上を証明しません。既定 OFF を維持します。

R396–R401 と復元試作・decode 集計の v1/v2 を registry v194 に記録しました。件数は 958 です。
R398 により、既存の chunk ごとの KV stage seed を保持する層順序変更と、stage を保持して
再転送を省く最適化を区別しました。R401 の GEN128 速度比較提案は採用せず、速度比較は
変更していない上流 bench.sh に統一します。調査エージェントの提案は実測結果ではありません。

32K logit のみを次の実 producer 登録・数値比較の入力として Git 外に残します。
完了した短文 tensor、partial の別名、重複 command ログは manifest の hash を確認して削除し、
個別結果と元の結果の status は保持します。次は通常 serve と streamed KV を扱える
layer-major 実装の検証を進めます。
