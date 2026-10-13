<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Phase cache の実モデル寿命確認

Arc B570 / Ryzen 5600X、free の実モデルで fresh 1024 と 32768 入力を
同じ process に送り、追加 cache の checked map・copy・readback・比較、
二回目の解放と workspace 再作成、正常 QUIT と exit 0 を確認しました。
検査前後でモデル・provider の hash が一致し、新しい GPU fault はありません。
これはログと validation を有効にした寿命検査です。性能値には採用しません。

single GPU の serial 設定は pipeline-windows=0 でした。誤った 1 の起動条件を
直し、free と contrib-llvm のビルドおよび新規 lint 指摘 0 件を確認しました。
元の起動失敗 v1 と、正常な EOS を検査側が誤判定して停止した v2 は失敗の
まま保存しました。v3 は EOS を上限以内の正常終了と扱い、QUIT 後に判定します。

速度比較は変更していない上流 bench.sh の long、3 回測定を使います。
既定 OFF のままです。品質、全物理 262144 入力、head lease の効果は
この検査では認定していません。生成文の完全一致を品質の一般条件にはしません。

R374 により、以前の約 1.18 GB は共有領域候補だけで、workspace 全体では
ないと確認しました。従来の容量から導いた fit 推定は撤回します。R375 の
copy・readback・hash 候補は未測定です。元報告と Main の訂正は JSON に
保存しました。閉じた重複ログだけを hash 照合と削除 manifest に従って削除します。
