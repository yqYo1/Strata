<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Gated region alias の実機診断

既存 8K の head 解放・再読込と scratch alias は、fresh GEN を 2 回完了しました。
別の 12K 診断では最初の 1,024 token を完了し、次の 32,768 token は workspace
確保前に容量 guard が拒否しました。要求は 4,886,504,448 bytes、空きは
4,422,336,512 bytes でした。管理側の FATAL メッセージ判定も誤って停止したため、
元の失敗と不完全な観測を保持し、別の閉鎖確認でプロセス終了、GPU、モデルと
provider のハッシュ、障害記録を確認しました。最初の空き値との差
897,024,000 bytes の原因は未帰属です。GPU ハングを示す記録はありません。

新しい候補は FP32 の gated を同じ in-order queue の共有領域に重ねます。
最後の HC mix reader より後に GDN、QSA、MoE が書きます。計算の精度は変えません。
stock 12K での要求量差は padding 込み 503,316,736 bytes です。
exact STRATA_PREFILL_GATED_ALIAS=1 の場合だけ有効で、既定 OFF を維持します。

候補は 1,024 と 32,768 token の fresh GEN を完了し、通常 QUIT と後続 GPU 確認も通りました。
実際の chunk、pointer、容量台帳、各要求と終了の結果は summary.json にあります。
この診断は速度、NLL、state の同等性、全物理 262144、restore の証明ではありません。

free と contrib-llvm の全ビルド、CPU Release と ASan / UBSan が通りました。
最初の lint は未知 owner を保持する fixture の漏れを指摘しました。Main は全 assertion
の後に子プロセスを終了することを明示し、fixture を 4 構成で再実行しました。
失敗記録を残し、変更 5 ファイルを実際の親に対して再検査した新規指摘はゼロです。
CPU helper fixture は layout の算術と範囲を検査し、実デバイスの所有権は証明しません。

registry v192 は 939 件です。R389 v1 は HIP / CUDA を調べており今回の対象外です。
元を保持し、v2 で実際の SYCL chooser を調べ直しました。R390 の所有者調査も
897 MB を特定の cache に帰属させていません。R391 は独立した source 監査です。R393 の調査から、既存の未同期で device 別ではない
chooser の値を読むだけでは bound を縮められないため、cold bound を維持します。

速度比較は変更していない上流 bench.sh --long --runs 3 に統一します。
今回の診断時間は速度試行に数えません。固定履歴の分布と state、restore、全物理
コンテキストを確認し、同じ上流 client の 8K / 12K 比較で採否を判断します。
個別の元結果は保持し、閉じた重複ログだけを retention manifest に沿って削除します。
