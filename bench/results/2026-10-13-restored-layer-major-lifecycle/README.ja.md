<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# レイヤー単位prefill復元候補の実機検証

1Kと32Kのfresh GENを同一プロセスで完了し、通常のQUITで終了した。
48層ごとに全512 expertを一度読み込み、32Kでは4チャンクを処理した。
実行前後の16384ワードGPU検証、カーネル障害検査、12モデル入力の全SHA検査は通過した。

候補はphase0、FP32中間表現の全RAM配置、8Kチャンク、context262144、int8 KVのresident32768、明示700 MiB reserve。
各入力はmax_new1、raw spec4、実効T1、reuseなし。実際のprefillは入力の最後の1 tokenを除く1023/32767行。
32KのRAM中間表現は1342136320 bytes、レイヤーcacheは1363148800 bytes。
容量は実行中のfresh freeから検査し、通常device miss ringを省略した。
終了値や診断時間はsummaryに個別保存した。速度比較のサンプルではない。

free/contrib-llvmのビルド、4構成のCPU CTest、ASan/UBSan、サーバーphase lifetime、実際の親との静的検査を通過。
最初の符号比較ビルド失敗、2件の新規静的指摘、reserve引数確認のcontroller失敗は失敗のまま保存した。
Mainが型、Impl状態の冗長bool、乗算前の型拡張、明示reserveのcontrollerを修正した。
検査の無効化やreserve削減は行っていない。

数値品質、256K全領域、RESTORE、bench.shによる速度比較、正式採用は未成立。
同じ出力IDは品質比較の代わりにならない。既存serve profileは有効時にmixer side branchを抑止するため、静かな実行のcritical pathをその比率から断定できない。
約6.45 GB/sはHost-USM H2Dの既存測定で、pageable H2Dの既存値は約4.61 GB/s。今回のbridge帯域の実測ではない。

別作業ツリーのdecode cache/head全解放案と、型付きdecode route ledgerはまだsource候補。
Mainは全物理leaseを2層・2回検査する小さな実機fixtureを追加し、ビルドを進めている。
次はそのlease検証、固定履歴の数値比較、256K境界、上流の変更しないbench.sh --long --runs 3。

調査台帳v195は過去958件と新規9件を参照する。
成功ログと親ソース表示の重複は検証済み構造化結果に集約して削除し、path/hash/reasonをmanifestに残す。
現在のcontrollerがhashで参照する元のreceiptは変更せず保持する。物理ストレージ解放量は主張しない。
スキルの検出リンクはこのプロジェクトの.agents/skillsだけにあり、ユーザー共通の2か所には存在しない。
