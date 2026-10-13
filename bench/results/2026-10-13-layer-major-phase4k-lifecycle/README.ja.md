<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# 4K作業領域と32K全GPU中間表現の実機検証

fresh GEN1024/32768を同一プロセスで完了し、通常QUITで終了した。
4Kチャンクとcache/head全物理lease、FP32、int8 KV resident32768、context262144、明示700 MiB reserveを使用。
32Kでは実際32767行の中間表現1342136320 bytesをGPUに確保し、RAM tailは0。48層・8チャンクで最後は4095行。
従来の処理順と全248320 logitsを位置32767でバイト比較し一致、同じ固定履歴target NLL差0。
これは1位置の数値比較で、能力全体・全256K・RESTOREや速度の合格ではない。

内部prefill小計55,568.811 msのうちsource取得とchecked同期コピーの露出区間は14,961.143 ms。
24576回・50,292,326,400 logical bytesのexpert uploadで、純PCIe時間ではない。公式TPSには換算しない。
各待ちが完了してから計算する区間なので、一括転送案の対象として測定上の根拠がある。

free/contrib-llvmのビルド、4構成CPU・ASan/UBSan、サーバーlifetime、静的検査、実機tiny full lease、前後GPU・model hash・無障害検査を通過。
初回静的検査のdoc空行2件は失敗のまま保持。Mainが各翻訳の余分な空行だけ修正した。
16個のコードとbinaryは不変で完了済みコード検査を再利用し、変更docsを全ツールで再検査した。
通常GENはfull256K/出力0の経路にならないため、別の明示診断armをsourceのみ用意した。
同armも新しいbulk upload案も未ビルド・未実行。callback coverageは最終物理storeの証明ではない。
次は変更しない上流bench.sh --long --runs3のdebug経路閉鎖とquiet比較。正式採用は保留。

台帳v197は987件と新規7件を参照。R422がMainのactive8の誤りをK10に訂正した。
982272000 bytesはNT1の論理payloadで、DRAM/PCIe実トラフィックや速度上限ではない。
スキルの現在の検出はこのプロジェクトだけ。採用時には旧root配置も修正版へ移す。
成功重複ログはmanifestと構造化結果へ集約し、現consumerがhash参照する原本receiptは保持する。
