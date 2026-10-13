<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# 上流 bench.sh による phase cache 比較

Arc B570 / Ryzen 5600X、free/source-built DPC++ 7.1.1、native IQ3_S で、
変更していない上流 v0.1.43 の bench.sh --long --runs 3 を使いました。
同じ binary、argv、model、client で OFF の後に ON を測定しました。
間の source commit は手順とレポートだけです。COPY 0、chunk 8192/group 16、
workers 5/tasks 18、no borrow/MTP/reuse、int8 KV resident 32768/context 262144。
trace と validation は無効で、違う設定は STRATA_PHASE_DECODE_CACHE だけです。

| Phase cache | Run | 実入力/read | 出力 | Prefill tok/s | Decode tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 32049 | 24 | 384.8 | 7.3 |
| 0 | 2 | 32049 | 24 | 383.1 | 7.8 |
| 0 | 3 | 32049 | 24 | 386.9 | 8.6 |
| 1 | 1 | 32049 | 24 | 362.4 | 7.1 |
| 1 | 2 | 32049 | 24 | 375.4 | 8.2 |
| 1 | 3 | 32049 | 24 | 383.0 | 9.1 |

再利用は全件 0、出力は 24 でした。prefill 中央値は OFF 384.8、
ON 375.4 tok/s（-2.44%）。decode 中央値は
OFF 7.8、ON 8.2（+5.13%）ですが、
範囲 7.3–8.6 と
7.1–9.1 は重なります。
ON の値は試行順に上がり、24 出力のばらつきもあるため、decode 改善の
確定や採用はしません。既定 OFF を維持し、decode の判断には対応する
OFF/ON の反復を増やします。prefill と decode を同じ判定にしません。

両条件は全リクエスト成功、正常 QUIT、exit 0、所有 process の終了、
前後の source/binary/model/provider 照合を通りました。新しい GPU fault は
なく、reset、サービス変更、再試行は行っていません。標準の約 32K ケース
であり、全物理 256K、持続 decode、品質を認定する結果ではありません。
上流 ON の results.json は元の bytes をそのまま保存し、OFF は前のレポートに
残しています。短い会話と agent は補助結果です。

並行調査 7 件を registry v190 に追加しました。入力量子化は既に expert 間で
共有されており、重複削減案は採りません。head lease、FP32 scratch aliases、
明示 12K の候補は別 worktree の source 差分だけを保存しています。ビルド、
実行、fit、数値品質は未認定です。Main が測定終了後に検証します。
実モデルの fixed-history logits producer と後続の gated overlay / Dm streaming
も別途検証が必要です。生成文の完全一致は品質の一般条件にしません。
成功した重複ログを、証拠と hash を確認した削除 manifest に従って削除します。
