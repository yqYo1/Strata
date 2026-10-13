<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# 上流 bench.sh による phase OFF 基準

Arc B570 / Ryzen 5600X、free、source-built DPC++ 7.1.1、native IQ3_S で、
変更していない上流 v0.1.43 の bench.sh --long --runs 3 を実行しました。
engine source は 0f112782、phase cache と native COPY は OFF、chunk 8192、
group 16、workers 5、tasks 18、no borrow/MTP/reuse、int8 KV resident 32768、
context 262144 です。trace と validation は性能測定中に無効です。

| Run | 実入力/read | 出力 | Prefill tok/s | Decode tok/s |
| --- | --- | --- | --- | --- |
| 1 | 32049 | 24 | 384.8 | 7.3 |
| 2 | 32049 | 24 | 383.1 | 7.8 |
| 3 | 32049 | 24 | 386.9 | 8.6 |

各入力は 32049、再利用 0、出力 24 でした。prefill 中央値は
384.8 tok/s、decode 中央値は 7.8 tok/s です。
短い会話や複数ターンの結果は標準スイートの補助結果として残しています。
全リクエストが成功し、正常 QUIT、exit 0、所有 process の終了を確認しました。
測定前後の source、binary、model、provider が一致し、新しい GPU fault は
ありません。reset、サービス変更、再試行はしていません。

これは OFF 基準の一回の campaign です。ON 効果や採用を主張しません。
prefill と decode は別々に判定し、差が曖昧なら反復します。標準の約 32K
入力と 24 出力であり、全物理 256K、持続 decode、品質の認定ではありません。
生成文の完全一致を一般の品質条件にはしません。

並行調査 5 件を registry v189 に追加しました。R376 の誤った KV mode 0
前提は元報告と訂正版 v2 を両方保存し、Main の訂正を JSON に残しました。
KV は全 context 分の authoritative host buffer と resident GPU page cache
です。静的要求量と free telemetry を同一視しません。R377 の hash 診断省略、
R378 の測定長設定拡張、R379 の全 context 復元確認は source 調査だけです。
閉じた重複ログは hash と証拠を確認した削除 manifest に従って削除します。
