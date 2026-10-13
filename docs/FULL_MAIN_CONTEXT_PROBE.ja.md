# メインモデルだけの全コンテキスト診断

`--full-main-context-probe` は、MTP を使わず、262144 個の入力 ID を位置 0 から production `Prefill::run` で処理する開発用診断です。生成数は 0 です。通常の生成や serve の入力制限は変わりません。実行の検証は未完了で、速度・予測品質の結果はありません。

ネイティブ pack と `--native`、正確に 262144 個の `--tokens-file`、`--max-context 262144 --max-new 0`、正の固定 `--prefill-chunk`、`--no-prefill-borrow`、有効な PLE を指定します。`--mtp`、spec、serve、batch、分割 GPU、借用、elastic/grow、dump は使えません。末尾の追加 ID は不要です。現在の検証対象は int8、streaming mode 1、resident 32768 です。

完了済みチャンクの範囲が `[0,262144)` を隙間なく覆い、PrefillStats の入力数・チャンク数が一致する必要があります。最終位置は 262143 です。続いて main-only snapshot API で QSA の実体 KV と indexer、GDN/conv、PLE を保存します。host の全 KV と GPU の限定 resident slot は別に表示します。RAM が不明・不足なら拒否し、保存直前にも再確認します。

保存された実体 KV の全バイトを照合し、running float/index/quantization scale の有限値を検査します。状態を破壊し、KV と running checkpoint の負の検査が破壊を検出することを要求します。同じ session に復元し、全 KV のバイトと指紋、全 running checkpoint を照合します。packed q4 の scale と fp16 KV の数値は個別の有限値検査対象ではありませんが、全バイトの一致は検査します。保存された論理 ID 数は処理入力数と別に記録します。復元後の推論はありません。転送の完了が不明なら fail-stop します。

`PASS` は入力処理と状態保存・復元の診断結果です。生成品質や通常の生成が全容量を使えること、性能を示しません。`bench.sh` は変更しません。host-only `full_context_probe_test` は境界・重複・欠落・最後の一セルの coverage を検査します。ビルド、CPU テスト、実機検証は Main が担当します。
