# BF16 QSA 長大prefill比較

公式bench.sh、実入力32K以上、context/resident65536、chunk8192、thinking ON、各arm独立3process。

|経路|中央値 tok/s|平均 ± sample SD|範囲|
|---|---:|---:|---:|
|current|393.4|393.4 ± 0.8|392.6–394.1|
|bf16|406.1|406.1 ± 0.3|405.8–406.4|
|bf16-prefix|445.4|445.4 ± 3.5|441.9–448.9|

全個別値・prompt時間・入力件数・request hash・配置をJSONに保存。3点はばらつきの把握であり、変動が消えた証明ではない。
これはprefill性能の検証。品質・SGLang/vLLM CUDA実行値・full-model256K・RESTORE・default採用は別の判定。次は新経路の資源別profile。
