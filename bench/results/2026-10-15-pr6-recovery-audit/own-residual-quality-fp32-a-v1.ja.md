# 長入力の最初のFP32損失基準

公式prefillベンチの固定word-listをthinking有効のモデルtemplateでtokenizeした
78444入力、78443 batched行について、全行の次トークンNLLを記録した。
48層×10チャンク、FP32 residualのGPU32768行/host45675行、context262144、
KV int8/resident32768、MTP4、native/canonical dense bank、radix4=1。
ソースは `dfca0e14` の固定snapshot。実行binary、source、環境、コマンドの
identityは `own-residual-quality-fp32-a-v1.json` に固定している。

全78443 target ID、raw/meta/recipe/closureの対応、全損失の有限性・非負を確認し、
所有プロセスは正常終了した。新しいGPU障害0、dumpなし、boot不変。
平均NLL4.823238516902749、p50=4.611052417886379、p99=9.498535981588432、
最小3.902456135941772e-11、最大28.16826821829457。
観測callbackは749978.137127 msであり、この実行は性能cohortから除外する。

これは最初のFP32基準であり、FP32反復差、FP16品質、全GPU収容、
物理262144境界、RESTOREの合格ではない。word-listは汎用の自然文・タスク品質を代表しない。
次は同じcontroller、入力、binary、環境で独立したFP32-bを実行する。

`main-own-residual-quality-root-v1/fp32-a` のrecipe、closed、faults、NLL raw/metaの
5ファイルは、次の2基準対FP16比較の実入力として残した。その他の重複protocol/traceは、
全48層完了・配置・bank復帰・診断終了を抽出し、ハッシュ確認後に削除した。
元のPASS receiptは変更していない。
