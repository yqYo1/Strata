# BF16-prefix K load配置比較

2026-10-17 JST、Arc B570 10GiB、Ryzen5600X。production kernelは変更していない。
独立fixtureの同一binaryに現行とMain独自のcoalesced-global-K変種をリンクした。
BF16 Q/K/V/P、FP32のmatrix累積・online状態、32cell update順、全allocation/input、
workgroup128、SLM形状、barrierは同じ。prefix packを両経路の各timed callに含める。

8192 queries、24heads、head_dim256、width2051、physical32768、reverse page table。
各queryの選択はrotated uniform spreadで一意、全queryを通じ全32768cellを覆う。
同一入力が実モデルの全選択分布を代表すると仮定しない。モデルPPは未測定。

|経路|独立3processの中央値 ms|3processの中央値 ms|
|---|---|---:|
|現行|390.847012 / 390.771326 / 390.915277|390.847012|
|coalesced K|454.675873 / 455.253925 / 454.594058|454.675873|

各processに2warmup・10標本。順番はbase0/candidate0/candidate1/base1/base2/candidate2。
変更後は16.3309%遅かったため**不採用**。転送削減やGPU演算の変更ではなく、
global読み込みの連続化とtransposed SLM書き込みのlane配置を入れ替えた実験。
物理global traffic・SLM bank-conflict counterは測っておらず、遅延増を一つのunitへ断定しない。

全6processで全outputの有限性、guard、全input不変、最終repeatのsigned-zeroを含むbit比較、
独立4query/24576outputのBF16 oracleに合格。全出力FNV1a64と入力fingerprintは一致した。
保存frame同士の全byte比較はしていない。timed反復は最終frameを検証する。
先行するtraced2smokeとcoalesced単体11case（physical262144含む）もPASS。
所有終了・新規fault0、dumpなし。モデル全体の256Kや保存session RESTOREは未認定。

次は現行mappingのまま確率exp2を16/128threadsから分散するcomponent比較。
最大値計算・同じpのexp2・32値の逐次sum・FP32 state更新順を保存し、追加barrierの費用も
同じ計測に含める。並列化が速いとは仮定しない。この変更はまだ実装・計測していない。
モデルのclean中央値445.4 tok/sは今回の部品比較で更新していない。
