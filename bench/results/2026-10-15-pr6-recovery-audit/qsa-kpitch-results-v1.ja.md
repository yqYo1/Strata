# BF16 attentionの共有メモリ行ピッチ比較

2026-10-18 JST、Arc B570 10 GiB、Ryzen5600X、RAM128 GiB、
xe／Level Zero driver1.17.39395+14。現在のrow-major attentionから、
共有メモリのK/V行ピッチと対応する配列容量・アドレスだけを変更した。
global側の読み込み担当、BF16ビット、QK/PV、FP32オンライン演算、
Cells32、WG128、SG16、score stride33、バリアは維持した。
独立したテスト用ソースであり、製品ソース・既定値は変更していない。

K/Vピッチは32/256、36/256、32/260、36/260、40/264の5方式。
公式matrix拡張のBMG向けstride・アラインメント条件を確認し、全producerと
consumerの位置が配列内で対応することをMainが全数確認した。
R1054／native R1055でも、型付き配列・最大アクセス位置・バリア・必要容量に
source contract上の欠陥は見つからなかった。実際の物理bankやoccupancyは
このソース検査では分からない。

|K/Vピッチ|共有メモリpayloadのソース上の計算値 bytes|配列要素数 BF16|
|---|---:|---:|
|32/256|28224|8192|
|36/256|30272|9216|
|32/260|28480|8320|
|36/260|30272|9216|
|40/264|32320|10240|

この値はcompilerによる隠れた確保・paddingや実際のresidencyの実測ではない。
既存のlocal-memory admissionにも新しい容量を反映した。

5方式それぞれでsmoke2例とcomponent11例が成功した。NaN、無効幅、
欠損ページ、prefix全要素、ガード、入力不変、alias・domain・extent拒否を含む。
物理262144セルの例は1クエリであり、256K実モデルや既存の大きなクエリ集合の
数値失敗の解決を証明しない。元の失敗結果は保持した。

次に、同じプロセス内で引数により5方式を選び、順を入れ替えて各6回測定した。
環境変数は起動前に固定。78453セル、8192クエリで、admission・prefix展開・
attention・完了待ちのwall時間を測り、入力転送と結果比較を除外した。
全APIトレースを時間測定時には無効にし、parameter validationは維持した。

|K/Vピッチ|33セル選択・gather 中央値 ms|2051セル選択・gather 中央値 ms|2051セル選択・clustered 中央値 ms|
|---|---:|---:|---:|
|32/256|13.943782|419.751401|299.191100|
|36/256|13.945475|416.419903|297.206408|
|32/260|14.002393|420.927044|301.833226|
|36/260|13.952073|417.663048|300.041888|
|40/264|21.778360|673.624987|477.723635|

K36/V256は2051セルの条件で0.79%／0.66%の短縮だった。V側の変更や両側の
変更は大きな改善にならず、40/264は約60%遅くなった。独立プロセス間の
再現性や実モデルのPP差を証明した結果ではなく、採用・既定値変更は保留する。
この差だけからbank conflictやSLM占有率が原因だとは断定しない。

90個の大きな入力の計測すべてで50,331,648出力要素を全数比較し、同一ビット
だった。小さなfixtureにも30計測があり、同じ検査が成功した。これは配置だけを
変更したコードの検証であり、異なるLLM実装や精度全般の同一出力条件ではない。
12個のGPU検証プロセスが正常に終了し、新規GPU障害0。

K読み込み順と今回の行ピッチだけでは大幅な短縮を得られなかった。
次の具体的な確認は、実モデルで支配的なprefix attentionの生成GPU命令が
native DPASを使うか、実際のGRF・spill・SLM確保が何か、40/264の大きな段差を
説明するcompiler/resource変化があるか、である。`ocloc`と`libiga64`はこのPCに
存在するが、まだnative binaryの取得・逆アセンブルやリソースの確認は行っていない。
SYCL persistent cacheを無効にした今回のrun cacheにも保存物はなかった。
新しい診断はimmutableな起動環境と専用ディレクトリ・容量上限でMainが行う。

全サンプル・標準偏差・レシートの識別は
[qsa-kpitch-component-comparison-v1.json](qsa-kpitch-component-comparison-v1.json)。
ソースと差分は[own-qsa-kpitch-v1/manifest.json](own-qsa-kpitch-v1/manifest.json)、
実行前の全数アドレス検査は[qsa-kpitch-source-admission-v1.json](qsa-kpitch-source-admission-v1.json)。
ビルドは`build_qsa_kpitch_v1.py`、数値検証は
`run_qsa_kpitch_v1.py smoke|component baseline|k|v|both|wide`、
反復測定は`run_qsa_kpitch_perf_v1.py tiny|perf`。
保存先は1回専用で、再測定時は新しい版を用意して元の結果を変更しない。

source admission時点ではnative R1055レビューが進行中だった。Mainは自身の
全差分・公式規約・全数位置検査と完了済みR1054の照合から段階的なGPU検証を
開始し、その後R1055も同じソース契約を確認した。各時点の記録を変更して
後からレビュー完了扱いにはしていない。

成功した生ログと用途を終えたK読み込み順fixtureを157ファイル、
12,128,838バイト退役した。元の成功・失敗レシートは不変。
[削除記録](qsa-kpitch-retention-v1.json)にパス・ハッシュ・理由と次のconsumerを保存した。
現在のpitch fixtureとfull engine／archivesは、上記native命令・resource確認に
必要なものとして保持する。GDNの次の比較境界は
[gdn-current-contract-next-v1.ja.md](gdn-current-contract-next-v1.ja.md)に記録した。
