# 公式unitraceによるKVグラフ部品の計測

Arc B570、公式Intel PTIの固定commit
`6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033`から以前ビルドしたunitraceを使用。
バイナリとライブラリ、XPTIのhashを再確認した。
`check_kv_resolver_profile_v1.py`で既存の720サンプルfixtureを変更せずに
`unitrace --device-timing --result-dir ...`配下で実行した。
詳細なChromeイベント、PMU metrics、動的pause/resume、製品への同期追加は使っていない。
固定版Linux CLIの非metrics/非KMD経路は同じPIDでnativeをexecし、
Mainのpidfd ownerが実際のnative終了を観測している。

全720件のoracle一致、正常終了・回収、新規GPU障害0を確認した。
`kv-resolver-profile-v1.json`が元の表と設定を保存し、
`analyze_device_profile_v1.py`が表の合計とグラフ再生回数を照合する。

| 記録された処理 | 回数 | GPU duration合計 |
| --- | ---: | ---: |
| 順序付き検索・CLOCK | 756 | 2.800183秒 |
| 従来コピー | 378 | 1.606382秒 |
| 512グループのコピー | 378 | 1.717001秒 |
| 初期化・照合等のM2Dコピー | 7,518 | 8.167570秒 |
| 初期化・照合等のD2Mコピー | 8,820 | 6.638158秒 |

検索は3形式×6ケース×(warmup2+計測40)=756回であり、
コピーも両armが各378回で一致する。capture時の36回だけを計っている状態ではない。
これはグラフ部品における可視性の確認である。

表は全ヒットから8192ミスまで、FP16/int8/Q4、両arm、warmupを合算したもの。
検索が常に支配的、int8全ヒットの検索が3.7ms、コピーに上表の割合で
decode全体の時間がかかる、と解釈しない。初期化・照合用の転送が多数含まれ、
durationの総和はGPU busyの時間和集合やモデルのwall critical pathではない。
候補コピーの採否は別のclean ABBA結果で決め、今回のprofiler表を速度比較には使わない。

次は現在の製品バイナリの34,603入力・128出力・通常MTP4を同じ方法で取得し、
実モデルでのカーネル時間と既存のCPU service記録を対応付ける。
全体aggregateでは共用カーネルのprefill/decode分離ができない点も記録する。
必要になった限定窓は別のdefault-OFF診断案で対応し、完全なphase計測を既成事実にしない。
