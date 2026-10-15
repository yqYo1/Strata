# 選択的なIQ展開と実モデル形状記録

2026-10-15、feature/sycl。対象はArc B570 10 GiB / Ryzen 5600X / RAM128 GiB。
推論全体の性能採用は未確定。新しい展開処理と形状記録はどちらも既定OFF。

## 本体へ組み込んだもの

`STRATA_SYCL_PARALLEL_IQ_DEQUANT=1`で、GUのIQ3_S(21)、IQ2_S(22)、IQ4_XS(23)と、
generic/DownのIQ4_NL(20)、Q2_0(42)だけを出力要素単位のFP16展開へ切り替える。
ne64の独立3プロセスずつの部品測定で勝った組合せに限定した。
IQ3_XXS(18)は約9%遅かったため既存処理。他の役割・形式も既存処理。
F32出力、公開関数の全span/形状/overflow検査、queue/finish契約は維持した。
初期6形式案との相違はこのdispatchの絞込みで、数式・半精度変換は同一。

`STRATA_PREFILL_LAYER_MAJOR_SHAPE_CENSUS=1`は一つのlayer-major要求について、
層ごとのGU/Down形式、exact ne、route countと実GEMM呼出しの対応を記録する。
状態は4 MiB以下、出力1 MiB以下。issuerのhostカウンタだけで、queue操作、追加wait、
tensorコピーは増やさない。成功した外側要求の終了後にだけ完全な48層記録を出す。
借用pointerはchecked drainと元callback復元後、owner破棄前に復元する。
形式表とのjoin、bin総数・row総数・GU/Down呼出数は独立collectorでも再計算する。
これは構成上の512 expert常駐と論理byte数を示す。ハードウェアcache hitやPCIe実byteではない。

## 完了した確認

- 元の要素別decoderはCPU normal/ASanで503,808 blocks / 94,044,160 elementsを比較済み。
  本体headerとその検証対象headerは同じSHA256。
- censusの実C++ headerはnormal/ASan/UBSanのassertion fixtureを通過。
  tail、262144論理形状、overflow、欠落呼出し、出力上限、借用pointer例外復元を含む。
  追加の`/dev/full` fixtureで出力失敗も未適格になることを確認した。
- 実際の本体prefill.cppを既存SYCL compile databaseのflagsで独立objectへコンパイルした。
- 本体IQをコンパイルし、元のnativeとは別の実行ファイルをlink。link mapで旧IQ/prefill
  archive memberが選択されていないことを確認。元nativeのbinaryは書き換えていない。
- この実際の本体IQ objectを使う公開関数fixtureはquick42件/full48件でOFF/ON全配列一致。
  guard、入力不変性、F32と負例を含み、各プロセス正常終了・回収、新規GPU faultなし。

主receipt: `layermajor-census-host-v1.json`、`next-diagnostics-build-v1.json`、
`selective-iq-native-build-v1.json`、`selective-iq-component-{quick,full}-v1.json`。

## 未解決停止と診断側の修正

追加wide測定は31件後に既存OFF/type23/GU/ne16で期限切れ。
180秒監督終了付近にxe Engine resetがあり、原因は未確定。
`parallel-iq-timing-wide-v2.json`と`parallel-iq-timing-wide-failure.ja.md`を保持する。

v3診断はAPIログのburstを一度の256 KiB read上限で拒絶した収集側不具合により失敗。
nativeはretire終端まで進んだが監督SIGTERM/exit=-15であり、完走と扱わない。
v4は64 KiBずつ、総量4 MiBまでのstream読取りへ修正。元1,333,588 B traceの70 phase、
partial line、重複sequence拒絶、partial line上限のCPU回帰確認を通過。
同じv3 binaryの独立v4診断はexit0、全出力検証・70 phase・解放完了、新規fault0。
停止再現なしは元wide障害の解決を意味しない。次の停止時は45秒semantic quietで一度だけ
所有nativeの全thread stackを取得し、元の終了状態を別に保存する。

## 次の判断

現在の本体nativeで、既存失敗調査と同じ32089入力/gen24を一件実行中。
結果は`selective-iq-model-v1.json`へ保存し、完全な形状記録をmodel sidecarと照合する。
この実行は診断用でclean性能対ではない。その後、同条件のclean OFF/ON反復、
bench.shのlong/agent/thinking有効長生成、decode単独の反復を行って採否を決める。
full physical262144、RESTORE、task能力、1000/70目標達成はいずれも未証明。

R621–624は別ファイルに保存した。R621のBF16表記とCUDA combine参照はR623で訂正し、
実際はFP16 mixed_hとSYCL postopsの10-slot FMAである。全prefix MoE遅延実行は候補だが、
RAMからrouteごとに入力を送る方式やDm退避はPCIe量を増やすため、resident waveの容量と
数値誤差を検証してから判断する。decodeは既存cache容量とMTP幅の実比較を優先する。
