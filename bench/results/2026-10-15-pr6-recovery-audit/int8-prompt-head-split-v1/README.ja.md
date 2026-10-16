# INT8 prompt 8+4 head split v1 — 適用前の候補

2026-10-15 JST。**UNAPPLIED / UNBUILT / UNTESTED / GPU-UNQUALIFIED。**
Mainの公式bench.sh Direct0 full thinking 3 runsと並行するため、このディレクトリの
新規提案ファイルだけを書いた。製品ソース、CMake、既存fixture、binary、build、test、
GPU、server、profiler、cleanup、Gitには変更・実行していない。
実行したPythonは文字列から候補ソース・差分・hash台帳を作る生成器だけであり、
fixture、host test、コンパイラ、runtimeは起動していない。

`source.diff`は`src/kernels/sycl/prompt_attention.cpp`だけの差分。
`files/src/kernels/sycl/prompt_attention.cpp`が適用後の全体提案である。
既存`launch<Mode,Direct,Native>`本文、既存mode別admission、外部API、既定分岐を維持し、
別template `launch_mode1_head_split<32/64>`と明示opt-in分岐を追加した。
`generate_source.py`は同じ現行入力から提案と差分だけを再生成する。
他の提案/R605/registryを書換えていない。

選択条件はMode1 INT8 K/V、`STRATA_SYCL_PROMPT_DIRECT`がexact `1`、
`STRATA_SYCL_PROMPT_HEAD_SPLIT`がexact `32`または`64`の全て。
未設定/空/0/off/不正値/他modeは従来経路を使う。
従来Direct判定のprefix `1`規則は維持するが、`10`/`1x`では新候補を選ばない。
CH32は32セルの演算順を維持する比較候補で、有限値のbitwise一致を期待する。
これはソース設計の期待であり、コンパイラ・実機での証明ではない。
CH64はsoftmax/factor/PVの境界が変わるため許容差とモデル品質の別gateが必要。

現行実装のWGは**1 query × 1 KV head**。R605の「2 KV headsを1 WG」は誤り。
候補はそのWGを8+4 query headsの2 WGへ分け、local rows8、WG64を使う。
両WGは12 heads全体から同じQ scaleを再計算する。tile1の未使用4行はQ/P/stateをzero、
matrix collectiveと全WG barrierには全laneが収束する。外部head strideは12/24のまま。

`design-audit.ja.md`にindex、local_accessor算術、演算順、収束、残るcaller契約を記録した。
`files/tools/sycl/int8_prompt_head_split_probe.cpp`は独立FP64 attention oracle、全出力、guard、
入力不変、dispatch isolation、24-head ownership列挙、tail、共通scale、subnormal、
cap262144/ID262143を検査する**未実行**のcomponent fixture提案。
`generate_probe.py`は既存matrix-direct-probe-v1のhost/reference/owner実装から再生成する。
元fixtureは変更していない。候補CH32のみの`--validate32`と、CH64も含む`--validate`を分けた。

`main-controller-plan.ja.md`と`controller-plan.json`はMain向けの直列適用・build・probe・
モデル検証案。`probe-target.cmake`は手動target追加案で、実CMakeへ適用していない。
`input-sha256.json`は読んだ入力のhash、`artifact-sha256.json`は生成物のhashとbyte数。
モデル入出力tensorや測定結果のhashではない。binary/実測/cleanup/deletion manifestは
まだ存在しない。新しいrunや削除をしていないため、そのような証拠を捏造しない。

既存profileのINT8 Directは108 calls/同じ128生成IDs・仕事で44.066063496秒から
29.805022972秒へ減った。これは各1 profiler付きprocessのdevice duration合計であり、
clean benchmark・prefill専用時間・このhead-split候補の性能ではない。
Mainのclean Direct benchmarkは本提案作成開始時に未完。結果を先取りしない。
R600の12.073秒はDown GEMMではなくgeneric FP16 dequantというMain訂正も保持する。

モデルでの候補評価は最低32K実入力とagent sessionを含め、公式benchのthinking有効で行う。
物理full256K入力、境界消費、近傍の通常生成、状態/RESTOREは別検証が必要。
小さい物理poolへID262143をmapする論理index fixtureで代用できない。
採用・既定化・速度/品質/256K合格の主張は一切ない。

保持consumerはMainのhead-split実装レビューと最初の直列コンパイル/検証。
次の見直しは最初のcompile/host/device結果またはsource却下時。
本版を元の提案として保持し、修正案や失敗結果は別version/receiptへ記録する。
