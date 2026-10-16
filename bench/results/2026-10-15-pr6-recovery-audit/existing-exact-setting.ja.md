# 既存single-exact設定の実モデル比較

2026-10-15 JST。同じb6838d28 binary・34,603入力・max128を2回と513入力/max8。
変更はSTRATA_IQ_SINGLE_EXACT=1。FP32残差固定4096、MTP4、pool5+host等はbaselineと同じ。
3件正常終了、freshGPUfault0、boot不変、owner終了確認済み。
CPU fixtureのtypes18/21/22・384件bitwise一致は既存receiptを参照する。

| sample | baseline / exact decode ms | ID/work | GU thread時間比 exact/base |
| --- | --- | --- | --- |
| 長入力初回 | 25412.4 / 25182.9 | 全128ID・処理量一致 | 0.983875 |
| 長入力二回目 | 19028.8 / 18141.6 | 0始まりID46から異なり処理量も変化 | 0.942015 |
| 短入力 | 2345.0 / 2344.3 | 一致 | 1.002729 |

初回のIQ3_S GU thread時間は比0.947833、IQ3_XXS0.991981、IQ2_S1.000347。
単体IQ3_Sの14–15%短縮をモデル全体へ外挿できない。初回loopは約0.9%差に留まり、
二回目の約4.7%差はwork不一致のためkernel効果としない。CPUthread合計と重なるwallspanは加算しない。
生成の相違だけで品質劣化と判定するものでもない。未変更演算のCPU単体一致と実LLM品質は別gate。
反復clean公式benchと独立品質評価を欠くため、設定はdefault OFFのままとする。

analyze_existing_exact_v1.pyとexisting-exact-setting-summary-v1.jsonに全3標本・全format/role集計・
元receipt hashと判断を保存した。元structured receiptは変更しない。

# 次の境界診断と並行proposal

layer26-phase-audit-v1は未適用。Main CPU fixtureは通常とASan/UBSanで完走し、
独立literal raw byte数、grouping境界・欠損・重複を確認した。比較器は同一/1byte差/切断/余剰/
footer・dtype・extent・selection・offset・low schema異常等13件を期待どおり区別した。
phase-audit-host-v1.jsonがその証拠。これはSYCL実行・正確性・性能の合格ではない。
比較器のfirst_selected_differenceはrow/phaseソート順なので、演算の最初の差を示すと解釈しない。
実GPU比較前に記録順も明示する版を別ファイルで作成する。

bounded-kv-copy-groups-v1はMainレビュー待ちの未適用proposal。全hitでも8192groupを発行する
現行copyのworkgroup上限を変更する候補で、copy対象の切捨てをしない。実性能は未検証。
owned-decode-profile-window-v1はsource.diff無し。実providerにITT exportがなく、動的setenvの
安全契約も未確認のため収集を開始していない。安全な制御方法をR591で追加調査する。
R589のQSA score-only FP16/BF16 GEMM案も未実装・未測定・品質未評価。
R590 CPU IQ2_S、R591 profiler制御のreadonly Luna調査を並行して開始した。

次はphase診断の本体適用・CPU既存fixture・SYCL build、小入力OFF/ON、34K固定/autoの同binary比較。
GPU/test/buildはMainが直列管理する。fullphysical256K/RESTORE、正式bench更新、目標1000/70は未完了。
