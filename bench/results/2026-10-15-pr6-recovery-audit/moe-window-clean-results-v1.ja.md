# MoE窓のclean反復比較

2026-10-17、Arc B570、driver1.17.39395+14、同一binary243c34d6…。
公式bench.shのprefill-only、39,262入力token、context65,536、chunk8192、
GPU residual32,768、BF16/vLLM/optimized QSA、thinking有効、reuse0。
TRACE/UR layer/logging/CPU・queue profileを外した独立6processを
0,2,2,0,0,2の順に実行し、全試行正常終了、所有process終了、fresh fault/dumpなし。

| 条件 | 各PP tok/s | 中央値 |
| --- | --- | ---: |
| window0 | 452.0,464.1,464.1 | 464.1 |
| window2 | 491.7,491.8,490.8 | 491.7 |

中央値は5.95%改善、prompt時間中央値84,594.8→79,852.1ms。
有効時の3sampleはいずれも無効時の最大値より高かった。
3回ずつのこのcohortは将来の全ばらつきを無くすものではない。
最初の無効時452.0も除外せず保存し、最良値だけを選んでいない。
24生成tokenは完了確認でありdecode性能や一般品質の合格判定ではない。
600/1000tok/sとdecode70tok/sは未達。全state・physical262144・RESTORE等も未資格付け。
候補はDraft PR32に保持し、現在のprimary engineへ採用したとは扱わない。

旧v1のSTRATA_TRACE=0はpresence判定でtraceを有効にしたため失敗のまま除外。
v2は固定portのbind失敗でserver/GPU起動前に終了したため失敗のまま除外。
v3は変数を除き、各sampleでOSが選んだ空きloopback portを使い、6sample全部を保存した。
元receiptを合格へ変更せず、無効試行をこのcohortのsampleへ混ぜていない。

診断でMoEが短縮してもQSAの非排他区間は約24.7秒（attention約18.8秒）のまま。
次は現行row-major score33でsoftmax max/sumをSIMD16 subgroupへ分けるprivate候補を
compileし、既存の独立数値・bounds/guards・physical262144単体fixtureを実行する。
現行配置/16bit storage/FP32 online stateを保ち、加算順だけの変更を別に評価する。
またupload slabが通常new[]であるsourceと、source copy/upload submit各約4.8CPU秒を
host USM候補の根拠として記録する。runtime staging原因や除去可能wall時間は未測定。

再現: run_moe_window_clean_cohort_v3.py（新規run専用、既存receipt再利用不可）。
検証解析: analyze_moe_window_clean_cohort_v1.py。
個別receipt: moe-window-clean-window{0,2}-r{0,1,2}-v3.json。
cohortと解析: moe-window-clean-cohort-v3.json、moe-window-clean-comparison-v1.json。
成功raw protocol/重複traceのretention reviewは閉鎖済みcohortを対象に、次のGPU実行と直列化する。
