# QSA subgroup 実モデル初回診断

2026-10-17 Arc B570/driver1.17.39395+14/LLVM22.1、候補f685b16a。
実エンジンbinary24ea1edf88570c456df2364594662c1f77e8b7bae0db8c17e981cfffe0bfbad9。
公式bench.shのprefill-only、実入力39262/read39262/reuse0、context65536、
chunk8192/RGPU32768/window2、thinking有効・24生成。
STRATA_TRACE=1とUR parameter validationをexec前に固定し、詳細API traceは無効。
actual wrapperのbinaryと選択env、nativeのscore_pitch33/row-major/softmax=subgroupを確認。
全request完了、server/native exit0、所有終了、fresh fault0、driver dumpなし、boot不変。

診断ログのPP492.0tok/s・decode8.3tok/sは計装付きかつ24生成のみ。
性能の採用比較やsustained decode/一般品質には使わない。
単体の3.29%短縮が実モデルにも出るかは、同一new binaryのoptimized/subgroup
各3独立processを交互に測って判断する。研究wave984-986のnative review後に実行予定。
物理262144モデルconsume/RESTORE、cancel/reuse、一般品質と採用は未資格付け。
