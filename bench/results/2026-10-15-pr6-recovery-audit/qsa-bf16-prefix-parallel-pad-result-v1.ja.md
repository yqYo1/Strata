# 確率分散＋score行幅33の組合せ（部品測定）

各arm3独立process、各2warmup/10sample。同一binary/入力、packを毎回含む。基準process中央値390.884434/390.791151/390.7919165ms、組合せ366.978047/367.074047/366.840404ms。各中央値の中央値390.7919165→366.978047ms、遅延6.09375%減。

全6processの独立4query oracle、全出力finite、guard、入力不変、repeat bitsがPASS。全出力FNV一致は保存tensorの全byte比較を意味しない。小さな初回URトレース・物理262144を含む11ケースはPASS、全process正常終了/ownership closed/新規GPU faultゼロ。モデル全体/RESTORE/一般品質は未認定。

別cohortの行幅33だけ（基準390.857487→332.9269265ms、14.8214%減）の方が速い。組合せはモデル統合候補から外し、同じ32セル逐次確率計算順を保つ行幅33だけを次のエンジン比較に採る。SLM bank conflictやSFU使用率の改善とは断定しない。
