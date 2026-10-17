# 現行row-major QSA subgroup: clean反復測定

2026-10-17、Arc B570、driver1.17.39395+14、LLVM22.1。
同一binary、8192 query・width2051・physical32768、各process2warmup+10測定、
base,candidate,candidate,base,base,candidateの6独立process。詳細ログ・validationなし。

| 経路 | processごとの中央値(ms) | 中央値(ms) |
| --- | --- | --- |
| 現行score33 row-major | 333.249416, 333.396278, 333.318242 | 333.318242 |
| subgroup max/sum | 322.240273, 322.336910, 322.390725 | 322.336910 |

latencyは3.29455%短縮。全candidateのprocess中央値が全baselineより小さい。
全60sampleはcohortと個別receiptに保持。両経路で独立BF16 oracle、全finite/guard/input不変、
同経路repeat bit検査PASS。同経路process間hashも一致。和の順番が違うため腕間hash一致は
事前基準に含めず、独立oracleの許容値を変更していない。
全6process正常終了・所有終了、fresh fault0、dumpなし。

合成selected IDはuniform spreadで実モデルの選択局所性ではない。host wall timeであり、
EU/DRAM/PCIe飽和やモデル全体のprefill改善は示さない。一般品質、全physical262144の
実モデルconsume/RESTORE、cancel/reuseは未資格付け。
次は検証用エンジンへ独立した選択肢として組み込み、小さい初回確認後に実モデル比較する。
再現: python3 -B bench/results/2026-10-15-pr6-recovery-audit/run_qsa_rowmajor_subgroup_cohort_v1.py
既存receiptを上書きしないため同名での再実行は禁止。
