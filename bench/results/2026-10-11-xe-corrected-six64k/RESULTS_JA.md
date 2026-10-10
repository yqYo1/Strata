旧39bd由来Xeにpublication/lifetime対策を入れた実64K入力・4Kchunk・64出力のquiet比較を各3回完了。実行順B1/X1/X2/B2/B3/X3。全6件exit0、直前直後の16384整数一致、new fault/reset/dumpなし、source/binary安定。各arm内64tokenとLP5は3回同一。

中央値はbaseline prefill201.33 / decode8.86 tok/s、修正Xe prefill205.20 / decode10.73 tok/s。prefill paired差は+0.28%/+5.32%/-1.55%で一貫しない。decodeは各pair+31.08%/+20.97%/+18.93%で有利な信号。ただしcache/ring/host USM等が異なり、arm間の出力は5token以降分岐、MTP37/78対40/72。実装単位の速度差であり、特定同期やCPU最適化の寄与、能力同等、採用を証明しない。全個別値と範囲はsummary.jsonに保存。

長入力診断1件はLevel Zero/UR loggingありのため速度比較不適格。256Kは要求容量だけでfull262144の使用/restore証明ではなく、全候補未採用。従来403.54 tok/sのbaselineは別入力fixture/8Kchunkの1件で今回から除外。

最新Xe e32b8b05は40commit更新。stock prefillはfaultした39bdと同一なので未対策再実行を拒否し、対策を移植して全source/dependencyを新規buildする。新GPU/host doorbellのvolatile+fence同期は、実B570 capabilityと全経路fallbackを確認・対策後にGPU実行。公式v0.1.42はその後、未変更版と統合版を別測定。サーバー全機能を取り込み、推論変更のprefill/decode採否は個別に判断する。
