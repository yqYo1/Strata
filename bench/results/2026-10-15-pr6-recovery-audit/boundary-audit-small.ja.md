# Boundary diagnostic small model gate

2026-10-15 JST。Main新binary b6838d28、513 input / 512 batched、chunk257+255、context262144設定。
FP32 auditOFF、FP32 auditON、FP16 auditONを別processで順番に実行した。

| 条件 | 同dtype旧controlと全selected/first logits・生成ID一致 | output / store / input行数 | MTP restore ms |
| --- | --- | --- | --- |
| fp32-off | True | OFF / OFF / OFF | 156.555 |
| fp32-audit | True | 432 / 423 / 423 | 282.114 |
| fp16-audit | True | 432 / 423 / 423 | 161.034 |

ON両dtypeで48層の保存・次inputが全actual memcmp一致。FP32とFP16の相互bitwise一致を要求したものではない。
全process正常exit/reap/session empty、boot不変、fresh GPU fault0、dumpなし。
診断無効の出力一致と診断内のcopy coverageを確認した。追加drainがraceを隠す可能性は残る。
短文は数値・寿命gateのみで速度比較には使わない。34K拡大prefix、品質、正式性能、physical256K/RESTOREは未qualified。

raw consumerはMainの診断実装reviewと34K境界比較。旧selected controlsはcanonicalのまま保持し、
成功candidateの同一logits/rawは長い比較後のretention reviewで再検討する。
