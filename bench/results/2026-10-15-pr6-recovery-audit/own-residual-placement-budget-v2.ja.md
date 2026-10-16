# FP16中間状態の配置とstagingの容量確認

固定入力は78444トークン、batched行は78443。N=2560、HC=4、D=N×HC=10240。
Rには4本のhyper-connectionの状態があり、postopsのmixが4本を参照する。
中間状態1行をN要素だけと数えることはできない。以下はソースからの計算であり、VRAM収容や精度の実測結果ではない。

| 対象 | 保存形式/GPU行 | GPU状態+staging (GiB) | host (GiB) |
| --- | --- | ---: | ---: |
| current-NLL | fp32/32768 | 1.250000 | 1.742363 |
| current-NLL | fp16/32768 | 0.781250 | 0.871181 |
| current-NLL | fp16/78443 | 1.496181 | 0.000000 |
| physical-row-capacity-only | fp32/32768 | 1.250000 | 8.750000 |
| physical-row-capacity-only | fp16/32768 | 0.781250 | 4.375000 |
| physical-row-capacity-only | fp16/262144 | 5.000000 | 0.000000 |

FP16混在配置では8192行分のGPU stagingが別途160 MiB必要になる。
現在のFP32/GPU32768行に対して、FP16/GPU32768行はstaging込みで480 MiB減る。
FP16全GPU保持ではstagingは不要だが、FP32/GPU32768行より252.089844 MiB増える。
全GPU化を単にGPU保持行数だけから予測せず、stagingの置換も含めて数える。

host残存分は47の層境界で書き戻し・読み込みが発生する。この論理バイト数はJSONに記録した。
全GPU保持でもGPU内のpack/unpackやlive FP32作業領域は必要である。
固定のlayer cache、prefill scratch、KV、head、native bankなどのVRAMは表に含めていない。
したがって表は収容可否の証明でも、帯域から導いた時間や速度向上の予測でもない。

physical262144行は容量算術の別ケースであり、現在のNLL試験が物理境界を消費したという意味ではない。
FP16品質と全GPUの実際のadmissionは、実行条件を固定した別のarmで確認する。
現在はFP32-aのみ実行中であり、品質採用・性能採用はしていない。
