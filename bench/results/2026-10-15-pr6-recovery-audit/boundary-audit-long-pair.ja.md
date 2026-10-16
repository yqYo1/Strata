# 34K residual配置差の層別比較

2026-10-15 JST。Main同binary b6838d28、34603input/34602batched、chunk4096、FP32、
MTP675MiB退役ON、GPU residual固定4096と自動24576を別processで実行した。
両runは正常exit/reap/session empty、freshGPUfault0、dumpなし、boot不変。
`analyze_boundary_pair_v1.py`を両run終了後のexclusive lock下で実行した。

全19選択位置×48層について、run内source→store→次層inputがactual memcmp一致。
二runのraw計算出力はlayer0..25の全選択位置で同一だった。最初の選択位置の差は
**layer26、relative row24575**。float index0はfixed0.03594839200377464に対し
auto0.03594634309411049。このrowのmaxabsは1.6435980796813965e-5、layer26の選択範囲最大は
1.8215738236904144e-5だった。layer26以後は24575,24576,32767,32768,34600,34601の6位置が
異なり、最終層の選択範囲maxabsは0.12206459045410156へ増えた。

fixedは旧controlの全selected/first logitsと生成IDに一致。autoは旧controlとも、旧非診断
44cafcc2 autoFAILのselected logitsとも異なる。生成1IDは1596で同じ。autoのnumerical FAILは
維持し、元FAILのreceipt/rawも上書きしない。新auto receiptはcomplete=trueが採取完了を表し、
passed/numerical_passed=falseは数値比較不合格を表す。offline comparatorのcompleteは解析終了。

19位置の観測であり、全tokenの最初の差や、全GDN入力の一致は分からない。layer26はGDN層だが
GDN/HC/FFN/MoEをこの結果だけで原因確定しない。各run内のcopy一致も採取範囲に限定する。
診断D2H/drainが同期条件を変える限界を残す。次は同層のGDN入出力、HC/FFN/router、
MoE combineの有限phase比較で大枠を分ける。R585/586のsource reviewは具体的UBを発見していないが、
実行の正しさや故障不在をsourceだけから証明したものではない。

raw consumerはMain phase-localization、各arm37,355,520 B、二arm計74,711,040 B。
旧非診断autoのlogitsは異なる数値機構の比較consumerとして保持する。次reviewはphase比較が閉じた時。
速度・品質・physical256K/RESTOREの採用証拠ではない。prefill1000/decode70目標は未達。
