# 34K fixed4096 boundary gate

2026-10-15 JST。binaryb6838d28、34603input/34602batched、chunk4096、FP32、MTP675MiB退役ON、
GPU residual4096行固定、auditON。旧controlと全19selected logits、first logits、生成1IDがbitwise一致した。
48層×19rowのoutput912、store893、nextinput893を取得し、保存/読み戻しが全memcmp一致した。
raw outputは37,355,520 B。元の自動24576-row FAILは変更しない。

正常exit/reap/session empty、freshGPUfault0、dumpなし、boot不変。MTPrestore158.443 ms。
追加同期付き診断であり速度採否には使わない。次は同じbinaryで自動GPU-prefixを採取し、
`analyze_boundary_pair_v1.py`で最初の異なるselected layer/rowを比較する。
rawはこの未解決比較のcanonical fixtureとして保持し、比較と原因特定後にretention reviewする。
