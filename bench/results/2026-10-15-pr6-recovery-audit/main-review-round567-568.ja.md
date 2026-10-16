# VRAM制約とmatrix head tileのレビュー

2026-10-15 JST。Luna2件を並行して受領した。両調査中、Mainは実モデル比較と
別Sol proposalのレビュー・適用・buildを行った。調査agentへ途中介入はしていない。

実モデル34,603 input/34,602 batched tokensの自然経路比較は完了した。
248320 logitsと541位置の最終残差は旧control6180/当時のcurrent2beでbitwise一致。
matrix108 calls/415224 query rows、fallback0であり、このcaseの実prefillはmatrixを使う。
原recordと`natural-layout-long-pair-summary-v2.json`を参照。これはbench.sh速度比較ではない。

R567の追加回収候補はMTP experts675MiB、decode cache約0.26GiBなどであるが、
現APIに安全な一時退役と再構成がない。`bind`だけではgraphのborrowed pointerは退役しない。
native/canonicalのloaderはserved_namesをskipしており、重量値だけで重複と断定しない。
初版layer-majorはそれらを解放しない。prompt時free1960MiBとrounded blob2.66MBの
単純差し引きでは、臨時full cache後に大きなGPU残差が置けない制約が残る。
実装のauto planが計算する正確なbytesをGPU結果で確認する。
R567の34,603行算術は入力全体を想定したもの。今回のnative prefill対象は34,602行であり、
残差全体は1,417,297,920B、host-only47層往復は133,226,004,480Bとなる。
full262144行は10GiBのまま。論理256Kとresident枠を混同しない。

R568の8-head×CH64 head tile案は、CH32の65loopを各WG33loopへ減らす一方、
同じKVを二つのWGが読む。matrix issueのpadding/hi-loを含む仕事は約1.54%増え、
requested KV bytesは倍になる。速さの根拠にはまだならない。
mode1・Directのlocal accessor合計53876BはMainも計算で照合した。
原文のindirect62068Bと「add8192B」は4B多い。Directのfinal1floatを8×256floatsへ
置き換えるため加算は8188B、**indirect62064B**が正しい。原文は保持する。
新しいgroup geometryとQの共通scaleの扱いは未実装。tile別scaleは丸めを変えるため、
math同一・bitwise一致を前提にしない。現在のmatrix route観測は候補実装の合格ではない。

当面はlayer-majorの数値とowner検証を先に完了する。matrixの既存Direct optionとhead tileは、
その後に同一入力の独立component比較で評価する。prefill/decode採用は引き続き別に判断する。
