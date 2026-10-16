R970判定: ホストroute map単体は、今回の検査範囲で正しさの欠陥を見つけず、engineへ渡せる候補。512 expert・top-k=10の全ID検査、token内重複拒否、安定したsource/slot対応、route総数保存、8K job分割が実装され、現行combine/gatherのslot向きとも一致する。

Clang ASan/UBSanのv2 receiptは、合成IDによる40ケース・2,892,850 slotと14拒否条件のホスト検査を示す。実router出力、GPU演算、数値、DMA/queue寿命、実allocation境界、fit、262144-context、RESTORE、速度は証明しない。shared planはsource vectorを固定するが、raw upload pointerの寿命はchecked drainまでcallerがownerを保持する責務。

R969で欠けていたhost route contractは埋まった。Mainは2-piece GPU windowへ接続し、実allocationとID/source bounds、両HC writeまでのowner保持、callback/MTP境界を検証する。checkpoint gateは維持。新規researchは不要。
