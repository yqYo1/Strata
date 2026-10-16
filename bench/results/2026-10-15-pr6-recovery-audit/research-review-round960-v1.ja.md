# R960: R959 split-barrier report review

- R959は1,489語で指定予算内。R807/R810のloop/convergence、R811のblock2D/prefetch、R924の別QSA・明示SLM kernelと重複せず、現行vLLM Xe2 K-loopのSPIR-V contractを判定した。`SemanticsNone`でWorkgroup execution rendezvousは残るがmemory fenceにはならない。削除・scope縮小を勧めない結論は妥当。
- 固定TLA sourceでは2D loadはregister fragment宛て、prefetchはnull destination、reorder/DPASはsubgroup fragment path。追加すべき反証はTLA `xe_rearchitecture.md` の注意で、VNNI loadとDPASの間にnative B型演算があるとprivate-memory trafficが起きる場合がある。現loopのreorderが該当するかはlowering未確認。private spillはcross-subgroup SLMやbarrier必要性の証拠ではない。
- R959の適用範囲は修正が必要。adapter gateはBMG G21に加えG31も許すが、G31の実行資格は示していない。また一般のnon-None semanticsをrelease/acquireと断定せず、指定flagによるstart/end memory effectと書く。
- local dense-GEMMのlowering・correctnessは物理262144-contextのmodel consume/RESTOREを資格付けしない。後者は未認定の別gate。MainのQSA subgroup-softmax candidateは別kernelなのでR959の結果を適用しない。
- 新規調査は不要。MainがこのXe2 GEMMの変更を具体化した後、configured compiler/device loweringがあれば再検討する。現CPU benchmarkのprofile/censusは未実施との指示で、ここでは確認していない。index提案は`research_review_round960`、`schedulable_now=false`。
