# Residual boundary audit proposal review

2026-10-15 JST。`layer-major-boundary-audit-v1/source.diff`は未適用。
現製品binary44cafcc…は変更しない。新decode ledger GPU job中にbuild/test/cleanupを行わない。

19行×48層の有限raw出力と、run内source→stored→nextinputの実memcmpは、配置に伴う
FP32差の局所化に適している。FNV64だけをcross-run同値の証明にせず、raw bytesを比べる。
元の24576-row FAILをPASSに変える変更ではない。

Mainが適用する前に以下を直す。

- helper constructorでも`0<n<=262144`、`0<=pos<=262144`、`n<=262144-pos`を確認する。
  現callerでは既に検査するが、helper単独でも`pos+row`のoverflowを防ぐ。
- FP32のexpected row保存はfloat要素代入から`memcpy`へ変える。finiteの実modelでは同じでも、
  signaling NaN payload等を含むexact-bit fixtureで余計な浮動代入を導入しない。
- false/exceptionでのaudit scratch寿命は外側のchecked leaseで保持される。source/store/input
  hookに入る前にallocationとfile writeを終え、copy後のthrowは必ずchecked drain後に限定する。
  現案のcopy lambdaはenqueue結果にかかわらずdrainする点を維持する。
- completeは採取coverageであり、memcmpがdifferentならMain gateは不一致と記録する。
  CSV/shape/予算/I/O/不完全sequenceのCPU fixtureはMain管理下で実施する。

追加D2Hとqueue drainは同期条件を変える。診断で差が消えた場合も、元の非診断経路の
正しさや問題解決を証明しない。allocation量/アドレス/library algorithmが変わる可能性と、
別queueやmapped host inputの時間依存を分けて調べる。
最初に各layerの計算output自体が変わる位置を見つけ、そこから前の必要部分を調べる。

`decode-cache-prefill-release-v1`の約260 MiB退役案も未適用。34K FP16は675 MiB返却だけで
全residualが収まる見積もりだが、その大きいprefixの数値gateは未通過である。
追加cache退役を先に混ぜて不一致の要因を増やさない。候補の実装/測定/adoptionを区別して保持する。
