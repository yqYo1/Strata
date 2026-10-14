# 主ブランチ・実装系統の再点検

2026-10-14追記: [セッションと元測定の再照合](../2026-10-14-performance-regression-audit/README.ja.md)
で、10月5–9日のこのfork自身の速いNiko系統が未統合と確認した。
以下の「主の速い経路を残す」は当時の4fa/root系統だけの点検であり、Xe前の最良実装を
保持した証拠ではない。667–694のXe値の区別は正しいが、旧449.7/645.8を見落としていた。
原報告は当時の境界のまま残し、訂正と次の比較はリンク先を参照する。

この個人forkのベースと唯一のアップストリームは Niko1221/Strata。
このPCのIntel Arc B570、Ryzen 5 5600X、RAMへの最適化が目的で、上流への貢献は予定しない。
CUDA/HIPの推論実装と検証は対象外。XeStrataは比較・参考先であり、全機能やエンジンを
同期する義務はない。優れた部品を移植する場合も主実装との個別比較が必要。

## 確認した状態

- 主作業 `feature/sycl`: `4fa478f50f8da396e651e3ae862f8b1849d49b54`、表示版0.1.38。
- forkの `main`: `beddab2fd6b61631d73b39b7484bc9a585ebe1de`、表示版0.1.39。
- 最近のXe由来実験: `28a7b9ae7ca5eba28eecd65f6453a7479c04fce8`。
  Xe 0.1.40.2系にNiko 0.1.43の非推論変更の一部を移した別の実装系統。
- 正確な枝、元SHA、worktree、未コミットファイルは [branches.json](branches.json)。
  版表示と履歴上の包含は、機能・推論実装の採用証拠ではない。

最近の667～694 token/sはXe由来の実験値で、主ブランチの現在の性能ではない。
主ブランチの新しい同条件測定は未実施。以前の性能記録は元のsource・設定・測定方式で保持し、
未比較の値で置き換えない。[reference-measurements.json](reference-measurements.json) に
閉じた公式benchの全個別値と限界を記録した。実入力32049、出力24であり、厳密32768入力や
256Kの物理末尾、RESTORE、モデル全体の品質を合格としない。decodeが対照側でも低下したため
bulk転送の正式採用は保留。prefill改善だけでdecode採用を決めない。

## 実装と機能の点検

主ブランチのruntime、prefill、decode、CPU/SYCL、expert source/cache、CLI/server/setupの
実ファイルを [primary-component-source-pins.json](primary-component-source-pins.json) に固定した。
この点検では推論コードを変更していない。Xeのkernel群を全置換した実験を部品採用と呼ばない。
個別候補と検証不足は [decisions.json](decisions.json) に記録した。

主とmainには共通祖先から143/285の固有commitがある。通常マージのプレビューは7ファイルで
競合した。Responses API、runconfig、並列要求、VRAM制御、UI、setup、C++内のCLI/protocol等は
機能単位の適合と検証が必要。主の速い経路を残し、mainの機能を失わせる単純置換は行わない。
正確なCLI文字列差は [cli-option-inventory.json](cli-option-inventory.json)。文字列の存在だけで
意味の同等性を判断しない。R447の機能・依存・テスト対応表とR448の推論候補点検を
[research-index.json](research-index.json) から参照できる。

## 修正した運用

プロジェクト専用スキルを実際の `feature/sycl` 内の
[SKILL.md](../../../.claude/skills/strata-upstream-sync/SKILL.md) に置いた。
`.agents/skills` は `../.claude/skills` への相対リンクで、別worktreeの絶対パスに依存しない。
ユーザー共通のskillsディレクトリにはインストールしない。
Nikoの非推論機能は全件対象、推論は部品ごと・prefill/decodeごとに選ぶ。Xeは参考先と明記した。

主作業は `feature/sycl`。必要な子ブランチはその現在のcommitから作る。
完了した実体を主に統合し、区切りで自身のforkのmainをPR経由で更新、終了した枝を退役する。
原commitを復元可能にするための履歴保存mergeは、推論コードや機能の統合ではない。
wrong-baseの検証案は未採用と明記して保存し、使用を終えた枝を整理する。

この点検では新しいGPU実行や本家のremote更新確認をしていない。
まず保存・整理とmainとの機能適合を済ませ、その後、本家の更新確認と評価をこのスキルで続ける。
既知のGPU障害版は対処前に実行しない。
主に既存のchecked waitとerror伝達も確認した。成功時と失敗・終了時の所有権を区別した
[runtime契約点検](runtime-contract-notes.md) に、次の実行前に閉じる具体的な問いを記録した。

## 証拠と制限

R441～R448の元レポートは本文・hash・由来を保って格納した。調査の提案はMainの採否と区別する。
R448が参照した古いREADMEではquiet測定が未完だったが、その後の完了値を本報告に追加した。
R448の一部prefillパス表記は実際の `src/prefill/sycl/` パスへ訂正した。
R446の読み取り専用計画にある確認案は、ユーザーの既存承認を取り消すものではない。

CPUサービス/route-unionのhost fixture、ASan/UBSan、referenceのビルド結果は
[receipts](receipts) に必要なコマンド・環境・source・終了証拠を要約した。
それらは主ブランチの検証、モデル比較、default-OFFの負荷、物理256Kを証明しない。
Xe実験側への旧metadata公開試行の失敗も元statusのまま残した。
今回の生成手順は [controller.json](controller.json)。
