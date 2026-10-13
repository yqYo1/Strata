<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# 使用済みブランチの履歴保存と整理

各枝の元コミットと未採用source snapshotを、作業ブランチのancestorとして保存する。
履歴だけのmerge前後でtreeが一致し、推論コードの採用ではない。
元のコミットIDとrepo相対pathからgit showでレポート・sourceを復元できる。
manifestに削除対象、残す用途、snapshotの未検証状態、必要な依存ファイルを記録。
LLVM/runtime、共有lint、比較baseline、次の262144/union/decode/adapter作業は用途を持つため保持。
profileとoneMath archiveは必要な小ファイルだけを外部へ保持し、既存の参照pathをaliasで維持。
公式bulk測定は32049入力・24出力・再利用0で完了し、正常QUIT・無障害。
prefill693.5/691.1/701.1、decode10.1/7.9/8.1 token/s。
decode低下を棚上げせず、非bulk baselineとの再比較・部品時間の計測が必要。正式採用は保留。
