<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# 上流更新スキルと性能履歴の確認

strata-upstream-sync を作成し、個人の Codex skills にインストールしました。
本家・XeStrata の未変更単体を箱出しと環境に適した設定で測定し、非推論機能を全件
対応表で取り込みます。推論は役割・量子化・shape・個別部品で判断し、fork が優れる
部品を保持します。prefill の利益で decode の退行を隠しません。上流 bench.sh を使い、
既存の最良設定と呼び出し経路、個別結果、反復調査、256KとRESTOREの範囲を記録します。
validator と文書・秘密情報・ライセンス検査を行い、synthetic fixture の独立評価を
2回受けました。これは実上流の測定や統合ではありません。ユーザーの必須機能全件と
複数エージェント反復指示を optional に変える提案は採用せず、scope、実行不可の上流、
bounded search と最大入力の実消費証拠に関する明確化を反映しました。

この session の履歴を確認し、PR #6 の paired-normal-wall を Git から読み直しました。
32768入力、8192 chunk、context65538、3組交互比較で中央値453.2935から645.848 t/s。
全248320 logits と出力の一致を伴う有効な過去記録です。現在の標準benchとは入力、
時計、context/KV、出力長等が異なるため退行率を単純計算しませんが、古い値を棄却しません。
PR #8 の約520/533/540は64Kの単発・計測付き観測として保持します。

旧 layer-major は層ごとに全入力chunkを処理し、expertの再転送を減らします。
現在のroot srcにはその経路がありません。旧実装はsycl/配下の別topic commitで、
c3aa33aeのsource系統選択で使われるroot engineに引き継がれませんでした。
後のv0.1.43サーバー移植でこの部品を削除したという証拠ではありません。
R394/R395初版は過去の調査範囲が不足していました。元を保持し、v2で訂正しました。
短いcontextのIQ3_S decode18.30と現在の長文・MTPなし8前後も条件が違い、
decodeの直接の退行は未確定です。新しい移植候補と以前の最良構成を再比較します。

12K候補の繰り返しでは1024、32768、32768入力を完了しました。その後の65536要求は
要求4383187712 bytesに対し空き4355325952 bytesで、workspace確保前に拒否しました。
元結果はpassed=falseを維持し、観測はcompleteです。後続GPUと全モデルhash確認は
通り、faultはゼロです。64K完走、品質、全物理262144、速度改善の証明にしません。
閉じた重複ログだけをmanifestに沿って削除し、元statusを変更せずcompact evidenceを
commitしました。registry v193は948件です。目標1000/70 t/sは引き続き未達です。
