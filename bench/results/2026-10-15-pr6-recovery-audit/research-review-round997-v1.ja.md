# Round 997 review
- R996のSYCL順序契約と参照コミットを確認。`start >= submit` と `end >= start` は維持し、同値は厳密な計測ゲートで引き続き不採用。
- L0 0.95のコピー制限は版依存。1.4.8は非カーネル操作も許すため、実機ランタイムの版を特定するまで原因候補に留める。
- イベント再利用修正は旧Level Zeroアダプター側の差分。v2実装への適用や今回の原因を示さない。
- v2ソースの10件のSHA256 pinは一致。ビルド、OFF/ON、事前drain例外fixtureはPASS。受理条件は変更なし。
- 例外fixtureはstderr内のdrain開始・終了・Trace出力順とprofiling queryゼロをrunnerで検証。保存物は判定値とstderr hash。
- v1モデル結果はFAILEDのまま: 508/532、拒否24件のraw値なし。通常コピー負荷試験は原因を特定しない。
- v2 39K結果は本レビューでは未評価。終了・閉鎖 receipt後にmaskと全raw値を確認する。
- 部分窓からGPU busy、CPUボトルネック、速度の主張をしない。
