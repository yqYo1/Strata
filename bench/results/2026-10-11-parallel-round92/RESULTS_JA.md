# 本家の設定探索と末尾境界 — 並行調査92

[本家の設定調整TODO](../2026-10-11-upstream-parameter-audit/TODO.md)の履歴確認を完了した。R361は過去索引v113の465件のコンパクトレポートと直近関連調査を対象にした。その保持範囲では本家pristine単体の候補探索・選択基準・選択後の独立検証を確認できない。v0.1.40.2/v0.1.41の過去測定には設定を明示しているが、一つのレシピの反復である。別のtuned比較はcontrol/integratedであり、本家の最適化済み設定の証拠として扱わない。本家単体のprefill/decode個別の設定探索、32K以上・decode3回以上の独立検証は未完了として残した。本家v0.1.42の新規性能測定はない。

R359はXe e32修正版MTP v4のchecked queue completion、upload source所有権、prompt/decode/restore callerの寿命を静的に確認した。R360は別の境界問題を見つけた。通常MTPのp+a+jが論理max_contextを越え得て、ringの余剰resident slotはpage table/RoPEの論理範囲を増やさない。通常generate/serveの末尾windowと残り出力数を制限するソース修正をSol v5へ依頼した。実行・テストはメインが管理する。元のexact262144 full-context probeは別の診断であり、通常生成のこの問題を検証したものではない。

この3報告の返答前には介入していない。R361/R362開始時に過去レポートの実在する絶対パスを共有した。R362は最新本家の同種の境界を別スコープで調査中。Solはソースのみ実装し、メインは同時にソース製LLVMのビルドを継続している。current sourceは編集中なので、この境界で新しいoverlay patchは作らない。

[私設依存の記録](../2026-10-11-xe-llvm-private-dependencies/RESULTS_JA.md)にはcompiler v3/v4のclosed失敗を元のステータスのまま残した。private hwlocのheaderとlibrary参照を補ったv5は進行中。active receipt/logをコピー・削除しない。free/contrib-llvm full build、host ASan/UBSan、実helperテスト、新runtime identity/capabilityはまだ未検証。GPUはこの記録のために実行していない。

[report-registry-v115.json](report-registry-v115.json)で新しい3件を登録し、過去473件はSHA付きv114参照で辿る。削除は0件。
