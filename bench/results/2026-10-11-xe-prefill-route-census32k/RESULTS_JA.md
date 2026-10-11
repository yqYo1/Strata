32K・6144幅の診断実測で、処理分岐とexpert数を全48layer・6chunk・288行で確認した。計算・転送・同期処理には変更を加えず、消費側のhost集計だけを追加した。通常ビルドとhost ASan／UBSanをfree/contrib-llvmの両方で通し、実際の必須lintを変更前と比較した。初回lintの整数cast指摘1件は診断出力の加算前castに修正した。元のv10 FAILEDは保持し、修正後v11は両方追加指摘ゼロ。

診断付きprefillは269.16 tok/s、初回16token decodeは8.27 tok/s。採用判断用の静かな反復測定ではない。MMQ/FP16等の実分岐と全counterは[summary.json](summary.json)。成功wrapperの回数と転送要求extentであり、exclusive GPU serviceや物理PCIe帯域の測定ではない。

全288行でMMQ=0、fallback=1で、GU/Down展開の成功wrapper呼び出しは各136,846回だった。グループgather呼び出しは0回。未使用expertへのfull-stream転送要求は8,065件・16,473,728,000 byte（全expert転送要求量の5.57%）である。まず実際に使われたFP16展開をまとめる候補をbyte単位で検証し、component serviceを測る。MMQのgather最適化はこの設定の実測経路には寄与しない。

16token ID・selected logprob・top5は従来の同入力基準と完全一致した。normal QUIT、実runtimeの照合、前後のfresh GPU health、新faultなしを確認した。256Kは設定値であり、実際の全262144tokenやRESTORE、広い能力評価は未完了。

別の実装エージェントから、最大16expertのGU/Down展開APIと、32K/64Kなどの可変chunk protocolをファイルで受領し全文・hashを確認した。前者は未適用・未build・未GPU検証である。後者はMainのCPU12正常／97拒否とtrace12正常／8拒否を通り、閉じた実6144入力captureも検証した。zero-output fixtureは明示された仮の契約であり、実256Kで0出力になることを証明しない。

本家単体の設定最適化は[追跡TODO](../2026-10-11-upstream-parameter-audit/TODO.md)で未完了を維持する。今後のprefill/decode採用は個別に決め、decodeは独立反復でばらつきを確認する。

保存前のgitleaks検査は、境界controllerの公開ケース識別子をgeneric-api-keyとして誤検出し、commit前に終了した。元のFAILEDと検査コマンドを保持した。Python変数名だけを変更した保存用recipeから元の全文・hashを復元できることを確認した。検査の抑制や例外追加は行わず、再度同じ検査を通す。測定statusは変更していない。
