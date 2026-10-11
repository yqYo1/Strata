# dense GEMM比較のビルドと追加の寿命修正

修正Xe e32のdense GEMM全数比較テストをmainが適用してビルドした。BF16/FP16、独立double oracle、beta、変更入力、全出力とguardsを対象とし、GPU実行はまだ行っていない。モデル精度の検証ではない。

MTP内部captureの行数guard、max-new/max-contextのchecked整数変換、CPUテストの副作用分離を適用した。通常free v6/v8、contrib-llvm v7はWERRORビルドとCPU 10 PASS・ISA不足1 SKIP、不正引数3件のGPU初期化前拒否を確認した。free v8のバイナリーはv6と同じdbaaf5973a8559c6fb333440e6091cca188454de200ba727fad2269cd7a7619f。contrib-llvm v7はoneMath初回試行の寿命修正も含み、507753532ea49c956e47511d9299fce450fff50810b18a90dc1dce78f1dcdffb。full物理262144セル、private MTP、モデル数値・速度は未検証。

必須lint v1の失敗は既存archiveに保存済み。比較v2は元ソースの参照に現行includeを混ぜてしまい、mainが自分のcontrollerをPID/start timeで確認して中断した。子processはreaped、session emptyでsurvivorなし。失敗を維持し、既存指摘の証明には用いない。v3はbare include directoryも正しく置換し、pristine MTP/prefillの構文検査を通した状態で比較した。結果は下記のとおりFAILEDで閉じ、active receiptはコピーしていない。

parity helperの転送失敗時にcaller stagingをunwindさせないpatchはv3が閉じてから適用・ビルドした。最新lintは未実行。最新sourceのASan/UBSanとguard付き小整数GPU確認も準備段階で、sourceの資格と実行結果を混同しない。

原本receipt、正確なsource overlay・untracked file参照、レシピ、失敗比較、staged patchは[retention-review.json](retention-review.json)にhashを記録した。全成功compilerログやobject、GPU traceはGitにコピーしていない。この境界で削除は行っていない。

比較v3はpristine実行132診断、現行91診断を記録して閉じたが、文字列一致で未解決3件がありFAILEDのまま保存した。Verifierの余分なpaddingは70−6=64から63−7=56 bytesに減り、Optionsは140−4=136と143−7=136で同じ。必要なfield名と最小padding量が変わったため警告本文が変わったもので、既存の同じclass/checkとして定量比較する。原本の診断やcheckを抑制しない。host handoff gateで不要となったhelpers変数は条件式へ戻し、denseテストの乗算を明示的にint64へ広げ、ビルド種別文字列をconst char*として型付けした。

parity helperも適用したfree/contrib-llvm v9は、WERROR・CPU 10 PASS/1 ISA SKIP・不正CLI3件を通過。各engine hashはfree v8/contrib-llvm v7と同一だった。GPUプログラムは両モードでビルドしただけで実行していない。最新sourceは[source-manifest-v9.json](source-manifest-v9.json)、v8からの正確な増分は[post-v8-source.patch](post-v8-source.patch)。後続lint v4と最新host ASan/UBSan v6は未実行のレシピであり、資格完了や採用・新しい性能を意味しない。
