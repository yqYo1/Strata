# Native dequantのprefill一括処理の資格確認

Arc B570（10GiB）、Ryzen 5 5600X（AVX2、6 core/12 thread）、Linux 7.0.0-38-genericで、Mainがビルド・検証・GPU実行を共有lockの下で直列に管理した。検証済みのprefill callerを、既定OFFの実験版として `fa3e20e842fbed6986a69bade8afe3edabedee07` にcommitし、自分のforkの作業branchへpushした。モデルの正しさ・速度による採用は未完了。リセット・サービス操作・自動retryは行っていない。

## コードと実際の検証

`STRATA_PREFILL_NATIVE_DEQUANT_BATCH` を明示して有効にすると、native expertのGU/Down逆量子化を最大16件ずつ既存APIに渡す。形式・領域の条件を満たさない場合は各expertの従来wrapperへ戻す。正常に受理されたsubmitの例外をrefusalとして扱わず、GUとDownの両方の読取り完了を確認してから保持したring slotを返す。lookaheadの再利用衝突、group末尾、層末尾などでflushするため、実際のwindowが常に16件になるとは主張しない。

既定OFFの従来grouped処理20,088byteは、旧commitの文字列と完全一致し、SHA-256は `9d6c91d570e1df2d14048995feec108280899e27d423f478f7d057f4b8104d75`。`source/default-path-proof.json` に今回の再確認を保存した。

最初のv19 freeビルドはPASSだったが、未定義の `STRATA_ENABLE_SYCL` によってON処理がコンパイル時に消えていた。これはON処理の資格にならない。Mainがこのguardを除き、実際にコンパイルした。次の必須lint v15は、ON処理のexpert終端をint64からint32へ渡す新しいnarrowing診断でFAIL。正のexpert数とINT32_MAX以下を確認してからint32へ変換する修正を入れた。元のPASS/FAILは変更していない。

修正後のrelease v21、host ASan/UBSan v18、実際の `tools/lint/run.sh` v16はfree・contrib-llvmの両モードでPASS。各release/sanitizerの既存CPUテストは12PASS、AVX512非対応による1skip。変更したprefill.cppのlintは各モードで既存10診断、新規0。deviceコードはhost sanitizerの対象外。`nm` により、両モードのprefill archiveからGU/Down group APIの2つの実参照を確認した。ソースに存在することだけで実行を証明していない。

group censusの文法はdevice・層範囲・chunk offset・位置・token数・層・fused/CPU/選択数を含む。Mainは正常3ケースと破損47ケース、計50ケースを実行しPASS。model controllerのCPU確認は正常9・破損35ケースと、4つの不正CLI入力の期待したexit 2を確認した。これはログ・protocol・入力拒否の確認であり、モデル数値や全256Kの検証ではない。実装agentが記載した27件の拒否条件は一覧であり、27件の故障試験を実行したとは扱わない。

## 最初のモデル診断で判明した起動時の問題

64 tokenのOFF診断を実行した。これは起動・処理順の初回確認で、速度比較ではない。物理context 262144、chunk 6144、group 16、要求cache 128から実効144 slot、CPU worker 5、pcie fraction 0、MTP 4、INT8 KV/常駐32768という設定でINFOとREADYへ到達した。しかしGEN送信前に、詳細UR/Level Zeroログのエラー判定で停止した。

検出したのは3つのdevice module build failureで、同じlocal joint-matrix load builtinのundefined referenceを含む。BF16/FP16の `Kernel<2,...>` と3形式の `PromptAttnMma` がbuild logに現れた。Mainはエラー判定を緩めていない。controllerのSIGTERM cleanupの後にlongjmp警告2,241行とexit -11を記録した。childはreapされ、sessionは空。GEN、native batch ON、モデルの数値比較は実行していない。この時系列だけでlongjmpの内部原因やGPUハングとは断定しない。

前後のkernel journalに新しいGPU障害はなく、停止後に既存の小さなhealthを新しいownerで1回実行しPASS。64KiB H2D、16,384語の整数演算、64KiB D2Hが完全一致し、同じ8つのruntime library hashとfresh journalも一致した。これは整数経路の健康確認であり、失敗した行列kernelの再試行・正しさの確認ではない。

ソースでは `Runtime::preload_kernels()` がjoinableな背景threadで全 `get_kernel_ids()` にexecutable bundleを要求し、compile例外をskippedとして握り潰していた。モデルで選ばれないspecializationも事前コンパイルされ得る。ただし今回の各specializationが実際のGENで不要かは未確定。source/runtime/binaryの詳細はreceiptsとenvironment.json、代表ログはstartup-error-excerpt.txtに保存した。失敗runのbinary hashはv21と一致し、両者の37 source pinsはpublished commitとも一致する。source_uncommittedという元の記録は変更せず、hashに基づく対応を別途確認した。

実装agentは `STRATA_PRELOAD_KERNELS=0` で全IDの背景compileを開始しない5行の診断候補を作成した。未適用・未build・未実行。実際に使用するkernelのcompileやerror handlingは変更せず、初回JITの時間が変わるため速度測定とは分ける。これを資格確認してから、未使用の事前compileと実使用経路を切り分ける。必要なkernelが失敗するなら、そのエラーを保持して修正する。

## 調査・採否・次の比較

Lunaの読み取り専用調査を、compiled caller、memory admission、driver lowering、service時間、stream選択、AVX2 decode、起動順、kernel到達条件などに分けて反復した。Mainが全報告を読んだ。直近のprovenance誤記は元報を編集せず別addendumで訂正した。registry v130は従来508件に19の固有報告を追加した527件。過去報告が返っただけのdecode再割当は、新規調査完了として数えない。実装は別のgpt-6.1-sol、全検証はMainが担当した。以前のLuna quota制約は、今回の実際の起動・完了によって解消している。

32Kの旧censusでは、unused full-stream copyは全H2Dの5.566%。route選択で省ける量よりも、メモリに入る範囲でchunk数と全expertの読み直し回数を減らす方が、削減できるbyteの計算上の余地は大きい。この候補は未測定で、256K全長の容量・寿命を満たして採否を決める。API単体の改善をモデル全体の改善とは呼ばない。

新しい32K/64Kのprefill・decode速度は未測定。旧32K診断のprefill 269.159 token/s、cold decode 8.271 token/sは現在のcallerの性能ではない。次は起動時JITの切り分け、短い正しさ確認、2049 tokenによるfull-stream境界確認、その後32K以上の同じ入力でOFF/ONを比較する。quiet速度を別processで取り、decodeは複数回測る。prefillとdecodeの採否は別に決める。full 262144位置、RESTORE、全モデル品質は引き続き未完了。

本家測定の設定については[既存TODO](../2026-10-11-upstream-parameter-audit/TODO.md)を継続する。過去本家は設定を明示しており箱出しだけではないが、本家単体の最良設定探索と選択後の独立検証は確認できていない。本家を独立に調整する前に「最適化済み本家との差」と主張しない。

## 保存と削除

個別PASS/FAIL、command、allowlist環境、source/binary/runtime hash、process終了、fault判定、検証範囲を18の元receiptそのままで保持した。成功した全API/poll/build/lint出力300ファイル、論理4,459,848byteは、Git保存とhash確認後に削除する。失敗の代表窓を抽出した後の重複出力も含め、削除予定は計336ファイル・論理35,324,933byte。retention-v1.jsonがpath/hash/reasonを記録する。失敗したmodelの27.9MB traceも、該当調査が戻り代表窓で十分と確認したため削除する。現在の調査consumerは30,751byteの代表窓と静的source/metadataを参照する。元の測定status、現在のfixture、model/source、GPU/service状態はcleanupで変更しない。

IGC公式sourceのGetMatrixFuncNameはrow-major AをPackedAという内部名へ変換し、SG16という文字列をplatformのDPAS能力で付ける。sourceのsubgroup 32とは矛盾せず、Kernel<2>のlocal A loadと強く対応する。ただし実際の失敗moduleのIRとproviderはまだ取得していない。[IGC v2.41.5 source](https://github.com/intel/intel-graphics-compiler/blob/v2.41.5/IGC/Compiler/Optimizer/OpenCLPasses/JointMatrixFuncsResolutionPass/JointMatrixFuncsResolutionPass.cpp) と最新レポートに根拠を保存した。host上のIGC候補2.40.13と過去inventoryの2.41.5は区別し、今回の実mapにIGC/FCLを含めていなかった点を未確定として調査している。

公開前のgitleaksはexit 1で2件を検出した。1件は既知のbyte試験receiptのSHA-256、1件は公開commit IDであり、Mainが元のreceipt/Git objectと一致することを確認した。元レポートはbyteそのまま保持した。scannerのstatusはPASSに変更せず、suppressionも追加していない。詳細はsecret-scan-review.json。

最新2件の調査も戻った。IGC v2.41.5/v2.40.13のsourceには該当builtinを生成する定義があり、単純なsource上の欠落とは扱わない。実際にどのIGC/FCLがロードされたかは既存filterでは収集しておらず、次のMainの有限な実行でpath/hash/build-idを追加収集する。2件のagentは完了済み。次のprovider判別調査は、この実mapと選択した実演算の証拠に依存する。

初回のreport公開は、3つのunified diffの空行context（1つのspace）がGitのtrailing-whitespace判定となり、commit前にFAILした。元patchの意味・byteを変えず、その3件だけlossless base64 JSONに保存し、decodeしたbytes/hashの一致を確認した。元receiptとこの公開失敗も保存し、whitespace checkのsuppressionは追加していない。
