# 計算資源ごとの測定結果と未確定の律速箇所

機能時間に加え、転送量・資源容量・CPU待機・GPUコマンドの重なりを対応させる。
今回の追加測定による最適化採用はない。全個別値と入力・実行条件は同名JSONから参照できる。

| 資源 | 測定で分かったこと | まだ言えないこと |
| --- | --- | --- |
| CPU/RAM | 実expertサイズ7種類、512copy/層の通常RAM proxy。3独立process、各cell9標本。48層50.293GBへの換算は1core4.612秒、3core3.903秒、6core4.084秒。以前の実source-copy4.852秒との差は3core proxyで約0.95秒。 | pinned/mappedの実入力や物理RAM trafficの飽和。copyのthread追加だけで600/1000tok/sへ届くとは言えない。 |
| CPU制御 | 既存の長入力診断では発行thread98.420CPU秒のうち69.696秒、約70.8%が明示GPU待ち。grouping0.106CPU秒、GEMM投入12万回超で5.643CPU秒。 | 1core100%をCPU演算の飽和とは呼べない。未分類部分、発行依存、ライブラリ内部処理の寄与は別途必要。 |
| PCIeを含むhost/device経路 | 新しい320MiB proxyはH2D約4.35GB/s、D2H約3.89GB/s。各方向3process×3標本。以前の同binaryの約6.02/5.54GB/sより低い。監視の有無も比較した。 | 要求payloadはwire bytesではない。実モデルの転送serviceや物理PCIe上限にこのproxy値をそのまま置き換えない。差の原因も未確定。 |
| GPU内部 | 既存の長入力Attention観測区間はcompute占有77.31%、copy9.11%、重なり約8μs。QSA60回のservice計33.963秒。まずGPU内部と発行/同期を調べる根拠になる。 | computeという分類だけでALU/XMX飽和とは言えない。VRAM/cache、SLM、同期、依存のどこが律速かは未確定。 |

以前のGPU占有率の分母は約94.945秒のAttention envelopeで、全prefillではない。
全prefillの計装wallは105.007秒であり、別のclean速度403台tok/sとは混ぜない。
GPU内部の物理traffic、演算活動、stall/occupancyは今回も未取得。
古いVRAM copyや大きなGEMMの値は、操作・形状・stackを明示した参照値に留める。

## 同じ転送binaryの再確認

Mainが現在のsource全体を再読し、既に104copy/checkで検証済みのcopy専用binary
SHA256 `2fccadda6686349ad4dbc081511d99f4f6cb85683055797cc21bb28e15f9894c`
を変更せず使用した。モデル、演算kernel、queryはない。既存runtimeファイルhash、
GPU selector、copy-offload無効、DirectSubmission無効、同queueを保持した。
環境は新しい明示的な最小集合で、旧cohortに含まれた未使用のtooling/engine変数を継承していない。
したがって以前のcohortとの完全同条件比較や原因特定を主張しない。

各cohortは4MiBのUR tracing/validation付き初回確認後、320MiB通常RAMの3独立process。
各processは1warmup＋3回測定、双方向とも全pattern・両端guard・source不変性を確認した。
D2H前にはhostをpoisonし、未転送でも古い値で通ることを防いでいる。

| 条件 | H2D中央値 GB/s | D2H中央値 GB/s |
| --- | ---: | ---: |
| sysfsを20ms間隔で外部監視 | 4.307978 | 3.926239 |
| 監視threadなしの対照 | 4.347490 | 3.892074 |

中央値はprocess内の中央値をさらに3processで中央値化したもの。9個別値を全件保存した。
監視を外しても差は残った。CPU/GPU clock、staging、queue、環境差等の原因は未確定で、
この結果をコードのデグレや新しいPCIe律速と決めつけない。全8processはexit0、所有終了、
同boot、新規fault0、dumpなし。モデル速度の測定・採用ではない。

同時に読んだsysfsはcurrent/maxとも2.5GT/s・x1と表示した。
この表示値を物理上限と解釈すると実測payload速度に整合しない。
表示と実測の両方を保存し、表示だけでバスの能力を決めない。
次のモデル比較でも、古い容量値と現在の環境差をコードの効果へ混ぜない。

## GPUカウンター経路の実結果

MainがAPI専用helperを別途build/reviewした。モデル、queue/command list、query begin、
計算kernelを投入せず、ComputeBasic EVENT/domain3/83列のpool64作成を一回だけ試す。
実行前後に実PID/start・環境・maps・configを照合し、親を解放する前のcleanupも確認した。
元のcollector失敗や数値未完了のstatusは変更していない。

最初の診断ではMetricsLibraryのログにparanoid warning、kernel metric-set取得失敗、
TBS/Kernel初期化失敗が現れた。小さな追加syscall診断ではrenderD128への2要求がEACCES、
固定GUIDのmetric IDが2回ENOENTだった。poolは引き続きINVALID_ARGUMENT/exit125。
deactivate/context destroyはSUCCESS、所有終了、新規fault0、dumpなし。GPUハングではない。
両方のreceiptでcollection/numerical_resultはfalseのまま。

paranoid warningの関数はpinした公式sourceでは常にSuccessを返すため、警告単独は原因の証明に
ならない。straceのI915_CMDBUFFERという表示にも注意が要る。Xeとi915の公式headerには同じ
command番号・同サイズの構造体があり、このdecoder表示だけでGPUやdriverの取り違えとは言えない。
正確なOA sub-operation、権限判定、registration順や版の整合性までは未確定。
具体的な拒否とID欠落は `metric-pool-kernel-denial-v1.json` と代表windowへ残した。

このevent-query経路の調査はここで区切り、追加native retry、権限/sysctl、package、runtime、
driver、serviceの変更は行わない。全てのprofiler/counterが使えないという一般化もしない。
Orca Luna R911の限定調査をnative Codex Luna-max R912がreviewし、本文をMainへ広く読み込ませず
metadataと適用限界を保存した。追加の反復調査は新しい具体的な問いができた時に再開する。

次は再利用・作業分割・発行回数等を一つずつ変え、同じ数値仕事のcomponent比較からGPU内部の
競合仮説を切り分ける。独立したPR#6復元は短入力の受付16hostケースと構成・生成111commandの
確認が通ったが、最初のbuildはCPU cpuidとSYCLの衝突で停止した。GPU実行はまだない。
同じCPU expert TUを、sourceを変更せず `-fno-sycl` の追加だけで別objectへcompileすると
通った。CPU専用targetのbuild設定に限定した修正をSolへ依頼し、Mainが再構成・buildを管理する。
復元版の600再測定、数値/state/品質、物理262144、RESTOREはいずれも未認定。
