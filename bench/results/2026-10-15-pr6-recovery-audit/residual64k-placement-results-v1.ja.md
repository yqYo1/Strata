# 78K入力の全量VRAM配置と次のMoE方式

同じ実入力78,453tokens（batched78,452、再利用0）、context/KVresident131072、chunk8192、BF16hidden state、既存MoEwindow2、row-major QSAで比較した。公式benchのprefill-onlyを使用し、thinking有効・24tokenの生成まで完了した。プロファイラー、STRATA_TRACE、UR検証層を外した新規プロセスを配置毎に3回、従来0→全量0→全量1→従来1→従来2→全量2の順に実行した。

|配置|各PP(tok/s)|中央値|平均|標本標準偏差|
|---|---|---:|---:|---:|
|VRAM32768行、残りRAM|486.5 / 493.3 / 493.5|493.3|491.1|3.98|
|全78452行をVRAM|571.3 / 561.9 / 529.2|561.9|554.13|22.10|

中央値は+13.91%。全量側の最小529.2も従来側最大493.5を7.23%上回った。ただし全量側には42.1tok/sの幅があり、最後の低下の原因は未特定。入力・バイナリ・環境・配置・window/dequant/jobカウンターは各配置の3回で一致した。ばらつきが解消したとは扱わない。最小目標600および目標1000は未達。今後の長prefillチューニングでは全量がadmissionを通る配置を性能比較候補にするが、製品の既定値変更や汎用contextの採用ではない。

全量配置は追加のBF16residualを671,088,640Bから1,606,696,960Bへ増やし、RAM側935,608,320Bをゼロにした。中間47層の往復で要求する転送量は計算上87,947,182,080B減る。同時にGPU内のMoEペア処理対象も96から240windowsへ増えるため、この差だけでPCIe単独のボトルネックを確定しない。カウンターはwindow経路だけで、従来側のsingle-chunk処理を含まず、全dequant量の比較には使えない。

全6回がnativeexit0・完全なSSE・所有プロセス終了・新しいxe障害/ダンプなしを通過した。24tokenのreasoning hashは全回一致したが、回答は空で出力上限に達するprefill専用probeであり、一般的な能力、長い出力、物理262144、RESTORE、持続decodeの検証にはならない。元の失敗したprofiling receiptは変更していない。時刻異常は小規模待機テストで別途再現されており、GPUbusyや絶対的なサービス時間の資格は未取得。

## 次の実装

同一層のexpert再展開と起動をより多くの入力行で共有するexpert-major方式を準備した。既存FP32一時RをBF16mixed入力として借り、共有領域にFP32加算器と固定8192行のjob領域を置く。BF16保存は維持し、shared項はrouted和の後で加える。expert順にFP32fmaするため既存route-slot順とは丸め順が変わり、数値/実タスク比較が必要。

コード化した容量plannerはregion上の16個の非重複view、各256Bの予約間隔、R/embeddingの容量、論理chunkの境界を検査する。GCC13.3の通常-O2/-DNDEBUGとUBSanの両方で9種類の配置（8193〜81920行）・34拒否条件を通過した。最大時1,058,836,480Bに対し、実行ログのregionは1,180,698,624B。物理context全体262144行を単一windowへ誤って通すことは拒否するが、リクエスト全体を拒否する設計ではなく、将来のschedulerがwindow分割と既存hosttailを残す必要がある。

plannerは未接続の部品で、GPUkernel・alias寿命・route契約・モデル精度を検証したことにはならない。まず65536行までの単一R領域で実装し、78Kを65536+12916に分ける候補を優先する。81920の二領域gatherは別の段階で扱う。CUDA/HIPおよびXeStrata実装は使用していない。

R998〜1000、R1002〜1003のOrcaLuna調査を反復し、nativeLuna-maxのR1001/R1004がレビューした。収集R1000/R1003の本文はMain未読。R998のaged合否数の取り違えとR1002の寸法表の誤りはレビューで明記し、原文/receiptは保持した。今後の判定には訂正済みレビューとMainの実データを使う。

個別測定、全環境/入力/ソース/バイナリ/所有とfault結果はresidual64k-placement-clean-cohort-v1.jsonおよび6件のsample、集計はresidual64k-placement-clean-analysis-v1.json。容量テストはmoe-accum-plan-host-v1.json、元の配置計算はmoe-expert-major-arena-census-v1.json。上流は2026-10-16T01:15Zの296b59d2確認記録があり、次の48時間確認は10-18T01:15Z以降。今回は新しい上流確認を行ったとは扱わない。

正常実行の全ログ・重複bench結果・終了済みworkerの起動記録など235ファイル、論理10,309,594Bを要点抽出後に削除した。未解決の時刻問題は1件の代表UR/stdout計3,897,369Bだけを4MiB予算で保持する。削除内容・ハッシュ・置換先はclosed-placement-diagnostics-retention-v1.json。ソース、モデル、バイナリ、最後のfixture入力、元の結果判定は変更していない。
