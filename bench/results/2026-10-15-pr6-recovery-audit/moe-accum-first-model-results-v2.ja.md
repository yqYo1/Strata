# 大きなexpert-major windowの初回モデル検証

2026-10-17。検証ブランチc05c2545のfull SYCL engine build/linkがPASS。
同じバイナリで、bench.shの32K最小入力（実入力39,262、batched39,261）を
従来window2と新window8で1回ずつ実行し、両方PASSした。
context/KV resident131,072、chunk8,192、BF16 HC/FP16 matrix入力、
row-major BF16 QSA、全residual GPU、thinking ON、生成24 tokens。

| 診断ケース | prompt tok/s | window数 | window内dequant/jobs |
|---|---:|---:|---:|
| 旧window2 | 501.8 | 96 | 27,679 / 28,196 |
| 新window8 | 532.3 | 48 | 17,544 / 18,748 |

TRACE/parameter validation/coarse/CPU profile付きの単回診断であり、clean性能の
優劣判定には使わない。旧counterはwindow外の最終single chunkを含まないので、
上のdequant/jobsはモデル全体の回数比較ではない。
両方とも全48層、完全なSSE、24 thinking tokens、正常終了、owner終了、
新しいxe faultなし、devcoredumpなし。今回の24-token reasoning SHAも一致したが、
短いthinkingのみのprobeから一般品質や長い生成のdecode速度は判定しない。

部品はCPU Release/UBSanと全10 GPUケースで検証済み。
Orca Lunaのsource auditとnative Luna-maxレビューで、具体的な新しい境界・
処理順・旧mode0/2の不具合は指摘されなかった。レビューのsnapshotに含め忘れた
DraftUploadGuardはMainが実buildのsource pinと照合し、定義とownerの破棄順を確認した。
これはGPU障害を注入した失敗処理テストではない。

最初に検査用controllerへ8Kを指定したが、bench.shのprefill-onlyは32K/64K限定で
引数エラーとなった。GPU初期化後、prefill前の失敗で、engine/serverは正常終了。
元のfailed receiptと呼出しを保持し、別v2 controllerで32Kへ修正した。

新経路は最大65,536行。共有gateを全window行分保持し、FP16 mixedを別allocationに
保存する。512MiBを残す予算にmixedが収まらなければ旧window2を選ぶ。
その後のplanner/実allocationエラーをwindow2へ再試行する処理はなく、model前に失敗する。
実際のcodeは検証ブランチにあり、production/performance採用はしていない。

次は同じbinaryで64K最小入力の診断を両modeに行い、65,536行と部分windowを通す。
その後TRACE/UR layersを外して3回ずつ交互測定する。
一般品質、physical262144、RESTORE、600/1000目標、sustainedTG70は未達/未検証のまま。

全個別receiptはmoe-accum-model-profile-window{2,8}-tokens32768-v2.json、
比較はmoe-accum-first-model-pair-v2.json。ビルドはmoe-accum-engine-build-v1.json。
必要な根拠を抽出した後、成功した重複ログはretention manifest付きで削除する。
