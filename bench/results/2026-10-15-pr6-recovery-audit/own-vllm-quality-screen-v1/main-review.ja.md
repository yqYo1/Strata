# R837: Main向け4 arm品質screenソース確認メモ

状態は **UNRUN**。本workerは指定4ファイルのみを追加した。ビルド、テスト、syntax/source checker、対象Pythonのimport、GPU、profiler、library loading、process inspection、Git、install、cleanupは実施していない。Mainの全ソースレビューと実行はこれからであり、このメモ自体はレビュー承認でも測定結果でもない。

対象は統合ソース `6354f0b4ff3731466795410e11ca9ef785c9d326`、build receipt `own-vllm-engine-build-v2.json` SHA256 `990635731be1198706abeb682817fdab247f8f51d31bab971d3d4f4db53cb639`、binary SHA256 `85643f3926bb9d212945659cb7ddb264e101ca9b2d5fa8f19430e945140ad191`。synthetic public-hook model3armの既存PASS receipt `own-vllm-model-smoke-v1.json` SHA256 `03b2bd388a897873bb2dc8c310ec7c0386e6b484828d1e633fb872cd059a14d3` を固定する。これは既存receiptの事実の参照であり、本パッケージのPASSを意味しない。既存FAIL/PASS receiptの内容・pinは変更していない。

`own-native16-quality-screen-v1/run_screen.py` の所有・source freeze・observer契約を現buildに適応した。旧比較器は3arm名、旧build chain、旧contextに固定されているため、4arm用 `compare_screen.py` を追加した。ただし旧比較器の `open_capture`、`read_row`、`pair_metrics`、`exp_or_overflow` をSHA固定importしてそのまま使う。FP64 max-shift、`math.fsum`、top1 lowest-ID/ties、KL/L1、桁落ち回避式を変更していない。denseと独立FP64 NLLの許容値は統合model smokeの `1e-10` を維持する。品質採用の新しい閾値は設けない。旧hostfixtureもコピー・変更・実行しておらず、必要ならMainだけが既存のものを実行する。

| 実プロセス順 | native16 | R dtype | backend | tile | QSA |
| --- | --- | --- | --- | --- | --- |
| fp32-a | 0 | fp32 | own | auto | 0 |
| fp32-b | 0 | fp32 | own | auto | 0 |
| bf16-vllm | bf16 | bf16 | vllm | auto | 0 |
| bf16-vllm-qsa | bf16 | bf16 | vllm | auto | 1 |

全armは同じbinary/引数で、context=resident=65536、chunk8192、R_GPU32768、ROPE_TABLE=1、max-new1、単一in-order engine stream、token graph無効。既存のthinking ON自然文32863 IDをそのままコピーする。retokenizationやpaddingは行わない。位置は255+512iの64行、語彙248320。各armに外部expert profileの同じ初期SHAを私有コピーへresetし、外部は照合だけにする。native processへの環境は明示mapでexec前に確定し、UR v2/copy-offload-disable1/DirectSubmission0/parameter validationを継承する。親の `os.environ`、runtime/provider、engineやmathは変更しない。

FP32-bを終えた時点でselected全語彙payloadとdense NLL payloadのSHAをFP32-aと比較する。一致しなければ候補を起動せずscreen失敗を保存する。生成1tokenのID/countは正常完了証拠として保存するが、生成IDや文字列の一致を要求しない。public Gemm backend/tileは既存stderr route stampを確認する。QSAには追加stampを導入せず、固定envとbuildで固定された既存dispatch sourceによる選択として記録する。

selected observerのnative形式は既存 `strata-prefill-logits-v1` のまま。独立request JSONにinput SHA、64位置・target、head/type、出力path、shapeを保存し、そのartifact SHAをrecipeへ結ぶ。owned commandの実argv/env、recipe、request、raw/meta SHA、厳密なheader/footerとtarget順を照合する。native selected metadata自体にrecipe digestがあるとは主張しない。dense observerは `STNLLV2` のheader64・全32862 target/loss・reserved bytes・native head・complete metadataを検証し、`STRATA_PREFILL_NLL_RECIPE_SHA256` でnative出力とrecipeを結ぶ。recipeは自己digest環境項目だけを除外して保存し、exec前にその項目を加える。

完了時には全48layer/5chunk、入力ID、実配置、正常終了、PID/start/reap/empty session、boot、各arm前後と最終のfresh journal窓、device dump不在が必要。faultは窓を進めても累積し、後の空窓で消さない。buildの全432 source pinと全314 file pin（旧source/recipe、実compiler、libraryを含む）、model smoke由来runtime/pack/MTP、model shardのsize+mtime、input/preparer/旧oracle、パッケージ自身を実行前後で固定する。shardのfull SHAを取得したとは記録しない。古い別queue regression FAILは不変・不適格のまま。明示priorはMainが選ぶ最新の同一boot、正常終了したPASS GPU cohortであり、失敗receiptを続行用gateへ読み替える経路はない。

比較はFP32-a対FP32-b、および各FP32対両候補の5ペア。全64位置のtarget、全語彙finite、独立FP64 NLL/top1/ties/row SHA、logit差、KL/L1、sampled PPL比と個別差を保存する。dense全位置のNLL配列と個別差もcompact receiptへ残し、独立oracleは64位置で照合する。dense NLLの全位置が独立oracleで再検証されたとは主張しない。

新規private rootは `/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-vllm-quality-v1` のみ。存在すれば拒否し、resume/overwrite/cleanupしない。`owned-v0141-measurement.lock` を全体で排他取得する。900秒/arm、work4140秒+閉鎖用60秒で全体4200秒、RSS88GiB、text64MiB/commandに加えて既存Ownerの全text64MiB制限、全raw captureは384MiB未満、private diskは1GiB以下。4つのselected payload計254279712 bytes、dense計1577632 bytes、text上限67108864 bytesで、計322966208 bytesと小さなmetadataを想定する。budget/timeoutは未完了・失敗として保存する。所有終了と証拠の最終書込みは必ず残す。

Mainはactive GPU jobの正常閉鎖後、最新PASS priorの実path/SHA、48時間以内の明示UTC review期限を下記3変数へ設定する。active private root/processは本workerから調べていない。次がrepository rootからの実CLIであり、固定値を差し替えて古い失敗結果を適格化する用途には使わない。

```sh
: "${screen_prior_receipt:?Main must set latest normally closed PASS receipt path}"
: "${screen_prior_sha256:?Main must set its exact SHA256}"
: "${screen_review_utc:?Main must set UTC ISO review deadline within48h}"
python3 bench/results/2026-10-15-pr6-recovery-audit/own-vllm-quality-screen-v1/run_screen.py \
  --build-receipt bench/results/2026-10-15-pr6-recovery-audit/own-vllm-engine-build-v2.json \
  --build-sha256 990635731be1198706abeb682817fdab247f8f51d31bab971d3d4f4db53cb639 \
  --prior-receipt "$screen_prior_receipt" \
  --prior-sha256 "$screen_prior_sha256" \
  --input-sha256 c0e89883e69aad815adc165ad199091c990dbcb8ce47275e155a2793056af0b1 \
  --retention-review-utc "$screen_review_utc"
```

Mainが実行した時だけ `A/own-vllm-quality-screen-v1.json` を新規保存する。`passed`/`correctness_capture_passed` が示すのは閉じた機械的correctness/capture比較であり、`model_quality_accepted`、`quality_accepted`、`model_quality_qualified`、`performance_qualified` は常にfalse。単一corpus/相関した64位置は実用精度の判断材料であり、一般/task能力・統計的保証・長生成・物理262144/RESTOREの採用判定は別途Mainが行う。denseを含む診断実行から速度を採用しない。rawの所有者/consumerはMainと本比較器、reviewは比較閉鎖・supersessionまたは明示期限。旧canonical rawはそのままにし、cleanupは本パッケージには含めない。

由来は既存Strataの上記2ソース、native selected/dense writer schema、統合model controllerとreceipt。外部engine source/helpers/tests、XeStrata実装は参照・採用していない。receipt内の既存LLVM toolchain pathは公式compiler/runtimeのidentityとしてだけ継承する。新規ファイルはrepository LICENSEに従う。ソースpathとSHAは隣のmanifestに固定し、manifest自身のSHAはOrca handoffへ記録する。
