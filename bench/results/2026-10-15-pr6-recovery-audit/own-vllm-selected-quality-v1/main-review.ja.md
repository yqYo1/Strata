# R852 selected-only controller: Main review

状態は **UNRUN**。追加はこのディレクトリの `run_screen.py`、`compare_screen.py`、`manifest.json`、本メモだけ。エンジン、旧controller、oracle、fixture、receiptは未変更。workerの確認は資料読取り、ソース目視、メモリ上の `compile()` のみで、対象moduleのimport、controller起動、テスト、ビルド、GPU、model/library loading、process inspection、Git、service/reset/cleanup、subagentは実施していない。Mainのレビュー・qualification・実行は未実施。

旧 `own-vllm-quality-screen-v1.json` は SHA256 `bf9d6f0036bff3d6c8f49a384ecc03d849453a41e3ab93b2d90fd4c9833a4f18` の FAIL/incomplete、formal ownership_closed=false のまま。FP32-bは900秒期限、677完全NLL行と不完全末尾、期限付近のxe resetが記録され、候補2つは未起動。原因やsignal/resetの順序はこの作業では判断しない。このcontrollerがその失敗を再試行・免除したり、新しいhealth結果で旧FAILを書き換える経路はない。

`STRATA_PREFILL_NLL` と `STRATA_PREFILL_NLL_RECIPE_SHA256` を含む全 `STRATA_PREFILL_NLL*` keyを、歴史的recipeから新native環境を作る前に除去する。親環境は継承しない。recipe、exec直前、独立比較器で不在を検証する。既存selected observerだけを使用し、全位置NLL処理は起動しない。native callbackの完了はnative終了後に公開された `.meta` の厳密なschema/行/target/footerで確認し、`.meta.partial` や進捗だけを完了証拠にしない。selected形式にengine署名のrecipe digestがあるとは主張しない。私有fresh出力、request SHA、recipe SHA、Ownerの実argv/envとlog SHA、共通bindingで結ぶ。

固定identityはsource `6354f0b4ff3731466795410e11ca9ef785c9d326`、build receipt SHA `990635731be1198706abeb682817fdab247f8f51d31bab971d3d4f4db53cb639`、実engine binary SHA `85643f3926bb9d212945659cb7ddb264e101ca9b2d5fa8f19430e945140ad191`。全432 source、全314 build file、全product、compiler、runtime library、pack/MTP metadata、model shard bytes+mtime_ns、外部expert profile、既存oracle/hostfixture/tokenization recipeとsourceを前後照合する。shard全体の暗号学的SHAを取得したとはしない。旧 `own-vllm-model-smoke-v1.json` SHA `03b2bd388a897873bb2dc8c310ec7c0386e6b484828d1e633fb872cd059a14d3` は既存args/env/model identityの由来だけであり、現在のhealth gateではない。reportのみのcommit進行はengine同一性に代用しない。

Mainは `--health-receipt` と実SHAを毎回明示する。最新receiptの自動探索や代替はしない。通常終了、complete/ownership_closed、全command exit0/reaped/session empty/cleanupなし、同一boot、fresh faultなし、dumpなし、旧失敗より後の終了時刻、全旧PID/start不在を要求する。受理するscopeは以下のいずれかだけ。

- `strata-postfault-allocation-health-v1`: original_failure_sha256とbuild SHA、source6354/source_pinsが一致し、実binaryはbuildの `sycl_allocation_teardown_test` product。64-byte allocation/fill/drain/free、`--early` commandと通常終了証拠を検証する。**adapterの小さなhealthだけ**を確認し、engine/model品質の資格にはしない。
- このpackageの同一source selected-only `smoke` receipt: 明示された通常閉鎖PASS、実engine SHA、3 arm/3位置、元callback completion/recipe/captureを比較器で再確認する。これもsynthetic model mechanicsだけであり、品質採用ではない。

Mainが「最新」を選ぶ責任は残る。Mainから示されていない後続の実行を本controllerが発見したとは記録しない。通常閉鎖した古いPASSを最新FAILの代わりに受け入れる汎用schema経路はない。

| mode | nativeの固定順 | 入力/selected位置 | context/resident/chunk/R_GPU |
| --- | --- | --- | --- |
| smoke | fp32, bf16-vllm, bf16-vllm-qsa | 元synthetic129、0/63/127 | 4096/4096/64/64 |
| screen | fp32-a, fp32-b, bf16-vllm, bf16-vllm-qsa | 元natural32863、255+512i (i=0..63) | 65536/65536/8192/32768 |

全48層、smokeは2chunk、screenは5chunk、全行248320語彙、MTP4、max-new1、same queue、native/canonical dense bankとscratch alias、ROPE_TABLE=1を維持する。backendはFP32=own、BF16=vllm/tile auto、QSAは最後のarmだけ1。各armで外部profileの同一SHAから私有copyを初期化する。実args全体は既存recipeとの差分を限定して照合する。実stderrの全layer/chunk、placement、native16/backend routeと、実prompt IDs、generated count=1を検証する。QSAの新しいruntime stampは追加せず、固定envと固定dispatch sourceによる選択として記録する。

screen入力はR837の `A/own-native16-natural-input-v1.json` SHA `c0e89883e69aad815adc165ad199091c990dbcb8ce47275e155a2793056af0b1` と、保持されている `B/main-vllm-quality-v1/tokens.txt` を使う。実token file SHA `6e5341d8e4898b4932b2e77d5c2fbc6f0ba304d1cf19ef258cb6a60bba279e95` とLE-u32 ID SHA `e4754da0c72e7495319ff1f0483a39032f0cffffac532a6d902b3aeb5d434f16` を固定する。指定説明にあったB側 `input.json` は読取り時点で存在せず、元receiptの参照は上記A側だったため、その事実を明示した。欠けたfileを作ったり、tokenizerを再実行したり、旧receiptのtoken_file欄を編集しない。元tokenization、thinking、template/document source pins、hostfixture/oracle/preparer pinsは継承する。

比較器はSHA固定した旧 `own-native16-quality-screen-v1/compare_screen.py` の `open_capture`、`read_row`、`pair_metrics`、`exp_or_overflow` を変更せず使う。各selected行の全logit finite、target/shape、独立FP64 max-shift+math.fsum NLL、top1/ties/row SHAを計算する。screenではFP32-a対FP32-b、各FP32対各候補の5pairについて、64個すべてのNLL delta、最大logit差、top1、KL(control||candidate)、L1を保存する。FP32 driftとbyte equalityは**記述結果**であり、一致しないだけでは候補の起動・正常captureを失敗にしない。dense NLLとの照合や全位置NLLの基準を移植しない。元oracleの計算法/有限性検査以外に新しい品質許容値はない。生成IDや文章のexact equalityを要求しない。

`passed` は有限な診断のcapture/独立計算/所有とfaultの正常閉鎖だけを示す。`quality_accepted`、`model_quality_accepted`、`model_quality_qualified`、`performance_qualified`、`physical_262144_qualified`、`restore_qualified` は常にfalse。64相関位置の比較から一般品質、task品質、長生成、256K/RESTORE採用を主張しない。STRATA_TRACE=1、UR parameter validationとwarning logger付きであり、clean speedとは別scope。

実行rootは `B/main-vllm-selected-{mode}-{run-id}`、receiptは `A/own-vllm-selected-{mode}-{run-id}.json`。既存root/receipt/symlinkは拒否し、freshなcanonical pathだけを作る。全体で `B/owned-v0141-measurement.lock` を排他取得し、SHA固定direct_owner `a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f` を使う。nativeは各300秒/CPU18000秒/RSS88GiB、全体1800秒、work1755秒とclosure reserve45秒。最初の失敗でnative後続を止め、4armすべての元process outcome/未起動状態を残す。通常と異常のownershipを分け、物理的reap/不在だけでformal正常閉鎖に書き換えない。Ownerのpidfdでpinした子以外にsignalしない。same-UIDの既知model/fixture argvをbounded read-only scanし、不明ownerがあれば拒否する。隠れたforeign GPU workloadの排除はMainの実行管理に残る。

各nativeの直前/直後に新しいjournal cursor境界を記録し、faultは全窓で累積する。最後の空窓でfaultを消さない。driver/PCI/kernel/boot/dumpも前後照合する。timeout、budget、不明所有、partial capture、元commandのsignal/cleanupや不完全observationはFAILのまま残す。失敗時のbounded raw/meta/partial metaのpath/size/SHAも保存する。

textはcommand合計かつOwner累積64MiB、selected tensor合計256MiB以下、全raw384MiB未満、private全体1GiB以下。screen4captureは254279712 bytes。通常の各text単独traceを重複保持する理由は付けない。任意の `--trace-first-smoke` をsmoke CLIへ追加すると最初のFP32だけUR tracing+parameter validation/flush loggingを有効にするが、text予算は64MiBから拡張しない。screenにはこのflagを拒否し、長いfull API traceを採らない。workerはこの任意traceも未実行。

raw ownerはMain、consumerは本 `compare_screen.py:compare_record` とMainのselected numerical review。Mainが48時間以内のUTC review期限を明示し、closure/supersessionでも確認する。最初に完了したFP32 selected captureを新比較のcanonicalとして記録し、旧FP32-a canonicalは触らない。候補rawもこの有限比較の現consumerとしてだけ保持する。cleanupは一切行わず、Mainが閉鎖後に必要なcompact evidenceを抽出・照合し、別のpath/hash/reason deletion manifestで処理する。

Mainの正確な起動例は以下。2modeを別々に実行し、それぞれの起動前にMainが最新の適格healthとSHA、実review期限をこの3変数へ設定する。screenの前に新smokeをhealthとして指定する場合も、Mainが実receiptとSHAを明示する。変数を自動的に旧PASSへ戻す処理はない。

```sh
: "${selected_health_receipt:?Main must set latest normally closed sameboot health receipt path}"
: "${selected_health_sha256:?Main must set its verified SHA256}"
: "${selected_review_utc:?Main must set UTC ISO review deadline within48h}"
python3 -B /home/yayoi/orca/workspaces/Strata/sycl/bench/results/2026-10-15-pr6-recovery-audit/own-vllm-selected-quality-v1/run_screen.py \
  --mode smoke --run-id v1 \
  --build-receipt /home/yayoi/orca/workspaces/Strata/sycl/bench/results/2026-10-15-pr6-recovery-audit/own-vllm-engine-build-v2.json \
  --build-sha256 990635731be1198706abeb682817fdab247f8f51d31bab971d3d4f4db53cb639 \
  --health-receipt "$selected_health_receipt" --health-sha256 "$selected_health_sha256" \
  --retention-review-utc "$selected_review_utc"
```

```sh
: "${selected_health_receipt:?Main must set latest normally closed sameboot health receipt path}"
: "${selected_health_sha256:?Main must set its verified SHA256}"
: "${selected_review_utc:?Main must set UTC ISO review deadline within48h}"
python3 -B /home/yayoi/orca/workspaces/Strata/sycl/bench/results/2026-10-15-pr6-recovery-audit/own-vllm-selected-quality-v1/run_screen.py \
  --mode screen --run-id v1 \
  --build-receipt /home/yayoi/orca/workspaces/Strata/sycl/bench/results/2026-10-15-pr6-recovery-audit/own-vllm-engine-build-v2.json \
  --build-sha256 990635731be1198706abeb682817fdab247f8f51d31bab971d3d4f4db53cb639 \
  --health-receipt "$selected_health_receipt" --health-sha256 "$selected_health_sha256" \
  --retention-review-utc "$selected_review_utc"
```

独立した後日のhost再比較は `compare_screen.py --receipt PATH --receipt-sha256 SHA --output NEW_PATH`。元PASSの保存済みraw/source/inputとpinが必要で、旧FAILを通すモードではない。必要なMain-only hostfixtureの原本は旧packageのままSHA固定し、本workerはコピー・実行していない。Mainのレビューとqualification、上記2modeの全実行、数値の解釈と独立した採用判断が残る。

由来は既存StrataのR837 controller/comparator、旧native16 oracle/hostfixture、integrated model controller/receipt、postfault allocation controllerのreceipt schemaとproject owner-scan契約。すべてmanifestにSHAを記録する。新規ファイルはrepository LICENSEに従う。外部engine実装、XeStrata source/helpers/tests/fixturesは一切使用していない。歴史的directory名を含むofficial LLVMのcompiler/runtime pathは、既存receiptのbinary identityとしてだけ継承する。Main以外への送信・publish・commitはしていない。
