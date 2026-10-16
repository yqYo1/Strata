# GPU内のprefill処理と中間表現の形式

2026-10-15。R671–675のFINALを受領してMainが現sourceと照合した。
調査reportは実行結果ではない。実行中のclean測定と混ぜて速度・品質の採用根拠にしない。

## 優先する変更

FP16とBF16はともに2 byteであり、同じ配置なら保存量と要求転送量は同じ。
FP16のfractionは10 bit、exponentは5 bit、BF16はfraction7 bit/exponent8 bit。
範囲内ならFP16の丸めは細かいが、繰り返す47層境界のcastでどちらが品質を保つかは未確認である。
一回だけのcastや少数位置のlogitsで全tokenのNLL/task品質を代用しない。

GPU内の追加計算が、削減するCPU処理・転送・同期の依存経路より短ければ採用候補になる。
ただしAPIのwait時間には先行する有用なGPU計算を待つ時間も含まれる。
その全時間を「CPU往復をなくせば消える損失」としない。
当面は、prefill中の不要なdecode所有領域を退去してGPU内の保存量/chunkを増やす変更と、
FP16/BF16保存を品質評価できる状態にする変更を優先する。

## R671: GPUで並べ替えても残るhost M

Mainは実`prefill.cpp:2420`以降と`sycl/moe_fused.cpp`のcompiled group APIを照合した。
現在はGPUのtop10 IDsをhostへ読み、同じqueueを同期し、CPUでcnt/off/slot/srcを作る。
compact FP16 branchはexpertごとの`ne=cnt[e]`をoneMKL GEMMのMへhost scalarとして渡す。
従ってGPUのhistogram/prefix/scatterへ替えるだけではhostへの件数返却と一回のwaitは残る。

最小案はIDs全件の代わりに512件のcountとinvalid-ID statusだけを返し、slot/srcをGPU内で作る。
GEMM/重み/活性化の精度は維持できるが、atomic scatterの行順を含めて出力を検査する必要がある。
既存groupはinvalid IDを無視するため、無変更でproductionへ接続してはならない。
既存fused/MMQ全体への切替は活性化のint8/Q8_1化も伴い、同じFP16計算の置換ではない。

全context262144で、現在のIDs readbackはlogical payload約480 MiB。
一方、R_GPU4096のFP32 residualを47境界でstore/loadする要求量は約993.5 GB。
前者のAPI同期依存と後者の実際の転送時間は別に測定する。
これらはsourceの要求byte数であり、PCIe wire bytesや所要時間ではない。

## R672/R673: warm serveでもheadを退去する条件

通常のprefillとMTP prompt graphはoutput headを読まない。
Verifierの全Tのwindow/segment/work graphとMTPのordinary/coupled full-draft graphは読める。
native head object、arena、session/KV/GDN/PLE、MTP K/V-only prompt graphs、prompt-cache/RESTOREを保持し、
全head-reader graphだけをchecked drain後に退去する分離APIが必要である。
非同期commitも完了させ、head detached中のdecode/rebind/recaptureを拒否する。
cancel/失敗でpromptが戻る場合にもheadを復帰し、完了不明のDMAやrestore失敗では通常unwindをしない。

packed target head 521472000 Bとduplicate draft head 223227900 Bの和は744699900 B。
実際に空くVRAM量ではない。draftは復帰したtargetと不変mapから再構築できる候補である。
MTP expert退去で既に空いた707788800 Bをこの新案の便益に再加算しない。
NativeEmbedは通常host mappedであり、大きいVRAM解放の対象だと仮定しない。

R672のstandalone優先という推奨は採用条件にしない。
warm serveでのhead再upload/graph再作成は未測定であり、ユーザー指示どおり
反復するRAM往復との実費比較を行う。Solへ両call siteを扱うsource-only案を依頼した。

## R674: BF16は境界storageだけを変える

現rootはFP32/FP16だけで、BF16 storageは未実装。
planner/offset/2-byte stage、store/restore、auditのbool halfと表示名、fixture/receiptを明示的dtypeへ広げる。
FP32 operator bufferは維持し、一境界で一回packして次層前にFP32へ正確にexpandする。
共有`bf16_from_f32`/`f32_from_bf16`のRNE/NaN保存契約を使う。
既存SYCL postopsのlocal bfはNaNの扱いが同じではなく、無関係なactivation converterを変更しない。
262144 rowsでは2-byte residual総量5 GiBだが、checked planは実物理admission/最大context実行の証明ではない。
別Solへsource-only実装を依頼し、tests/build/GPUの実施はMainで直列管理する。

## R675: 現collectorで可能な実行窓

固定PTIは6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033、実buildはBUILD_WITH_ITT=0。
ITT hooksの接続済み扱いを止め、公式の名前付き外部制御を使う。
`--start-paused --session NAME`でnative serveを起動し、READY後に`--resume NAME`、
一つのGEN/DONE後に`--pause NAME`、QUIT/実native退去後に必要なら同じprivate sessionだけを`--stop NAME`する。
これはprefill、復帰処理、一個のoutputを含むrequest windowであり、純prefill境界ではない。

Mainの新recipeは`run_prefill_window_profile_v1.py`。
通常startupは収集から外し、513入力componentを先に実行・検証してから34603入力longへ進む。
プロファイルは性能比較のbench.sh標本に含めない。
raw Chrome履歴が必要な問い・owner・consumer・次のreviewをlaunch前に記録し、
raw+provider合計1 GiB、job1152 MiB、RSS88 GiB、finite deadlineの例外を明示する。
既存owner stdout/stderrの64 MiB上限は維持する。budget超過はincompleteであり、traceを切り捨てて成功にしない。

Chromeのgpu_op/cpu_op X eventsはmicrosecondのts/durを持つ。
device pidはsynthetic laneでありnative PIDではない。operation id/metadataで照合する。
duration合計、lane union、visible-device全lane union、host API spansは別々に算出する。
flowのH2D/D2Hラベルだけからpayload方向・wire bytesを断定しない。
trace空白を全GPUのidleと断定せず、未帰属区間と収集範囲の限界を残す。

R675 FINAL messageのSHA表示は短かったが、ファイルの実SHAは
`d1a94892ee4ce3ee90109e6761c13ded3515ee504ef58a335c9b87da23cfd4bf`。
原reportは変更しない。
