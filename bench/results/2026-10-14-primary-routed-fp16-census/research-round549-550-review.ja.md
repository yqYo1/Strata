# R549/R550: state precisionとlong-context indexer

2026-10-15 JST。原reportとhashを保存し、Mainがcurrent SYCL sourceと一次資料を確認した。
GPU benchmarkのsource/client/configは変更していない。いずれも未実装・未測定・未採用。

## R549の資料と実験の境界

[DAMP v2](https://arxiv.org/html/2608.27513v2)はrecurrent stateの低精度化を実測した一次研究だが、
arXiv preprintであり、原reportの「primary paperという意味でpeer-reviewed」は誤った表現。
一次資料であることは査読済みであることを意味しない。其のFP16/BF16結果は別モデルであり、
このモデルの品質認定には使わない。[TensorRT-LLMのguide](https://nvidia.github.io/TensorRT-LLM/deployment-guide/deployment-guide-for-qwen3.8-flash-next-on-trtllm.html#key-configuration-options)
もstate dtypeの選択肢を示すが、等価な品質の証明ではない。
[DGX Sparkの実装報告](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/main/README.md#bf16-recurrent-state-default)
は候補の根拠として残し、小規模な品質チェックと別hardwareの速度をこのforkへ転用しない。

source geometryでrecurrent matrixは36 layers × 128 × 48 × 128 × 4 bytes = 108 MiB。
16-bit storageへの変更でmatrix allocationを54 MiB減らせる計算になるが、実peak VRAMや速度は
別測定が要る。conv historyは別の4.21875 MiBで、最初の比較ではFP32のまま保持する。
使用頻度や帯域も異なるため、容量だけで大きな効果を認定しない。

原reportの「本当の16-bit storageならprefillでも全token毎に丸める必要がある」は、
全ての実装に対する必要条件ではない。current fused recurrenceはchunk/windowの始めに
stateをFP32 registerへ読み、token loop中はregisterに保持する。prefillはchunk末尾に、
verifierはaccepted commit末尾にpersistent stateを書き戻す。verification自体は未commit
stateを変更せず、commitで採用prefixを再計算する。

従って二つの明確な候補を区別する。

- **storage boundaryだけ16-bit**: kernel内のFP32 register/updateを維持し、persistent read/writeで
  castする。chunk長、MTPのaccepted-prefix長で丸める頻度が変わるため、chunking、MTP有無、
  途中snapshot/restoreとcontinuationを比較する。これはuserの一時FP32を許す提案に沿う。
- **全logical tokenで16-bitへ丸める**: token loop内にもround-tripを入れる独立候補。
  prefill/decodeの丸める頻度を揃える検証に使えるが、追加conversionと別の誤差過程になる。
  其の結果を前者の実装や速度と同じものとして扱わない。

現在のstateはFP32を保持し、exact-modelのtask品質・long-context・停止挙動を確認してから
判断する。FP16/BF16は別armとする。出力完全一致だけをtask品質条件にはしない。
token単位の大量サンプルを独立な品質試行として数えない。

source anchors: `src/kernels/sycl/fused_gdn.cpp:104-168,249-258,280-290`;
`src/core/verify.cpp:630-652,1160-1170`;
`src/core/session.cpp:38-44,99`。reportのsource hashesと一致するbytesを確認した。

## R550のsource budget修正

現在のroot SYCL `qsa_block_topk`が32-bit thresholdを32 passesで求め、最後にcountとemitの
2 passesを行うことは確認できた。headerにある4-passの説明はroot SYCLの実体ではない。
`qsa_block_scores_tc`はこのbackendでfalseを返す。decoderのcaptured score gridはcontext
capacityに固定される一方、有効score candidatesとselected KV cellsは別の数になる。
radixを変えてもindexerの全block scoringは残る。

原R550のtableの`Score address references`列は誤り。queryは4 heads × 128 FP32 valuesで
2,048 bytesであり、512 bytesではない。さらにsource loopでは128-value keyを4 headsで
参照する。1 candidateのnominal operand referencesはkey 2,048 + query 2,048 + score write 4
= **4,100 bytes**。原列の1,028 bytesをこの4,100で置き換える。
compilerがkeyを保持すればloadは減り、queryも候補間で再利用され得る。unique inputと
sourceの参照回数、device cache/DRAM trafficを区別する。物理帯域の上限計算には使わない。

queryごとのunique input extentは`N*512 + 2048` bytes、score outputは`N*4` bytes。
これは4 headsが同じkeyを読むことと、全candidateが同じqueryを読むことを重複計上しない
address extentである。複数query/window/layer間のcache reuseやtail key更新は別条件になる。
top-kの`S*34*4*12`もsource-level reread量で、cache hitを除いたphysical trafficではない。
INT8 KVの1 cell/layer 1,056 bytes（64値毎のFP16 scale）と其のselected payload計算は確認できた。

4-pass exact weighted byte-radixは候補として残すが、histogram/reduction/atomic costを伴う。
「score-list pass数が減る」だけでselectorやdecodeが速いとは判断しない。
signed zero、NaN、infinity、partial tail weight、stable ascending cell IDsを維持する。
user指定の高速な既存fork回復を先に監査し、其の実装に同等のselectorがあれば重複実装を避ける。
現在のbenchmark後のPR #6監査・回復を優先し、必要な時にMainが直列で比較する。

source anchors: `src/kernels/sycl/qsa_select.cpp:18-27,47-64,109-159`;
`src/core/verify.cpp:748-753`;
`include/strata/kernels/kv_q8.hpp:25-28`。
