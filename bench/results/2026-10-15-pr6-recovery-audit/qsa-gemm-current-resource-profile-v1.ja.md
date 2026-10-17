# 現行score GEMM構成の資源診断

2026-10-18 JST。Arc B570 10 GiB、Ryzen 5600X、RAM 128 GiB、xe／
Level Zero 1.17.39395+14。同じ現行バイナリ `1e7bf1cb…88b26` で、
公式 `bench.sh` の64K指定、実入力78,453 token、context 131,072、
chunk 8192、MoE window 10、BF16 native16、全hidden-state GPU保持、
score GEMM有効を診断した。短い収集確認を先に実行し、両方とも正常終了、
所有プロセスの終了と新規GPU障害0件を確認した。

[全結果](own-qsa-gemm-unitrace-direct-model-v1.json)、
[解析](qsa-gemm-current-resource-profile-v1.json)、
[再集計コマンド](summarize_qsa_gemm_unitrace_v1.py)に保存した。
コマンドは `python3 bench/results/2026-10-15-pr6-recovery-audit/summarize_qsa_gemm_unitrace_v1.py`。
同一の閉じた成功receiptだけを再集計し、失敗した原結果は置換しない。

| 観測 | 値 | 範囲 |
| --- | ---: | --- |
| QSA attention本体 | 37.078 s／120 calls | 全プロセスのdevice command時間合計 |
| vLLM由来MoE DenseKernel Tile1 | 10.166 s／21,744 calls | 同上 |
| GDN recurrence | 8.032 s／1,476 calls | 同上 |
| MemoryCopy M2D | 10.361 s／26,904 calls | 同上。Mはallocation queryの分類 |
| MemoryCopy H2D | 2.987 s／8,948 calls | 同上。物理PCIe量は未測定 |
| QSA attention区間 | 38.438 s | 長prefill区間のqueue marker間隔合計 |
| 長prefill wrapper | 116.110 s | lease retirementまで。後片付けは別 |
| 発行元スレッドCPU時間 | 110.327 s | 同スレッドだけのCPU時計 |
| うち明示的wait内部 | 75.914 s、CPU時間の68.81% | completion pollingを含み得る |
| wait外のCPU時間 | 34.413 s | source copy、submit、未分類を含む |

[呼び出し対応の追加確認](qsa-attention-current-service-binding-v1.json)で、
このattention本体の120 callsを長prefillの12 QSA層×10 chunksに対応付けた。
選択されたtemplateと唯一のproduction call site、実際のmatrix dispatch 120回・
fallback 0回を照合した。この特定カーネルの37.078 sは長prefillのGPU実行サービス
として扱える。warmupやdecodeを含む全体のdevice/API合計とは区別する。
GPU内部のメモリー待ちや同期も実行時間に入り得るため、演算器の使用率や
PCIe時間、VRAM帯域飽和の証明にはならない。

CPU時間のうち、source copy 4.716 s、GEMM submit 4.781 s、upload submit
5.660 s、host grouping 0.162 s、未分類19.066 sだった。これは最初の長入力と
collector／validation付きの結果で、未分類をCPUの推論演算やJITと断定しない。
別のruntimeスレッドもこのCPU時計に含まれない。CPUの高使用率だけでは
single-thread bottleneckの証拠にならず、wait外の処理も継続して扱う。

MoEの要求uploadは50,292,326,400 bytes、queue upload区間8.503 s。
residualの要求host転送は0 bytes、device入力コピーは75,514,757,120 bytes。
これはソース側要求量であり、unitraceの全コピーへ一括対応させたり、
実際のPCIe転送量／帯域使用率へ変換したりはしない。

device command合計109.816 s、host API elapsed合計147.130 sには、初期化、
warmup、24 token生成、後片付けが含まれる。互いに重なり得るため利用率・
critical pathの分解ではない。queue markerにもhost submission gapが含まれる。
追加同期は入れず既存drain後に集計しているが、marker・collector・validationは
時間に影響する。この実行の667.2 tok/sを、通常測定の中央値725.2 tok/sの
置換や性能デグレ判定に使わない。

attentionについてcollectorが報告したSLM 32 bytesはkernel生成時の属性で、
動的local accessor込みの使用量ではない。実ソースのpayloadは28,224 bytes。
報告されたGRF数やspill 0も、実際のoccupancyや全入力の挙動を証明しない。

次は最大の単一deviceカーネルであるQSA attentionを優先する。
現行serial softmaxは16 threadで各32 cellを処理するため、既に存在する実際の
row-major subgroup版を独立oracle・境界・寿命の検証から進める。
以前のprivate packed版の再実行やC16／C64の再調査は行わない。
これは最適化先の選定であり、演算器・VRAM帯域・barrierのどれが飽和したかの
hardware counterによる確定ではない。quality、物理262144 context／RESTORE、
長decodeは未検証で、今回の24 token生成はそれらの合格根拠にならない。
