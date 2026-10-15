# 長文prefillのhost時間細分とGEMM実行場所

2026-10-16 JST。前goal turnはprogress。実source変更、6独立clean標本、実78K CPU時間測定、
正常GPU/host検証、compact化と9b83fbf8のcommit/pushがあり、単なる計画の繰り返しではない。
600超反復/1000 PPと70 TGは未達で、目標は維持する。現在の優先は長文prefill。

## 計測の追加

6.1Solのsource-only案`own-prefill-host-profile-v2/source.diff`をMainが全diffとcallerを審査。
SHA256 `602f6dace013af8a167f17559e480d35d2dedff69d0800aef838f587062cbdee`。
4 touched filesを実Git commit9b83fbf8のblobとhash照合した後に適用した。
独立MIT sourceだけを使用し、XeStrata/source/helpers/tests/archivesや旧observerを使っていない。

exact `STRATA_PREFILL_CPU_WAIT_PROFILE=1`のlong-only scopeを拡張し、source slab copy48区間、
router wait後のhost grouping区間、oneMKL GEMM call区間を測る。waitを重ねて足さず、各区間で
waitの増分を引いたCPU/wallとunclassified remainderを保存する。GEMMのqueue lookupとspan/shape
validation、`finish`は区間外。sourceはupload前のcopy_blob loop、groupはcounts/offset/slot/src/order
と対応表copy提出を含む。region重複/clock異常/NaN/Infはavailable0・duration-1になる。
GPU演算/待機policy/配置/同期順を変更せず、default OFF。CPU時間をcritical path時間と呼ばない。

Mainの実header consumer17 modesがPASS。元8 modesに加えてhost region・内部wait差引・nesting不正・
unwinding・時計失敗・NaN/Inf・guard overheadを検証した。empty guard10万回のCPUは
enabled0.120106358秒（1201.06ns/guard）、disabled0.000391133秒（3.91ns/guard、clock calls0）。
enabledのCPU clock readsは200002。これはhost microcase1標本で、実測値から差し引かない。
50万guardsへ単純換算すると約0.60CPU秒だが、実モデルでの正確な追加コストを証明しない。

`build_own_prefill_host_profile_v2.py`で実native/runtime/compatと新probeをbuild。
native SHA256 `850dec38169e0879fdb19ea572f20ba5eedf364b3921c9229b263788a8262705`。
runtime/compatの既存機能検証とGPU probeのsmoke4/full12ケースが正常終了、新規fault0。
最初の小GPU確認はflushed UR traceとparameter validation付き。全GPU/build/testをMainがlock下で直列化した。

## oneMKLの実行場所

Mainの新規`tools/sycl/gemm_device_probe.cpp`は現行production `Gemm::f16/bf16`を直接使う。
FP16/BF16、rows1/160/8192、GU(N1280,K2560)/Down(N2560,K640)の12ケース。
ld=N+3のpaddingと両端guard、全outputのdyadic analytic exact oracle、全入力不変性を確認した。
これはsyntheticの実行場所probeで、実モデルの全shape/stride、性能、task qualityの測定ではない。
同時に全モデルでverboseを有効にして数十万行のログを出すことはしていない。

exec前の`MKL_VERBOSE=1`を不変に保った。実libraryは
`oneMKL 2026 Update 1 Product build 20260612`と`Detected GPU0 Intel(R)_Xe2 Backend:Level_Zero`を報告。
full12とsmoke4の全GEMM callに`GPU0` suffixがあり、typeと実dimensions/strideを
`assess_mkl_device_probe_v1.py`で照合する。`mode:standard host:N/A device:N/A`はタイミングを
有効にしていない出力であり、時間が0、GPU計算がない、という意味にはしない。
CPU setupや他の未観測callでのfallbackは除外しない。

## R755のMain訂正

R755原文は不変に残す。現行compact FP16のGUはFP32だがHhはuint16である。
全T*K行を同時materializeする単純案は、GU1280*4+Hh640*2=6400B/row、
8192*10行で524,288,000B=500MiB。現在のcompact8192行分50MiBを置換するなら差分450MiB。
R755の両方FP32で600MiBという計算を現行Hh配置の見積もりには使わない。
さらに512 experts全てのFP16 GU+Down weightsは5,033,164,800Bあり、既存KV/hidden/workspaceに
追加して容易に保持できるとはしない。部分batchなら追加weight scratchも明示して計算する。
USM grouped APIは一回のcallで複数groupを表せる。異なるneをgroup_size1の別groupにする案もあり、
必ず一回のcallが一つのneだけになる、という意味にR755を読まない。host metadata依存と
library内部の起動/loopが残るため、速さは実componentで判断する。

## 実78K細分結果と次の対象

`own-prefill-host-profile-model-v2.json`は実78453 API入力、78452 batched、C8192、FP32、GPU32768行、
bank1/Direct1/専用radix1/no-token-graph。TRACEとvalidation付きの別owned processである。
全48層10chunks、正常server/native exit、所有退去、新規fault0。全clock/region/waitは正常終了。

| 区分 | CPU秒 | 経過秒 | 実回数 |
| --- | ---: | ---: | ---: |
| Scope全体 | 217.109655163 | 223.606305845 | 1 |
| 明示的GPU wait | 167.954787940 | 168.398063906 | 2053 |
| source slab準備 | 4.856936401 | 4.867617353 | 48 |
| host groupingと対応表copy提出 | 0.165362610 | 0.168038559 | 480 |
| GEMM call（queue lookup/validationを除く） | 6.784368076 | 7.282662714 | 239458 |
| 残るunclassified | 37.348200136 | 42.889923313 | — |

wait外CPU49.154867223秒の全体を上記4区分で分割できた。明示的waitはcallerCPUの77.3594%。
各host区分はwait差引後の値と同じで、そこに明示的waitの重なりは無かった。
GEMM回数はdense/shared/router等を含み、実expert数とは呼ばない。
R754の491520回は全expertがactiveという上限で、今回の実回数を置換しない。
empty guard microcaseの単純換算は239458*1201nsで約0.288秒だが、値を補正していない。

host groupingは今回0.17CPU秒だった。GPUへの移行だけをCPU律速の対処の第一候補にはしない。
GEMM submissionも6.78秒にとどまり、host call数削減だけの改善余地を全49秒と取り違えない。
grouped GEMMによるGPU効率改善の可否は別のcomponent測定で判断する。
次は37.35CPU秒のうち、層expert upload提出、host/device residual input/store提出、dequant/gather/
SwiGLU等のkernel提出を区別する。bankの`upload`と`rows`は現在`new uint8_t[]`で、通常host memory。
`ExpertCache::fill_slot`はそのsourceを`cudaMemcpyAsync`へ渡し、compatはSYCL `queue.memcpy`へ渡す。
APIの名前がAsyncでも、host memoryの登録・stagingや内部同期待ちのCPU時間があるかは未測定である。

過去の`qsa-subprofile-analysis-v2.json`（binary f01e、異なるselector等を含む）ではwallとして
upload_submit8.30秒、host residual input_submit13.88秒、host residual store21.31秒があった。
この古いwall値を現在のCPU秒へ読み替えない。現行でもcopy提出を先に細分する根拠として扱う。
現行の転送元USM化やbatchingを、まだ性能採用したとはしない。host全体のGPU化が容易とも言わない。

`mkl-device-probe-assessment-v1.json`は全16 observed callsのtype/dimensions/stride/一対一caseを照合し、
すべてGPU0だった。whole_model_fallback_excludedはfalse。MainもIntel公式2026.0 call-description文書を
直接開いてsuffixの意味を確認した。ld=N+3のprobeであり、実モデルのcontiguousを含む全strideの
実行場所を検証したものではない。次にこのprobeを実model strideへ拡張する場合も旧結果は保持する。

長い診断はclean PP標本に混ぜない。profile v1のFAIL/PASSや個別値は変えていない。
結果抽出を検証後、closed root2個の88 files/1,409,842 logical bytesをlock下で退役した。
`closed-own-prefill-host-profile-compact-v2.json`と`closed-own-prefill-host-profile-retention-v2.json`に
build/progress/旧path/bytes/hash/reasonを保存した。物理空き容量の増分とは呼ばない。
