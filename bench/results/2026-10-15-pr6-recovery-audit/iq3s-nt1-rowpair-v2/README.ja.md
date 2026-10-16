Main v2: Intel LLVM compile identified missing x86intrin.h declaration for __rdtscp. Added the header in the independent fixture only. Original proposal/build FAIL remains in v1; product diff unchanged.

# IQ3_S nt=1 独立二row probe v1（未適用・未compile・未実行）

2026-10-15。対象はRyzen 5600X、Clang/znver3、AVX2、IQ3_S gate/up、Q8_K activation、
N=2560 / FF=640 / nt=1だけ。MainのGPU試験中に製品treeを固定し、ここに提案だけを作った。
製品source変更、build、fixture実行、GPU、Git、service操作、cleanupは行っていない。
速度・正しさの測定値はない。R582はILP仮説であり、既存GGML vec_dotもAVX2 vector命令を使う。
scalar命令への置換による改善という前提は置かない。既存の品質FAILを変更しない。

`source.diff`は実際に読んだ現在の3fileへの非適用差分。全面source overlayは保存しない。
`pins.json`は元source/依存source/参照reportのSHA256、`manifest.json`は差分・提案適用後bytes・
このartifact内sourceのSHA256。Git identityは新規取得していない。原コードとGGML由来部分の
MIT著作権表記・LICENSEを維持する。読み取りにはプロジェクト専用strata-upstream-sync skillと
ARTIFACT_RETENTIONを使用した。禁止record、private Codex/envscan、active logは読んでいない。

## 実装とdispatch

`iq_single_avx2.cpp`の既存exact helper本体は維持し、同TUへ独立helperを追加する。
`rowpair_kernel.inc` / `rowpair_public.inc`が追加部分の唯一の生成元。
独立probeの`probe_candidate.cpp`も同じsnippetをincludeするので、candidateの手書き複製を作らない。
`iq_avx2.cpp`のnt>=2 weight reuse、他形式、Down、量子化、poolの所有期間・同期は変更しない。
configured root CMakeのCPU source resolverは`src/kernels/cpu/iq_single_avx2.cpp`を既に含み、
同ファイルへAVX2/FMA/F16C flagsを設定する。追加build targetは製品CMakeへ入れない。
歴史的`effective-build-flags.json`が指すGGML checkoutをbytesでpinした。
現在の実binaryがそのsourceをcompileした証明はMainの新build receiptで別途必要。

製品案は`STRATA_IQ3S_NT1_ROWPAIR=1`の完全一致だけで有効になる（default OFF）。
既存multi-token dispatchの後、既存`STRATA_IQ_SINGLE_EXACT`の前に置く。
既存multiが選ばれる設定、特に`STRATA_IQ_MT_MIN=1`ではそちらが引き続き先行する。
通常のmt_min=2でcandidateとsingle_exactを両方指定すれば、対象shapeだけcandidateが先行する。
`STRATA_NO_IQ256`、AVX-512 CPU除外、nt=1を守り、shape/type不一致は既存exact/GGMLへ戻る。
非Clang/非znver3は新helperもfalseを返すstub。CPU ISAを広げる提案ではない。
本番の試験では環境をprocess起動前に固定する（static getenv cache）。

隣接するoutput rows r/r+1を独立accumulatorで処理する。gateの二rowを処理した後にupの二rowを
処理し、それぞれ元のSwiGLU式へ渡す。各256-value block内の64-value groupで二rowの
lookup/dotを隣接して展開し、activationの二vector loadを共有する。weight decodeは各rowに必要。
nt>=2のweight reuseとは別の候補。Clangが独立workを実際に重ねるか、register spillが逆効果かは
assemblyと実測で判定する。二row化によりlive accumulatorは増えるため改善を約束できない。
gather、prefetch、形式変換、圧縮layout変更、float近似は導入しない。

## 数値・境界のsource review

各rowの二integer accumulatorへのgroup加算順、二accumulatorの合算、int32→FP32、
各block一回のFMA、block 0..9の順、GGMLと同じhsum treeを維持する。
二row間のFP加算やblock分割reductionを行わない。FP16→FP32とactivation scaleの積も同じ。
そのため期待するgateはraw g/uと最終ffのbitwise一致で、まだ成立を実行確認していない。
既存exact helperも独立controlとして最終ffをGGMLと比較する。任意の差を許容値へ自動で緩和しない。
差が出た場合はmismatch件数、max abs/relative/ULP、nonfinite差を残し、そのarmのtiming採用を止める。
意図的な演算順変更へ進む場合は別candidate/versionとし、logit/top-ID/greedy/MTP受理と独立task品質の
明示的gateが必要。microkernel bitwise PASSでも全model品質・coupled/RESTOREの証明にはしない。

* IQ3_S layoutはd@0, qs@2(64B), qh@66(8B), signs@74(32B), scales@106(4B)、block110B。
  static_assertでoffsetof/sizeofを固定する。Q8_Kはd@0, qs@4、block292B。row1100Bは32B整列しない。
* 全scalar metadataはbyte pointerとmemcpy、vector activationはloadu。uint16のsigned左shiftを
  新コードへ持ち込まない。9-bit indexはuint8 qsと単一qh bitから構成され、常に0..511。
  `manifest.json`にpin済みtableのentry数と最大byteをsource準備時に記録する。
* 現IQ3_S tableの最大magnitudeは15。activationの全int8格納域を保守的にabs<=128とすると、
  maddubsの一int16は2×15×128=3840で飽和しない。maddの一int32は2×3840×31=238080、
  sumi1/sumi2は各四groupで952320、合算は1904640。int32 overflowがなく、int32→FP32も正確。
  実quantizerは[-127,127]を作る。-128のsign反転がbyteでwrapする既存semanticsも変更しない。
  raw FP16 NaN/Infや非finite activationを有意なモデル入力とする品質保証は追加しない。
* 各blockのqsはj=3でも最後の64byte内、qhは最後の8byte内、signsは最後の32byte内、scaleは4byte内。
  Q8 loadはqsの最後の32byteまで。十blockのみを読む。先読みや次rowのspeculative loadをしない。
* 公開helperはn=2560,row=1100,up_off=704000、0<=r0<=r1<=640だけを受理する。
  dispatcherはint castの前にf.n_embd==2560を要求する。pointer offsetの積も最大約1.4MBで一定。
  odd tailは同じ算法のROWS=1 specialization、empty rangeはpointerを読まずtrue。
  無効geometry/range/null非emptyはfalseかつ無書込。buffer length自体はcallerが保証する既存契約。
  gate/up計1408000B、activation2920B、float output640個の有効extentと非aliasが必要。
  workersの排他的[r0,r1)だけを書き、pointerを保存しない。

## 独立fixture / benchmark（Main実行用）

`fixture_bench.cpp`は元sourceのexact helper、linked GGML traitのbaseline、上記candidateの三armを
直接呼ぶ。default OFFの製品dispatchを迂回するCPU-only probeであり、製品dispatchの実到達証明は別gate。
GGMLの実Q8_K `from_float`でactivationを一度だけ生成し、全armに同じpacked bytesを渡す。
weightはGGML IQ3_S layoutの全index範囲が有効な合成bytesで、有限dを設定する。
これは性能選別の合成fixtureであり、GGUF実モデルtensorを代替する品質試験ではない。

`--correctness`は8pattern（random/zero activation/交互極値/all-low/all-high grid/最大有限FP16 d/
subnormal d/zero d）、9range、全32byte開始offsetのcandidate、各境界のrow canary、
guard page直前の最後のpair/odd row、無効range/type/geometry/nullを検査する。
GGML/exact controlは自然alignmentの同一内容copyを使い、任意byte offsetのUBをcontrolへ要求しない。
raw candidate g/uと最終ffをbitwise比較する。検査ごとのJSONとclose passを保存し、失敗exit1、
compiler/target unavailableは77。ASan/UBSan版もMainが直列に作る。既存GGML archiveが未instrumentedなら
その内部read全てのASan証明にはならず、guard-pageとsource bounds reviewを併用する。
既存exact helperの不具合が見つかった場合も失敗を保持し、このprobeに合わせて弱めない。

`--timing`は別processでcorrectnessを実行せず、Mainによる先行receipt確認を要求するCLI markerを持つ。
markerは確認の証拠自体ではない。Mainがbitwise PASS receiptとbinary/source hashを結び付ける。
hotは1expert=1.408MB（L1/L2 residentの主張ではない）、streamは最低32expert=45.056MBで
5600Xの32MiB L3より大きいweight集合。64expertなら90.112MB。後者もDRAM traffic実測ではない。
同じ決定的bank順を全armへ渡し、各sampleに同じ一sweepのwarm-up、同じinner回のworkを与える。
baseline/candidate、exact/candidate、baseline/exactの三比較をrepeated ABBAとし、roundごとに
比較の順と左右armを反転する。個々のns、invariant TSC ticks、開始/終了CPU/AUX、ns/output-row、
jobs/output rows/dots/superblocks/grid lookups/logical weight bytesとsinkをJSONで保存する。
TSC ticksをcore cyclesと呼ばない。IPC/core cyclesが必要ならMainの別perf診断で取得し、clean timingと分ける。
開始終了CPU一致だけでは途中migrationや割込不在を証明しないため、Mainは単一coreへtasksetする。
平均だけへ集約せず、全標本・順・除外理由を残す。結果から自動speed threshold判定はしない。

inputのsource SHA256/seed/pattern/quantizer pinsが再生成identity。JSONのFNV64は補助的bytes locatorで
暗号学的hashと呼ばない。必要な実model captureはMainが別consumer/byte budgetを定めて採取する。

## 再生成とMainの実行recipe

この時点で実施したのはartifact file作成と`python3 regenerate.py --init`によるsource準備だけ。
再生成は次のコマンドで全元source pinを照合し、同じ非適用差分とmanifestを作る。
参照reportのhashは調査時点を記録し、製品source pinと区別して保持する。

```sh
python3 bench/results/2026-10-15-pr6-recovery-audit/iq3s-nt1-rowpair-v1/regenerate.py
```

以降は未実行。MainがGPU試験を閉じ、exclusive measurement/build lockとowned process不在を確認した後、
CPU build/test/timingを直列管理する。まず独立probeをbuildし、製品適用前に棄却可能にする。
`build_probe.py`へ絶対compiler path、pinsと一致するggml directory、Mainが来歴を確認したCPU/base archive、
空の実験専用out-dirを渡す。source.diffを適用する必要はない。元exact TUとcandidate TUを別objectでlinkする。
CPU archiveのcompile source/flags/dependency closureはMainの既存build receiptとarchive SHAを照合する。
scriptはcompiler version/bytes、全command、元source/archives/binary hash、build結果と各compile logを保存する。
製品binary・archive・serviceには書かない。ASanは別empty out-dirと`--sanitize`を使う。

```sh
python3 bench/results/2026-10-15-pr6-recovery-audit/iq3s-nt1-rowpair-v1/build_probe.py --compiler /ABS/clang++ --ggml-root /ABS/llama.cpp/ggml --cpu-archive /ABS/libggml-cpu.a --base-archive /ABS/libggml-base.a --out-dir /ABS/OWNED/rowpair-clean
timeout 120 taskset -c CORE /ABS/OWNED/rowpair-clean/fixture --correctness > /ABS/OWNED/correctness.jsonl
timeout 120 taskset -c CORE /ABS/OWNED/rowpair-asan/fixture --correctness > /ABS/OWNED/correctness-asan.jsonl
timeout 600 taskset -c CORE /ABS/OWNED/rowpair-clean/fixture --timing hot 1 12 8 --correctness-approved-by-main > /ABS/OWNED/hot.jsonl
timeout 600 taskset -c CORE /ABS/OWNED/rowpair-clean/fixture --timing stream 64 12 2 --correctness-approved-by-main > /ABS/OWNED/stream.jsonl
```

COREはMainがCPU topology/他workから選定する。最初のbuild例をASan用out-dirと`--sanitize`でも行ってから
上記ASan fixtureを実行する。各commandのstatusを独立に保存し、timeout/77/失敗をPASSへ変えない。
hardware、OS/runtime、compiler、affinity、CPU governor/frequency条件、worker/SMT使用、開始終了時刻、
binary/source/fixture identity、effective relevant STRATA flagsをMainのreceiptへ保存する。home-wide envdumpは不要。
raw text予算はjobごと64MiB、各sample JSONだけを保持する。成功full traceを集めない。

bitwise PASSかつclean timingに再現性がある場合に限り、Mainが`source.diff`の三fileをreviewして適用する。
pin一致を適用前に確認し、適用後bytesはmanifestのproposed_target_sha256と照合する。
既存`iq_single_exact_test`、native parity/CPU oracle、候補OFF/ONとsingle_exact/MT_MIN/NO_IQ256のdispatch、
nt2/3/4の維持をMainが検証する。`rowpair`だけで得たmicrobenchmark勝利を製品採用と扱わない。
format×ntの現censusは未取得であり、既存format別serviceとrequest-wide nt分布から対象service時間を捏造しない。
実モデルnt1/type21負荷の選別、same-source固定条件のserviceとclean end-to-end pairs、数値/MTP/quality/fault
gateまで進めて採否を決める。差がない/遅い/spillが増す場合はdefault OFFのままprobeだけ棄却可能。

保持consumerはMainのnt1 IQ3_S GU候補選別。raw model tensor/sessionやfull traceを作らない。
次reviewはMainのsource reviewと独立CPU fixture/build境界。今回削除したファイルはない。
