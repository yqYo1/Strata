# CPU GU lossless predecode 局所反証 fixture v1

2026-10-15 JST。**proposal_only / 未 build・未 test・未測定・製品未適用**。
このディレクトリだけを作成した。Git、GPU、service、benchmark、cleanup は実行していない。
Main が既存の実行を閉じ、build/test/GPU/cleanup を直列に管理してから実行するためのソース案。
速度・数値合格・モデル効果はまだない。

Ryzen 5 5600X / Zen 3 AVX2、GU types 18 IQ3_XXS、22 IQ2_S、21 IQ3_S、H=2560、FF=640 が対象。
仮説は「後続 token が同じ expert の GU を使うとき、codebook/sign decode の再利用が追加読出量を上回るか」。
packed も expanded も expert ごとに別の実配列を持ち、同じ packed bytes から展開する。
重量は deterministic な合法 packed IQ codes と Q8_K activation の合成 fixture。
モデルファイルは開かず、実モデル分布や CPU cache hit rate は測らない。

## ファイル

- `probe.cpp`: unchanged product `iq_avx2.cpp` と `iq_single_avx2.cpp` を一つの CPU-only TU に include。
  private decoder、GGML 順 exact dot、既存 custom dot を raw oracle として使う。
  新しい prepare、byte inverse、expanded exact/custom、GU+SwiGLU、correctness、timing を実装。
- `build_probe.py`: pin を照合し、既存 clang/znver3 と同じ static GGML archives で clean/ASan+UBSan build。
  compile command、実 target macros、compiler/archive/source/binary hashes を保存する。実行はしない。
- `run_main.py`: Main 専用 controller 案。共通 lock と既存 `direct_owner.py` を使い、
  clean correctness → sanitized correctness → clean timing を直列で実行する。
  全個別 JSONL sample、command status、全 output-row 数、paired ratio を receipt に保持。
  shared lock は非協調 job の不在を証明しないため、Main 自身の idle 確認が前提。
- `pins.json`: 読んだ source/reference と既存 provider の pins。`manifest.json` はこの提案の内容 hash。

## 表現と所有 byte 数

各 block は `[original half-D bits:2][signed codebook byte:256][raw subscales:8 or 4]`。
IQ3_XXS の subscale は sign-word の上位 nibble を half ごとに 1 byte、IQ2_S は元の 8 bytes、
IQ3_S は元の 4 bytes をそのまま持つ。float 化、fp16 再変換、再量子化、source pointer、packed witness はない。
packed と expanded payload はそれぞれ匿名 mmap。通常ページを `MADV_NOHUGEPAGE` で明示し、
payload 要求 byte とページ丸め後の mapping byte を別欄で出す。allocator の隠れた padding は使わない。
controller object、activation、inverse-oracle table と OS metadata は別所有で、下表に含めない。

| type | packed block B | expanded block B | packed GU B/expert | expanded GU B/expert | ratio | expanded 16 experts B | expanded 64 experts B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 18 IQ3_XXS | 98 | 266 | 1,254,400 | 3,404,800 | 2.7143 | 54,476,800 | 217,907,200 |
| 22 IQ2_S | 82 | 266 | 1,049,600 | 3,404,800 | 3.2439 | 54,476,800 | 217,907,200 |
| 21 IQ3_S | 110 | 262 | 1,408,000 | 3,353,600 | 2.3818 | 53,657,600 | 214,630,400 |

上は寸法算術で測定ではない。signed bytes だけの R595 の約 50/200 MiB に加えて metadata が必要。
実装の payload read 比は **2.38–3.24 倍**。R595 の metadata 除外 2.33–3.12 倍より大きい。
16 expert は packed が LLC に入り得る中間条件、64 expert は packed/expanded とも Zen 3 の 32 MiB LLC を超える条件。
1 expert hot は両 arm をそれぞれ prewarm する。64 expert の各 arm は同じ順で全 expert を sequential sweep。
logical payload bytes は DRAM traffic でも帯域実測でもない。

## 数学の contract

初期化時に codebook の各 magnitude が 1..127 で各 entry が一意であることを確認する。
その制約が変われば中止する。signed expanded byte を `w` として
`maddubs(abs_epi8(w), sign_epi8(q8_activation,w))` を使う。
重み側へ activation sign を移す実装は使わない。Q8_K の -128 を負 sign で反転すると
PSIGN の byte wrap によって -128 のままになる元の挙動を保持する。
activation=0 も元の dot と同じ。grid に zero magnitude が導入された場合はこの形式を拒否する。

exact arm は現 GGML / `IQ_SINGLE_EXACT` と同じ half lane、IQ2_S の 16-value scale lane、
偶奇 half の int32 accumulator、block 順 FMA、high+low → movehl → movehdup の水平和を保つ。
IQ3_XXS の 0.25 / IQ2_S の 0.125 は最終水平和の後に掛ける。
custom arm は現 `iq_avx2.cpp` と同じ half 順 int32 accumulator、
`(half_D * format_K) * activation_D`、block FMA、2 回の hadd を使う。
元の exact/custom の演算順を混ぜない。nt2..4 は同じ math の別 template arm で、decoded weight を token 間に共有する。

packed controls は GGML single-row、existing exact helper、existing custom multi。
新候補の raw gate と raw up を個別に照合した上で、両 GU + SwiGLU の全出力も照合する。
SwiGLU の飽和で raw 不一致を隠さない。exact は GGML と existing exact、custom は existing custom に一致が必要。
custom と GGML の異なる丸めを候補の不合格/合格へ混同しない。

`-fno-fast-math -fno-associative-math -ffp-contract=off` を用い、明示 FMA の順序を固定する。
source-level 保存はまだ実機検証ではない。Main は correctness と必要なら disassembly で確認する。

## correctness gate

独立 scalar inverse は expanded bytes だけから codebook/sign indices と元 D/subscales を復元し、
packed 全 byte と比較する。重複 codebook を canonicalize してごまかさず拒否する。
この oracle と reconstruct/read 処理の時間から性能は主張しない。

各 format の全 codebook index、全 sign byte / 7-bit sign index、全 scale nibble を field として網羅する。
全 field の Cartesian product を網羅するものではない。乱数 pattern、全 0/全 1 codes/sign/scales、
activation -128/0/+127/-127/±1、全 zero、全 -128、D ±0 / half subnormal min/max / half max /
±Inf / NaN、activation D の 0 / subnormal / Inf / NaN を含む。

9 row の nb=1/10、odd/partial/empty range、expanded byte offset 0..31、activation の自然 alignment 単位の offset 0..31、
packed/expanded/activation の両側 guard page、全 GU 640 row/nb10、nt1..4 を試す。
activation は block_q8_K の自然 alignment を守る。無意味な unaligned typed activation access は使わない。
finite output は signed zero を含む bitwise、Inf は class+sign、NaN は NaN class が必要。
NaN payload の一致や FP exception flags の一致は gate に含めない。
canary 外書込み、raw g/u 不一致、SwiGLU 不一致、inverse 不一致は hard failure。

ASan/UBSan は fixture と include した両 product CPU sources を instrument する。
prebuilt GGML static archive 内部は instrument されない。この限界を receipt に残す。
失敗時の stderr context と原 receipt を保持し、test を緩めず Main が原因を調べる。

## Main 実行案

未実行の command。既存 build/test/GPU/cleanup を全て閉じ、同じ CPU の SMT sibling も idle にする。
controller は pin した compiler/archives を使うため、勝手に別 build に fallback しない。
`--core` は Main が実環境の affinity/topology を確認して選ぶ。

```sh
python3 bench/results/2026-10-15-pr6-recovery-audit/cpu-predecode-probe-v1/run_main.py \
  --out-dir /home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-primary-cpu-predecode-root-v1 \
  --core 0 --main-serialized-owner-confirmed
```

これは nt1 の hot/16/64 expert 比較。追加の `--multi` は nt2..4 custom arms を独立に追加する。
Main が既存 serial controller に取り込む場合の低レベル command:

```sh
python3 bench/results/2026-10-15-pr6-recovery-audit/cpu-predecode-probe-v1/build_probe.py --out-dir /PRIVATE/clean
python3 bench/results/2026-10-15-pr6-recovery-audit/cpu-predecode-probe-v1/build_probe.py --out-dir /PRIVATE/sanitized --sanitize
taskset -c 0 /PRIVATE/clean/fixture --correctness
taskset -c 0 /PRIVATE/sanitized/fixture --correctness
taskset -c 0 /PRIVATE/clean/fixture --timing 1 1 12 8 --correctness-approved-by-main
taskset -c 0 /PRIVATE/clean/fixture --timing 16 1 12 2 --correctness-approved-by-main
taskset -c 0 /PRIVATE/clean/fixture --timing 64 1 12 1 --correctness-approved-by-main
```

`LD_LIBRARY_PATH` 等は過去の provider receipt と同じ実環境が必要。高レベル controller が設定する。
低レベル実行も Main の共通 owner/deadline/log-budget/lock の下で行う。
`/PRIVATE` は Main の新規実験専用 directory に置き換える。

各 timing process は 3 formats を順に扱い、format ごとに 12 rounds × 3 paired comparisons × ABBA 4 samples。
**432 clean samples/process** を省略せず保持する。pair 順と ABBA 向きを固定 seed で randomized にする。
nt1 は expanded exact vs GGML / existing exact、および existing exact vs GGML。
nt2..4 は expanded custom vs existing custom、および両 arm と GGML を比較する。
全 arm は同じ expert/token/output-row 数。各 sample の前に同じ cohort を 1 sweep prewarm。
source/activation fingerprint と sample の順序・仕事量を記録する。

prepare は destination が未 touch の fresh anonymous mapping への cold first-touch fill と、
同じ destination の overwrite refill を 1 件ずつ別欄で出す。cold は page fault を含み、allocation syscall は含まない。
prepare の 2 件は分布/信頼区間を示さない。hit kernel sample に prepare/oracle/入力生成を混ぜない。
定常 hit の差が小さければ prepare の追加分を回収できるとは主張しない。

wall は MONOTONIC_RAW、CPU service は CLOCK_THREAD_CPUTIME_ID、TSC は fenced RDTSCP の別列。
TSC は core cycles ではなく、hardware perf CPU-cycle counter は未実装。
`stamp` は cpu→wall→thread CPU→TSC の順で取り、clock read overhead は値に残る。
開始/終了 sched CPU と RDTSCP AUX の差があれば、その sample と run を不合格にする。
単一 CPU affinity と端点 check は NMI/SMT contention/frequency stability の完全な証明ではない。
Main は topology/governor、provider compile flags、source/archive hashes、binary、実 environment を receipt と照合する。

## 棄却しやすい判断

finite/例外 class/guard/reconstruction gate が落ちたら timing に進まない。
nt1 の主比較は既存 exact に対しても判断する。GGML だけに勝って既存 exact に負ける候補を新規採用しない。
64 expert の paired thread-CPU と wall の両方を見て、hot だけの改善を採用理由にしない。
中央値だけでなく全 12 paired ratios と順序を読む。repeatable な改善がなければここで棄却し、
8 GiB cache、LRU、lease、ExpertSource、model dispatch、GPU upload へ進めない。

局所 win があっても採用ではない。実モデルの expert 再訪頻度、miss/fill 代、pool 並列、RAM pressure、
数値/greedy IDs と clean decode wall は未評価。全モデル展開、モデル再量子化、Down、GPU packed upload、
本番 cache の新設はこの提案の対象外。

保持は source/commands/pins、個別 sample、correctness/fault/exit、判断。successful full compile/stdout の重複を
閉じた後に compact receipt と照合して整理するかは Main の次の保持レビューで決める。
この提案作成ではファイル削除や cleanup をしていない。
