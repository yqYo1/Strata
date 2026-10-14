# Balanced per-layer cache proposal v1

2026-10-15 JST。未適用の実装案。書込はこの新規ディレクトリだけで、製品ソース、設定、実行中の
試験を変更していない。ビルド、テスト、GPU、Git、サービス操作、cleanup は実行していない。
`regenerate.py` の実行は、指定ソースを読み、提案の unified diff をこの場所に生成するだけである。
次の consumer / review は Main のソースと CPU レビュー。GPU・モデル検証と測定は Main が直列管理する。

`source.diff` は `--expert-cache-balanced` を明示した SYCL native / 単一 GPU の per-layer cache だけを
balanced 配分へ変える案である。既定と既存の `--expert-cache-per-layer` は最後の層へ余りを全て渡す。
フラグは `--expert-cache-per-layer`、正の total slot 数または `auto` を要求し、非 native、非 SYCL、
layer split、peer、remote cache は拒否する。CUDA/HIP の実装導入や検証は行っていない。
共通 ExpertCache の正常な既存コピーの byte 数を保ちつつ、不正 geometry / copy size は新たに拒否する。
この境界 hardening は diff に明示しており、無効入力に対する既存挙動の完全保存を主張しない。

## 調査結果と動機

`ExpertCache::layer_slot_range` の旧式は `q=S/L, lo=l*q, hi=(l==L-1 ? S : (l+1)*q)`。
128 total / 48 layers なら 0..46 層は各 2、47 層は 34 slots。
R584 の「roughly two or three slots per layer」は現在の実装には当てはまらない。
balanced 案は `r=S%L, lo=l*q+min(l,r), hi=lo+q+(l<r)` で、0..31 層を各 3、32..47 層を各 2 とする。
先頭に余りを配る規則は明示的・決定的で、層ごとの最適配分や runtime の route 適応ではない。

通常 cache128/per-layer の CPU route 割合 97.00% / 96.94% は既存
`../decode-ledger-final.ja.md` の測定値で、この案の結果ではない。配分変更で良くなるかは未測定。
現 profile の route 適合度も未確定であり、均等な quota から hit rate や速度向上は推定しない。

## 実際のソース対応

`sourcepins.json` は差分元各ファイルの byte 数 / SHA256 と読み取り参照を固定する。
Git 操作を禁止されたため、HEAD の照会は行わず、実 source の完全 hash を基準にした。
`build-sycl/CMakeCache.txt` の `CMAKE_HOME_DIRECTORY` はこの `P`、`STRATA_ENABLE_SYCL=ON`。
root `CMakeLists.txt:185` は `cmake/sycl_backend.cmake` を読み、同ファイル 95..98 の engine は
root `src/core/expert_cache.cpp` / `src/core/expert_source.cpp` / `src/core/verify.cpp`、
111..112 の executable は root `src/program/generate.cpp` を使う。
この確認はその設定の resolver の証拠で、現在 Main が実行中の binary の同一性を新たに証明しない。

| 対象 | 現状の source と提案 |
| --- | --- |
| quota と owner | `include/strata/core/expert_slot_sizes.hpp` に同じ partition の range / slot→layer helper を置く。従来の helper 引数の既定は legacy。owner の逆引きは空 range に対応した二分探索。`layer*slots/layers` による中間 overflow は使わない。 |
| native slot sizing | `generate.cpp:3123..3135` と auto retry / shrink `3158..3163` の両方へ同じ mode を渡す。`per_layer_slot_sizes` は range の slot 数だけその層の raw blob サイズを並べる。balanced fit は full round と先頭層 prefix の byte 数で決める。legacy fit は従来の最後層 tail の算術を保つ。 |
| open / native open | 現 root の ExpertCache に `open_native` という関数は存在しない。native の入口は `open_sized`。その内部で総 byte 数を「1 byte の疑似 slot 数」として `open` する方式をやめ、実 slot 数と別の arena byte 数で `open_impl` する。uniform `open` / `open_uninitialized` は同じ entry を使う。割当前に int32 slot / residency / byte overflow と balanced の最大 512 quota を検査する。 |
| offsets / blob capacity | 各 slot は raw blob を 256 B に丸めた間隔で配置し、offset の合計を ptrdiff/int64 上限で検査。raw capacity を保持し、fill / blocking fill / queued fill / verify の全四入口でコピーの上限を検査。padding をコピー元の存在保証とみなさない。0 byte request はその slot の raw blob。最大 blob を小さい slot へコピーしない。 |
| setter / lifecycle | partition は open 前に一度設定。open 中の mode 変更を拒否し、close→auto retry は mode を保持する。per-layer admission の toggle は最初の admit 後に拒否する。cursor は実 slot 数と選んだ partition で初期化される。 |
| profile fill | `generate.cpp:3294..3331` は per-layer 時に ranking 全体を走査し、quota が空/満杯の層だけ skip する。最初に実際に fill した pair / slot を記録して verify する。`S<L` で `profile[0]` の quota が空なら、それを読戻し対象にして誤失敗する問題を避ける。 |
| native file addressing | `FileExpertSource::open` の native layout は `layer_offsets_` とその層の raw `layer_blob_bytes_` を保持する。GPU slot 番号をファイルの layer/expert offset として使わない。`pin_cache_complement` `1330..1346` は `slot_of(layer,expert)` を列挙して正しい層の byte 数を使うため、新たな q の逆算は不要。 |
| adaptive exchange | serve `generate.cpp:5084..5130` と one-shot `6870..6925` は候補/犠牲者を同じ層で作る。既存の event 完了後に pending residency を公開する順序は変更しない。direct H2D の前、および共通 `resident_stage_swaps` `215..221` の D2H 前に、balanced の range owner と raw capacity を照合する。新たな swap rule はない。 |
| hit map / native dispatch | `host_res` の組立 `generate.cpp:4146..4160` は `(layer,expert)→slot` の列挙。`expert_source.cpp:1878..1921,2041..2045,2102..2103,2395..2418` は明示 table と `cache_slot_off[slot]` を使う。range を別計算する hitmap はない。既存 CPU/GPU entry 分類と route ledger は変更しない。 |
| verifier / graphs | serve `generate.cpp:4731..4736` と one-shot `6801..6806` は `VerifyHits` に実 slots / offsets を渡す。`verify.cpp:431..435` が必要時の device offset copy、`797..800` が resident plan。`src/kernels/sycl/sequence.cpp:274` の plan は res の slot と offsets で pointer を作る。SYCL の device-plan shortcut は現 `verify.cpp:426` で無効。partition の再配分は capture 前に限り、mid-request 再配置や graph pointer 更新は追加しない。 |
| prefill loan / return | `generate.cpp:4249..4253,4391..4408` は tail の実 offsets から貸出 byte を決める。serve refill `5759..5777` と standalone `6512..6524` は保持した `(pair,slot)` を元の位置へ返す。新 partition から独自に owner を推測しない。変更後の tail byte 数は scratch の選択や placement に波及し得るため、モデル gate は別に必要。 |

旧 ExpertCache の residency_ は adaptive swap の live table ではなく、現経路は host_res と pending を更新する。
本案はこの既存の所有権設計を変更しない。slot の partition owner は同じ層内 swap で不変のため、
新しい guard は live expert ID を初期 residency_ から推定していない。

## 割当 byte 数の正確な差

既存 decode receipt が指定する pack の `native_experts.txt` だけを読み、重みを読まず、48 層の raw
blob byte 数を `sourcepins.json.external_model_layout` に保存した。元 metadata の path / hash も同じ場所にある。
7 種類は 1,510,400 / 1,715,200 / 1,868,800 / 1,971,200 / 2,176,000 / 2,329,600 / 2,662,400 B。
全て 256 B の倍数、48 層の一巡は 98,227,200 B、最後の層は 2,662,400 B。

`a[l]=align256(blob[l])`、`S=48*q+r` とすると、arena の厳密な byte 数は次になる。

```
legacy   = q * sum(a) + r * a[47]
balanced = q * sum(a) + sum(a[0:r])
delta    = sum(a[0:r]) - r * a[47]
```

次表は metadata からの整数算術であり、割当や測定の結果ではない。両実装ともこの byte 数の単一
device arena を要求し、host offset table / raw-capacity vector / staging は別である。

| total slots | legacy arena B | balanced arena B | balanced − legacy B |
| ---: | ---: | ---: | ---: |
| 1 | 2,662,400 | 2,176,000 | −486,400 |
| 47 | 125,132,800 | 95,564,800 | −29,568,000 |
| 48 | 98,227,200 | 98,227,200 | 0 |
| 49 | 100,889,600 | 100,403,200 | −486,400 |
| 128 | 281,651,200 | 260,044,800 | −21,606,400 |
| 192 | 392,908,800 | 392,908,800 | 0 |
| 256 | 533,734,400 | 521,190,400 | −12,544,000 |
| 512 | 1,067,468,800 | 1,045,862,400 | −21,606,400 |
| 24,576 | 50,292,326,400 | 50,292,326,400 | 0 |

128 slots は旧 268.603515625 MiB → 新 247.998046875 MiB。これは cache128 を 1.3 GiB とみなす算術ではない。
実際の要求では live allocation bytes と free memory を照合して初めて容量の証拠とする。
この差分が追加する host raw-capacity vector は 128 slots 時に payload 1,024 B
（別途 vector object / allocator overhead）。staging の最終 raw 最大容量はこの形状では両側とも 2,662,400 B。

## fixture 案と次の CPU レビュー

新しい shared fixture は CUDA/SYCL header や runtime を使わない。
既存 `tests/cpu/decode_joined_mtp4.cpp` の main からも呼び、既存 joined ledger の照合に加えて
slot ownership の boundary と capacity を確認する。単独 CPU target も提案し、既存
`tests/sycl/host_engine.cpp` の実 open/admit/fill/verify 例は両 partition を試すよう広げる。
名称が host_engine でもこちらの既存 test は GPU runtime を使うため、CPU-only の証拠とは分ける。

- 48 層で S=0/1/47/48/49/128/512/24576 の連続 range、空 quota、全 slot owner の逆引き。
  128 の旧 2×47+34 と新 3×32+2×16、512 の新 11×32+10×16 を明示した期待値で確認。
- balanced の S=24577、E=1 の S=49、負値、巨大整数を共通 helper で検査する。
  24577 slot の generic partition 自体は算術上作れても、実 cache の balanced admission は 512 quota を越すため拒否する。
- 1/255/256/257 B と未整列の 5001/2057/9000 B に対し、具体的な累積 offset、raw 容量の直前/一致/直後、
  alignment padding、0/負 copy request を確認。実 metadata 全 48 層・7 blob サイズの b−1/b/b+1 も検査する。
- `INT64_MAX` の range と owner、slot offset の signed-addressable 上限、round 合計の uint64 overflow。
  0/負 blob と空 layer vector を拒否する。budget 0 / exact−1 / exact と最大 slot 数、非整数比、shrink を確認する。
- 3 層 heterogeneous blob の既知の 8-slot 配置と正確な 51,712 B / 40,704 B、
  実 48 層 128-slot の 281,651,200 B / 260,044,800 B を期待値に持つ。
  legacy は round 境界で必要 bytes が下がり得るので、単純な monotonic binary search は使わない。

Main が製品 tree へ適用して source review を終えた後に使える CPU-only のコマンド例
（このエージェントは実行していない。`$review_out` は Main が指定する専用の新規ディレクトリ）。

```sh
c++ -std=c++17 -Wall -Wextra -Werror -O2 -Iinclude tests/cpu/expert_slot_partition.cpp -o "$review_out/expert-slot-partition"
"$review_out/expert-slot-partition"
c++ -std=c++17 -Wall -Wextra -Werror -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Iinclude tests/cpu/expert_slot_partition.cpp -o "$review_out/expert-slot-partition-sanitized"
"$review_out/expert-slot-partition-sanitized"
c++ -std=c++17 -UNDEBUG -Wall -Wextra -Werror -pthread -O2 -Iinclude tests/cpu/decode_joined_mtp4.cpp -o "$review_out/decode-joined"
"$review_out/decode-joined"
```

未実行の fixture で PASS を主張しない。CPU helper が通過しても GPU upload、driver、graph replay、
native kernel 数値、モデル品質や性能の証明にはならない。
source review の次に Main が実 cache の mode lock / close→reopen / uniform・sized の cursor / zero quota / first-filled verify、
全 slot の payload / guard boundary を確認する。実 native shape の小規模 GPU gate と repeated serve、
prefill loan の return、adaptation on/off は Main の別 measurement で管理する。

## 公平な比較と数値 gate

最初は既存の検証済み FP32 residual / 固定4096-row prefix、pool5+issuer、同じ native model / profile、
context / MTP4 を固定する。static placement は両側 `--adapt-swaps 0`。adaptive policy は同じ設定の別比較にする。
配分変更だけを 128 vs 128 で見る比較と、同じ VRAM budget を与える容量比較を分けて記録する。

旧128の arena budget 281,651,200 B で最大 slot 数を選ぶと、balanced は 138 slots / 280,832,000 B、
未使用819,200 B、139 slots は283,008,000 Bで予算超過となる。これは旧128 vs 新138の公平な上限予算比較案であり、
実 allocation が厳密に同 byte の比較ではない。余りは必ず記録し、slot 数と bytes の両方を提示する。
exact allocation equality が必要なら明示 padding を持つ別実験が必要で、本案に dummy allocation は加えない。
auto は同じ uniform-derived budget でも mode により slot 数が変わるため、別 arm として実 count / bytes を記録する。

cache の節約で residual placement / scratch prefix / graph allocation が変われば、それは同条件の比較として扱わない。
request ごとの pre/post allocation、prefill、MTP detach/restore、decode 開始・終了の free bytes を保存し、
既存 256K host fallback を保持する。metadata の収まりだけで live VRAM の入場は判定しない。
route-ID histogram が得られた場合だけ、この同じ quota/byte plan を用いて観測 route の static hit ceiling を出す。
集約97%だけから各 profile rank の頻度を復元しない。

GPU owner を変えると native Q8_1 GPU と CPU IQ dot の算術差で token / logit が変わり得る。
balanced が旧 token と完全一致することを採用条件と解釈しない。
まず同じ配置での再現、全 entry の owner 一意性、missing/double work ゼロ、slot bytes と correct expert の一致、
fresh fault ゼロを gate とし、変更された配置は teacher-forced logit / logprob / KL 等と独立した品質基準で評価する。
MTP acceptance や生成列が違えば wall の差を cache 性能だけに帰属させない。
性能は診断を外した、順序を交互にした複数の clean per-request samples で判断する。
本案は 256K、RESTORE、coupled、cancel、全 T、品質、性能を検証済みとしていない。

## 再生成と保管

`sourcepins.json`、この README、`source.diff`、`regenerate.py` だけを保管する。
大きい trace、tensor、ログ、別 progress backup は作っていない。削除したファイルはなく、deletion manifest は不要。

```sh
python3 bench/results/2026-10-15-pr6-recovery-audit/balanced-layer-cache-v1/regenerate.py
sha256sum bench/results/2026-10-15-pr6-recovery-audit/balanced-layer-cache-v1/source.diff
```

再生成は base source hash が変わると停止し、実 source へ適用しない。
Main の適用後や別の source revision では、hash を書き換えるだけで済ませず、各 anchor と新しい source に対する
差分を review して再生成する。proposal の hash と未検証の状態を Main の integration 記録へ残す。
