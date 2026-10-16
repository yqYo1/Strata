# 長大prefill用の容量試算・現在の前提

2026-10-15。256Kは262144 token、GiBは2^30 byte。
`prefill-capacity-plan.ja.md`の初期表に、その後のSYCL workspace削減とscratch共有を反映する。
**現在の一層の全重みだけをVRAMに置き、decode cache・output head・MTP重みを保持しない**仮定。
全層denseを常駐する現在のserveとは異なる配置であり、実装済みの最大contextとは言わない。

実pack/GGUF headerを全48層について集計した最大はlayer47（第48層）の1440161792 B、
**1.3413 GiB**。内訳は512 expertが1363148800 B、canonical重み32366592 B、
native dense重み44646400 B。元のpacked量子化形式を使い、全expertをFP16展開して常駐させない。
CPU上のembedding/PLE lookup tableはこのVRAM重みに含めない。
payloadの集計であり、ホストRSSやPCIe実転送量を計測した数字ではない。

実GPU allocation fixtureでC4096/C8192の作業領域を確保・使用範囲を確認済み。
full-resident KVでは不要な一層分KV stageを引いたsource計算が以下。
scratch共有はdefault OFFの検証候補であり、この表では有効と仮定する。
live演算と累積はFP32を維持し、A16は**層境界で保存するhidden**の2 byte化を意味する。
全カーネルの作業領域を一律16 bit化した数字ではない。

| 予約するVRAM | 4Kチャンク | 8Kチャンク |
| --- | ---: | ---: |
| 最大の一層の重み | 1.3413 GiB | 1.3413 GiB |
| prefill作業領域 | 1.1424 GiB | 2.1528 GiB |
| GDN全36層state・PLE history | 約0.1099 GiB | 約0.1099 GiB |
| runtime等の余裕・小owner予算 | 0.5156 GiB | 0.5156 GiB |
| **KVと保存hidden用に残る量** | **6.8244 GiB** | **5.8139 GiB** |

GPU総量の実表示は10666115072 B = 9.9336 GiB。
token-ID buffer 16384/32768 Bも差し引いた。
runtime余裕512 MiB・小owner16 MiBは設計上の仮置きで、実測した最大使用量ではない。
runtime余裕を1 GiBへ増やした感度分析も構造化結果に残す。

4Kのbyte収支は
`10666115072 - 1440161792 - 1226622208 - 117669888 - 368640 - 16777216 - 536870912 - 16384 = 7327628032`。
主表の作業領域は、実GPUで確保したstreamed-modeのslabから、全層KV常駐では使わない
一層分のKV stage 276825088 Bを除いたsource上の計算である。
実GPU gateはslabの確保・relayout・guard・不足予約の拒否を確認したもので、
最大レイヤーと全contextを同時に推論した実測ではない。

保存hiddenはhyperconnection4×2560要素×2 B、**20480 B/token**。
262144 tokenで**5 GiB**。全層分を同時保存する計算ではなく、一つの全prefix bankを順次更新する。
FP16とBF16の容量は同じ。品質合格は容量計算からは得られない。

全12 QSA層のKVに、scale、indexer、page table、RoPE等を含めると、
4 token単位のcontext CについてK8V8は`14476*C+43200` B、K8V4は`11596*C+43200` B。
256Kでそれぞれ**3.5342 GiB / 2.8311 GiB**。

| チャンク | KV形式 | KVと全hiddenを同じ長さで持つ計算上限 | 256K KV後のhidden・byte上限 | FP16変換領域と現在のchunk丸めを含むGPU hidden |
| --- | --- | ---: | ---: | ---: |
| 4K | K8V8 | 209620 token | 172499行 | **167936行** |
| 4K | K8V4 | 228444 token | 209363行 | **204800行** |
| 8K | K8V8 | 178584 token | 119522行 | **106496行** |
| 8K | K8V4 | 194616 token | 156386行 | **147456行** |

右端はhiddenの残りをRAMへ置く既存FP16 plannerの制約。
RAM tailがある場合は変換用にGPU上のFP16一chunk（4Kで80 MiB、8Kで160 MiB）が別途必要。
GPU prefixをさらにchunk単位に切り下げる。全hiddenがGPUに入る中央列にはこのstagingは不要。
4Kで256K KVを保持する場合、K8V8のhiddenはGPU 167936行・RAM 94208行、
K8V4はGPU 204800行・RAM 57344行となる。
以前の`prefill-capacity-alias-delta-v1.json`のpartial hidden値はbyte上限であり、
この変換領域と丸めを含む実planner行数ではなかった。元receiptは変更しない。
`measure_prefill_capacity_fp16_admission_v1.py`はproduction planner headerを直接compileし、
partial auto配置、全hidden上限の受理、次4 tokenの拒否を検査するためのhost専用recipeである。
実行成否は同名のJSON receiptで確認する。
このhost gateは正常終了し、alias OFF/ONとruntime余裕512 MiB/1 GiBの16条件について、
上記三検査の計48ケースが合格した（`prefill-capacity-fp16-admission-v1.json`）。
GPU割当やモデル実行をこのhost検査で行ったという意味ではない。

runtime余裕を1 GiBに増やす場合、4KのKV・hidden用予算は6.3244 GiBとなる。
同じ長さの全KV・全hiddenを持つ上限はK8V8 194264 token / K8V4 211704 token。
256K KVを先に確保する場合のGPU hiddenは、stagingとchunk丸め込みでそれぞれ
139264行 / 176128行となる。512 MiB予算の表を物理的な確保保証とは扱わない。

全12層KVと256K hiddenを同時にGPUへ残すには、4KでもK8V8で**1.7098 GiB**、
K8V4で**1.0067 GiB**不足する。単なるV4化だけでは全体は入らない。
一方、prefill中のKV/indexerも現在のQSA一層だけGPUへ置き、完了層をRAMへ退避する別案なら、
全256K hiddenを保持後の余裕は4KでK8V8 **1.4726 GiB**、K8V4 **1.5312 GiB**。
8Kではそれぞれ0.4621/0.5207 GiB。共通RoPEは残す。
これは過去層KV/indexerを含めた所有配置変更を要し、実装・速度・RESTOREの合格ではない。

主表は共通weight bankを各層のcombined最大へ詰め直す仮定。
expert/denseを別々の最大bankで固定するなら追加65011584 Bを引く。
次層norm等は小owner予算から明示して扱う必要がある。
現在のNativeDense bank案だけではcanonical全層は常駐したままで、主表の仮定を実現しない。
K8V4も現在のlayer-major経路は拒否するため、容量のみの仮定である。

根拠は`prefill-capacity-metadata-v2.json`、`prefill-scratch-allocation-v2.json`、
`prefill-capacity-alias-delta-v1.json`と各recipe/source hash。
物理262144モデル実行、FP16/BF16数値品質、実runtime最大量、KV退避・decode再構成費用は別の検証項目。

新nativeのFP32/QSA OFF・ON実78453入力の診断では、両armの48層×選択16行は全byte一致、
有限値、選択行の最大絶対値41.56731414794922、FP16最大有限値65504超は0要素だった。
`qsa-device-batch-model-comparison-v1.json`に層別値とcapture hashを残した。
選択行のレンジ観測であり、FP16へ変換した品質や全256K行のレンジは検証していない。

## 依頼に対する再確認（2026-10-15 13:59 UTC）

Mainがexclusive measurement lock下で実GGUF二shardのheader hash、file size/mtime、
pack/native index hashを再確認し、実headerから300 native tensorと全48層のbytesを再集計した。
最大layer47と上の1440161792 Bが一致した。新しいGPU推論・大量確保は行っていない。
既存GPU allocation receiptの正常閉鎖・新規faultなしと参照hash chainも確認した。
現在のproduction planner headerは既合格48ケースのsource pinと一致し、16条件の
context上限・partial hidden・staging・次page/次chunk超過をMainも独立再計算して一致した。

GPU総量は十進で10.6661 GB、最大レイヤーは1.4402 GB。
本文と表は一貫してGiB（2^30 B）を使う。
重みとworkspaceだけの和は4Kで2.4836 GiB、8Kで3.4941 GiB。
全12層の256K KVを残す主表と、処理中の一層だけKV/indexerを置く別案は混同しない。

読み取り専用Luna R711は容量式・partial staging・reserve感度を独立に再計算し、
現在の主表に明白な算術欠陥を見つけなかった。R712はsource/実測の範囲を監査した。
次層lookahead norm 40960 Bは小owner予算から明示確保する必要がある。
全48層normを恒久保持する場合の追加3850240 Bとnon-layer HC出力tensor13148160 Bを
両方16 MiBに押し込むことはできない（合計16998400 B）。
主表を実装するには、一層先normを個別に読む、最終出力tensorは出力段階で読む等の
生存期間を定めて予算内に収める必要がある。
PLEのcanonical借用pointerもfull arenaを解放する前にprivate bindingへ切り替える。
別提案の常駐PLE/HC poolを採用するときは、その実owner bytesで再計算し、
この理想配置の6.8244 GiBをそのまま使わない。

レビュー原本は`research-capacity-arithmetic-round711-v1.txt`と
`research-capacity-workspace-round712-v1.txt`。容量試算の確認であり、
現在の実装でこの配置が動作・同時確保できるという合格には変更しない。
