# v2 kernel設計監査 — 実機結果ではない

v2のkernel/source.diffはv1とbyte-identical。probeの実行方式だけを修正した。
環境はexec時固定、単一arm/capture、offline比較であり、v1の同一process env切替は実行しない。
既存STRATA_TEST_PROMPT_MATRIX test内部にもenv切替があるためv2の実行計画から除外する。
そのsource上のoracle許容差を参照することと、そのtestを起動することは別である。

## 由来と差分の境界

本候補はこのforkの`src/kernels/sycl/prompt_attention.cpp`から派生した。
NikoのCUDA generator、XeStrataのengineや別fork実装をimportしていない。
基準cppのSHA256は`97b1b9e4020b532bfabd5db2245224cfda52bbc793e5891f0e8bbbcb374763e4`。
R600/R605、research-index、Main profile、matrix-direct-probe-v1、現行API/runtime/
launch/tests/CMakeを参照した。入力の実hashはinput-sha256.jsonを正とする。

configured source resolverは現CMakeのSYCL backendから`strata_kernels`へ明示された
`src/kernels/sycl/prompt_attention.cpp`で、公開入口は`qsa_prompt_attn_batch`。
そこから`mode=p.k_q4?2:p.k_q?(p.v_q4?3:1):0`、既存matrix_available/span検査、
`dispatch<mode>`の順に進む。新分岐はdispatch<1>のDirect block内だけにある。
Mainは適用後の実build cache/compile command/link/binaryを照合すること。
この読み取りだけで現在ロードされたbinaryと提案の一致を主張しない。

既存`launch<>`本文はコピー元のまま残し、既存dispatchに新分岐を挿入した。
既存mode0 Native KV選択、mode2 Q4 OFF、mode3、fallback false、外部signature、
matrix-combination/device gate、span validation、finish/error/lifetime契約は変更しない。
新launcherはDirect専用のMode1、KG=VG=4、C=32/64のみ。
旧launcherがバイナリレベルでも同一になるかは、再コンパイル後のMain検証事項である。

## 演算順とQ scale

現行1 WGは1 query × 1 KV headの12 query headsをlocal16 rowsへ置く。
8 subgroupのうちQKは4 subgroup、PVは8 subgroupである。
R605の該当段落の「従って2 KV heads×12 query headsを1つのWG」は誤り。

候補のgroup indexは`tile=i%2, kv=(i/2)%2, batch=i/4`。
各query/KVに2 WG、tile0はglobal relative head0..7、tile1は8..11を担当する。
`q_group = q + (batch*n_head + kv*12)*256`を両WGが読み、tid stride64で
全3072値のmax absをreduceして、従来と同じfrexp/clamp/ldexpからqup/qdownを作る。
有限入力のmaxは加算と異なり順序依存の丸めを導入しないと期待できるが、
実機denormal/コンパイラ実装までのbitwise証明にはしない。NaN/Infの同等性は対象外。
担当tileだけでqmを決める変更は禁止。Q pad4 rowsは明示0、実queryを読まない。

C32ではQKのcell bandsをSG0/1へ割当て、SG2/3はQKを実行しない。
SG0..3はすべてPVの4 output bandsを担当し、各SGに4個の8×16 accumulatorを持つ。
各実head/cellで、K scale group0..3、各groupのdim slice昇順16、hi MAD→lo MAD、
FP32 `scores += part*ks*qdown`の順を保つ。
softmaxの32-cell max/exp/l/sum=fma、V scaleの32-cell max factor、P hi/lo分解、
PVのk=0,16のhi→lo MAD、acc*=alpha、acc+=part*down、最終sum除算も同じ。
headを別WGに分けてもhead間の加算・依存は存在しない。
したがって有限入力でC32は旧Directとbitwiseを期待し、fixtureでは必須にした。
未実行であり、異なるWG/kernel instantiationやcompiler allocationが期待を崩す余地がある。
bitwise不一致を自動で許容差へ緩めて合格にしてはいけない。

C64のQKはSG0..3が各16-cell bandを担当する。
K dim/scale group/hi-lo MADの順は維持するが、softmaxのl、オンラインsum/alpha、
V factor、P rounding、PV k=0,16,32,48とrescaleのchunk境界が変わる。
旧Direct/C32とのbitwiseは要求しない。全出力と独立FP64 oracle、旧Directとの双方に
事前固定`1e-5*(1+abs(reference))`を初期component gateとして置く。
許容差は既存STRATA_TEST_PROMPT_MATRIXに由来し、新候補の品質保証ではない。
高dynamic-range入力で既存baseline自体がoracle gateを外れた場合も失敗を保持し、
候補だけの退行と断定せず切り分ける。通すためにoracleや入力を変更しない。

## Collectiveとbarrier

subgroup size16、WG64である。QK条件`subgroup<C/16`はsubgroup-uniform。
C32のSG2/3も全WG barrierへ到達し、PVと初期acc fillは全4 SGで実行する。
pad headの有効性をmatrix collective自体のlane条件にしない。
scalar softmaxはtid<8、実head分だけ旧計算を行い、pad scores/P/alphaは0、sumは初期0。
最終joint_matrix_applyも全laneで呼び、element callbackのglobal storeだけをr<live_headsで制限。

barrierはloop前に1回、各chunkに7回あり、いずれも全WG-uniformのloop/body内。
(1)rowsとzero scores、(2)Kとks、(3)QK出力、(4)softmax/V/vs、
(5)factors、(6)phi/plo、(7)PVと次chunkのreuse、の順でpublish/reuseを保護する。
loop回数は同じqueryのstepsから決まり、work-item固有early returnはない。
WG間で同じ入力を再読するが、異なるhead output範囲にのみ書くため相互barrierは不要。
入力が処理中に書換わらないこと、in-order queueとowner寿命は既存caller契約のまま。

## local_accessorのpacked論理byte算術

padding/alignment、compiler spill、driver予約、実allocationは含まない。
本候補はDirect dummy `final[1]`も宣言しないため、R605の53,876 Bより4 B少ない。

| accessor | 式 | C32 B | C64 B |
| --- | ---: | ---: | ---: |
| qhi + qlo | 2*8*256*2 | 8192 | 8192 |
| kv | C*256*2 | 16384 | 32768 |
| phi + plo | 2*4*8*C*2 | 4096 | 8192 |
| scores | 8*C*4 | 1024 | 2048 |
| ks + vs | 2*C*4*4 | 1024 | 2048 |
| maxima + sums + alpha | 3*8*4 | 96 | 96 |
| factors | 4*4 | 16 | 16 |
| rows | C*8 | 256 | 512 |
| 合計 | 8304+712*C | **31088** | **53872** |

現行Mode1 Directのdeclared sumは44,500 B、Indirectは60,880 B。
既存matrix_availableはMode1で60,880 BとmaxWG>=128を要求するため、閾値を維持する。
新候補の必要WG64/packed bytesはその下にあるが、compiler/runtimeのkernel固有制限を
測っていない。device infoのlocal_mem_sizeでlaunch成功・spill・occupancyは証明できない。
Unitrace static SLM4096 Bをdynamic accessor totalやoccupancyへ読み替えない。
実allocationの取得方法・kernel固有work-group限界・spillの可視性はMainの実機確認事項。

各local matrixロードの最大indexは以下でextent未満。
Q hi/lo: `(7*256)+(240)+(15)=2047 < 8*256`。
QK B: `(255*C)+(C-16)+15=256*C-1 < 256*C`。
scores: `7*C+(C-1)=8*C-1`。
ks/vs: `(C-1)*4+3=4*C-1`。
PV A: `(3*8+7)*C+(C-1)=32*C-1 < 4*8*C`。
PV B: `(C-1)*256+240+15=256*C-1`。
rowsは0..C-1、scalar stateは0..7、factorsは0..3のみ。
上記はm8n16k16 row-majorの行列要素域算術であり、実機memory checkerのPASSではない。

## External index / INT_MAX / size_t / tails

外部query/output head strideは`n_head=24, kv*HEADS=kv*12`を維持。
local row8をexternal strideへ使わない。
出力は`(batch*24+kv*12+tile*8+r)*256+(sg*4+j)*16+c`。
rはtile0で0..7、tile1で0..3、sg0..3/j0..3/c0..15。
これは各queryの24*256要素を重複なく覆う。fixtureのhost ownership列挙はこの式を
検証する案で、GPUの同値重複storeを実測して検出するinstrumentationではない。
output sentinel/全要素oracleは穴・誤stride・余分な範囲を検出する。

public gateの`nq<=65535, n_head_kv=2, cap<=INT_MAX`を継承する。
64-bit size_t環境で最大gridは`65535*2*2*64=33553920` work-items。
query/outputは最大`65535*24*256=402647040` floats、1,610,588,160 B。
idsは最大`65535*2147483647=140735340806145` entries、562,941,363,224,580 B。
これは整数上界であり、その巨大なallocationを許可/実行する計画ではない。
現APIのchecked_count/spanはこれらの積を通す。実配列の確保量までは受取らない。
対象PC/compilerの64-bit size_tと`-fsycl-id-queries-range=size_t`をMainのbuildで確認する。

旧loop `int c0 += 32`にはcount=INT_MAX付近で最終加算のsigned overflow余地がある。
旧本文を保存する今回の境界では修正しない。新候補だけ`int64_t c0`とし、
`c0+tid`、末尾increment、size_t ids offsetがINT_MAXでoverflowしないようにした。
浮動演算順には影響しない。INT_MAX幅を小fixtureで実行したとの主張はしない。
count<0/count>capは0として旧規則どおり全出力0。
partial chunkでは`c0+tid<count`で選択ID loadをguardし、tail row=-1。
negative cellはpage tableを読まず、page<0のrowはK/V/scale loadを読まず0。
valid cell upper bound/page table長/physical pool容量は引数に無くcallerが保証する。

cap262144とcell262143はsigned32内。page_size4ならlogical page65535/offset3。
fixtureは65536-entry tableのentry65535をphysical page2へmapし、小poolで実readする。
cap262144を実ids strideに使うnq3ケースも別に用意した。
C32最終chunk c0=262112、C64は262080、最後の有効slotはともに262143。
fixtureのactive width65はこの最終chunk走査を実行しない。上記full-width境界は算術監査。
最大capのselectionを262144回処理した、full physical256Kを消費した、RESTOREしたという
証明にはならない。active width=262144/full physical入力は別のMain-owned長文検証が必要。

有効page/offsetを巨大な任意値にした場合、APIがpool row数を検査しない点も継承する。
理論的row計算の範囲と、`row*HD`/`row*4`が実poolを指す保証は別。
callerの実allocation/ID領域が必要であり、新候補の64-bit indexだけで解決しない。
