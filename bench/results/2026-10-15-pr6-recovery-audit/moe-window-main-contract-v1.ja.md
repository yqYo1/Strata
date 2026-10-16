# MoE窓のMain実装契約

2026-10-17。対象は `deea5cf1` の推論ソース。R966の19–45行だけを、R968の訂正を適用して
参照した。XeStrataの実装は使用していない。これは実装用の契約とホスト検証モデルであり、
C++エンジンへの統合・コンパイル・GPU実行・速度採用は未了。

## 選択と処理順

既存のnative16 BF16、compact MoE、MMQ=0、単一in-order queue、alias=1、
no-prefill-borrow、8192行のoperator chunkを前提にする。GDN/QSA/PLEは元の8K単位で
昇順処理する。同じレイヤーの連続する二つのpieceが両方GPU残差bankにある時だけ、
MoEを最大16384行の窓にまとめる。GPU境界をまたぐ組、RAM側、単独pieceは元の8K処理。
短い入力のdecode配置経由prefillと、他の設定の経路は選択を変えない。

窓内の順序は次で固定する。

1. 最初のpieceの入力を消費し、PLE/HC-attention/GDNまたはQSA/HC書き戻しを行う。
   出力は既存GPU残差bankの同じ範囲へ保存する。次のpieceも同じ順序で処理する。
   この時点のbankはpost-attentionであり、外部のfinal callbackを呼ばない。
2. 両pieceのattention完了後にFFNのHC入力を準備する。HC norm/mixはbankのGPU viewを
   直接読み、8Kの既存xn16/gated/lo等を使う。mixed_h、router ID/weight、shared出力、
   scalar gate、FP32 injectionだけを窓用viewに保存する。CPUで推論の算術は行わない。
3. IDのD2Hは窓ごとに一度checked drainし、host上で元と同じexpert順の索引を作る。
   GPUのID領域は、このreadback完了後にslot map領域へ役割を変える。
4. 各expertを一度だけFP16へ展開する。neは窓長まで許し、GEMM jobは最大8192行に分割する。
   gather/GU/hidden scratchは8K容量のまま使い、Downだけを全窓のBF16 slot-order planeへ書く。
   ne==16384は8192行の二つのjobとなり、展開を繰り返さない。ne==Wを拒否しない。
5. 最後のexpert入力読み出しより後に、同じqueue上で全窓のcombineを投入する。top-k slot順の
   FP32 fma、shared加算が最後、BF16への丸め位置を維持する。GEMMの行数・tile選択変更による
   差は別途数値評価する。
6. combineの後、両pieceのHC FFN書き戻しをGPU残差bankへ行う。両方の書き戻しをchecked drain
   するまで、FP32 injectionとBOは生きている。途中でMTPを呼ばない。
7. ここで全MoE region readerとBO/mixed viewを退役する。中間レイヤーでは各8Kの完了を記録する。
   最終レイヤーではbankの各8K sliceを順にFP32 m.Rへ展開し、既存callbackへ渡す。
   MTPはこの時点でm.regionを再利用できる。二番目の最終残差もbankにあり、m.Rの上書きで失われない。

FP32 `m.R` と旧gated allocationへ16Kの生きた残差を置かない。PLEはphase1でのみm.R/embを使い、
MTP/head/NLL callbackはphase7でのみm.Rを使う。native/canonical weight overlayの寿命、
outer checked lease、同じqueueと起動後不変の環境設定は既存どおり維持する。

## 窓のviewと所有者

regionとは別の既存m.emb（8K×2560×4=83,886,080 B）を、phase2–4の窓mixed FP16と
phase5–6の窓BO BF16に時間で分ける。8Kのgated allocationを二分したR16/gated_b16、
mixed_bf/mixed_h、xn16、lo等はoperator用として維持する。PLEのemb aliasと同時使用しない。

`moe_window_contract_v1.py::memory_plan` は既存m.region内へ独立したtyped viewを配置する。
legacy m.* pointerを置換せず、窓helperへviewを引数で渡す。下記は256 Bの各guardを含む。

| region内のview | 最大payload B |
| --- | ---: |
| BF16 Down、16K×10×2560 | 838,860,800 |
| BF16 shared、16K×2560 | 83,886,080 |
| FP32 route weight / I32 ID→slot | 各655,360 |
| FP32 scalar shared gate / injection | 65,536 / 262,144 |
| 8K jobのFP16 gather / BF16 GU / FP16 hidden | 41,943,040 / 20,971,520 / 10,485,760 |
| 8K jobのI32 source list | 32,768 |

10 viewの予約は **997,820,928 B**、既存GDN region 1,180,698,624 Bとの差は182,877,696 B。
R966のWサイズgather/GU/hidden案から、jobを8Kに分けて減らした。region自体を拡張する算術は不要。
phase2用のrouter logits/shared gate/shared up/shared hiddenは48,235,520 Bで、
まだDownを生成していないDown領域内に置く。この役割を終えてからexpert出力へ切り替える。
実装時には実際のRuntime登録allocation、context/device、全spanとalignmentを投入前に検証する。
この算術は物理fit、private workspaceやGPU同時実行の証明ではない。

GPU source listは常に先頭へコピーする。hostの全窓source listはW×10 int32を保持し、
H2D元は `host_src + global_expert_offset + job_offset`、転送先は `device_src[0]` とする。
一つのhostベクトルをjobごとに上書きしてDMAに渡さない。host ID/slot/sourceの最大payloadは
各655,360 Bで、counts/offset/orderを加えてRAM入場に計上する。全ownerは窓drainまで生かす。
gatherの既存helperは完全な入力行数を引数に持たないので、host index範囲と入力bank/embの
全W行extentを別途検証する。`src_dev + o0` の旧式をそのまま拡張しない。

## 記録・失敗・検証範囲

8K baselineと窓版は同じbinaryの不変設定で選び、変更途中に環境変数を切り替えない。
window数、ne、job数、展開回数、GEMM/backend tile、実転送bytes、CPU投入/group/waitと
GPU完了を記録する。既存ヒストグラムにはチャンク間expert ID対応がないため、正確な窓neの
観測には別の計数が必要。synthetic routingでその代用をしない。

中断・enqueue失敗後はchecked retirementまでownerを保持する。drain失敗ならownerを生かして
fatal終了し、部分処理したsessionを有効な復旧点として公開しない。状態tensorのrollbackとは区別する。

窓のcallback順を正しても、中間prefixの全層stateが一致するとは限らない。既存のlayer-major
checkpoint問題は `layer-major-checkpoint-coherence-v1/main-review.ja.md` に記録済み。
serverの周期checkpointを新窓経路へ接続する前に、実際のcaller要求を明示し、全層・MTPの同じprefixの
stateを持つproducerとRESTOREを確認する。古い未検証patchを取り込んだとは扱わない。
最終run境界だけを完全なstateと扱う検証設定は、周期checkpoint機能の代替・削除ではない。

ホストoracleはrow/position、GPU/RAM境界、phase alias、src bounds、callback順の設計検査に限る。
262144/262143行のケースもallocation/GDN/QSA実行・最後のprompt token消費・RESTOREの証明ではない。
実GPUではbiased ne=W、端数、nonzero pos0、PLE、MTP fast/fallback、fault/cancel、丸め/route/state、
full physical contextとRESTOREを検証し、その後に公式bench.shの32K以上・thinking有効で反復比較する。
