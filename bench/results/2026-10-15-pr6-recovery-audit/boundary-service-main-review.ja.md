# Boundary / CPU-format diagnostics integration

2026-10-15 JST。Mainがdefault OFFの中間residual境界診断と、既存CPU serviceのformat別集約を
実製品sourceへ差分適用した。`boundary-service-build-v1.json`が実buildと個別CPU実行の証跡。
新binaryはb6838d286a6ec00b0d92c7ce8f8480c81511c4ab823e400273c52b2585c55aa6。
GPU/modelでの新診断はこのbuild証跡の対象外。旧24576-row数値FAILは変更しない。

CPUでは既存decode joined fixture、追加format fixture、追加boundary fixtureが通過した。
追加二fixtureはAddressSanitizer/UBSanでも通過した。boundaryはFP32 signed zero/subnormal/
infinity/NaN payloadを含むraw bytes、二dtypeの48層coverage、CSV/shape/context上限、
exclusive出力、欠落sequence/copy失敗、破損storeのdifferent判定を確認する。
formatは正常集約/zero-work/16形式上限と不正workers/layers/代表layer/count/overflowを確認。
CPU helper試験はGPU copy、device conversion、kernel数値、異常時GPU寿命を証明しない。

境界診断は19行×48層の計算直後rawを37,355,520 B保存し、run内source→store→next-inputを
actual memcmpする。hashはlocatorに限定する。追加D2H/drainは実行順を変え、raceを隠す可能性がある。
固定4096と自動prefixを同じ診断binaryで比較し、最初の異なる層/rowを選定する。
診断で差が消えても元経路の解決とは扱わない。raw consumerはこの有限比較で、次reviewは比較終了時。

CPU-format集約は既存完了済みthread countersをissuerがdispatch前後に集約し、追加clock/waitや
worker writerを導入しない。形式別総量は既存issuer/worker×role総量と完全一致を要求する。
形式とntのクロス集計は追加していないため、形式別costと既存request-wide nt分布から
形式別nt時間まで推定しない。次のactual serveは新binary/recipe/出力rootで別測定する。

## R582/583 Main review

R582のIQ3_S nt=1二row pipelineは未測定のILP仮説。既存GGML single-row vec_dotもAVX2 vector命令を
使うため、「scalar single-token kernel」をscalar命令による遅さの根拠にしない。codebook gatherも
Zen3で優位と未確認。まずformat別serviceで対象を絞る。現在の集約だけでformat×nt時間は出ない。
外部リンクのmoving mainや別format/CPU測定を、このPCの改善証明に使わない。

R583はhalf/BF16 GEMM入力と自前SYCL QSAの実sourceを区別し、VRAM依存variantを未確認としている。
内部memory managerの存在は本件の原因証明ではない。報告のLevel Zero tracing有効という文言は
今回のrecorded environmentとは区別する（parameter validationとwarning loggingは有効）。
SYCL_UR_TRACE等の環境変数を初期化後に変更して短区間だけtraceできるとは仮定しない。
必要なら初期化前設定の別processか、runtimeに実在する動的APIの検証が必要。
MKL verbose等はまず境界比較で差を特定した後の候補に留め、今回追加しない。

正式性能は旧`upload-fix-suite-final.ja.md`から更新していない。prefill1000/decode70、
physical262144、RESTORE、coupled/cancel/全Tの条件を完了扱いにしない。
成果はfeature/syclへ保存する。defaultへの機能整合は別途未完了で、古いengine rootへの
置換で新しいdefaultのserver/CLI機能を失わない。

## R584とcache配分のMain追加確認

R584のprofile形状と頻度証拠の区別、128が総slot数である点はsourceと整合する。
ただし「roughly two or three」は実装を正確に表さない。`expert_cache.cpp::layer_slot_range`は
`q=slots/layers`として最後の層だけに全余りを渡す。128/48では最初47層が2、最後が34である。
均等配分でhit率/速度が上がる証明はまだないが、native offsetと同じpartitionを使う
opt-in候補を独立実装担当へ依頼した。既定配分のままの測定結果を変更しない。

R584の「one-byte-per-route owner/ID stream」は512 expert IDだけでも9bit必要で成立しない。
将来replay用streamが必要なら明示したbit layout・layer/window境界を含むbyte算術を行う。
192/384KiB追加counter案は現在のobserver96KiB state/combined256KiB契約にはそのまま入らず、
別owner/予算/入場条件のreviewが必要。既存usageはfloatでdecayされ、adapt-swaps0では
allocateされないため、そのままstatic decodeのexact histogramに流用しない。
追加測定の許可はユーザーの継続最適化依頼に含まれる。研究報告のpermission文言を
新しい承認要件として扱わない。

実formatは旧ledgerから7組確認でき、GU type18が17層、22が20層、21が10層、23が1層。
これは層数でありtime割合やroute頻度ではない。IQ3_S(type21)候補だけでCPU GU全体を
改善できると見積もらず、実format別serviceを次に取得する。
旧service quant role1/4はpool内のhidden量子化であり、pool前の入力activation量子化を
含む全量子化処理の合計ではない。約0.3%は計測したpool CPU service内の比率に限定する。
