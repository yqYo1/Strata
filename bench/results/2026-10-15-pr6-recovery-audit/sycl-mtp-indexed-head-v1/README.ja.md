# SYCL MTP indexed Q6_K draft head proposal v1

未適用 / 未build / 未test / 未GPU。新しいこのdirectoryだけにsourceを保存した。root product/client/b1 binary、実行中IQ ON exec10041、他proposalは変更していない。Mainが測定終了後にCPU/build/IR/GPU/モデル検証を直列管理する。authorはtest/build/py_compile/GPU/native/process/Git/cleanupを実行していない。

## 方式とopt-in

exact STRATA_SYCL_MTP_INDEXED_HEAD=1だけをSYCLのbind/preflightで読み、defaultOFF。process環境を書き換えない。NativeHeadの既存device-resident full weightsを借り、既存draft_vocab.binのdvocab順でMMVQの行addressだけを選ぶ。Q6_K=type14、input幅256の倍数、実head row_bytes=210*(N/256)、full row数とweight_bytesが一致する場合だけ。N2560ならrow2100bytes。提示されたstartup106299-row subsetならdhead requested bytesは106299*2100=223227900bytes(約212.9MiB)となるが、このproposalは実head typeの新測定やcudaMemGetInfo差を取得していない。実native typeが14であることはMainのgate。type非14/非対応幅やsubset無しはlegacy。

dvocabの同じsubset bytesは引き続きdeviceに保持し、hostにも不変ID copyを所有する。dhead_だけを未確保にし、元full head、logits、quantization、subset順、token remap、sampler/coupled policyは変えない。追加精度変更やnative data layout変更ではない。非単調/重複IDはそのまま保持する。coupled逆引きも旧実装と同じ最初のsubset indexを使う。

## MMVQとspan契約

native_q6_k_mmvq_indexedはfull_rowsとsubset_rowsを別引数に持つSYCL専用entry。既存Q6KTraitsとgeneric dispatch/launchを共有し、template Indexed=falseの通常pathにはmap read/branchを生成しない設計。Indexed=trueだけがweight rowをids[subset_row]へ置換する。全multiply/accumulate、k順、cross-warp partial、XOR reduction、exact/nonexactのROWS/NW選択は同じsourceを通る。通常kernelのsymbol/template引数は変わるため、同じdevice code生成までsourceだけで証明したとは扱わず、Mainはfalse specializationのIRも確認する。

新entryはenqueue only。明示in-order queueが必須で、内部waitや新allocationはない。full weights/map/Q8_1 x/subset yのbyte geometry、alignment、overflow、writable no-overlapは既存validate_spansに従う。加えて既存compat cudaMalloc allocation registryへread-only span照合を小さな関数として公開した。これはregistry lookupだけで、既存compat APIやTLS error処理を変えない。map/activation/outputを含めregistered device spanを要求し、full matrix全spanを検査する。registryを持たない任意sycl::malloc_device pointerはこのentryでは受け付けない。既存MMVQ通常entryのspan契約は変更しない。

device IDが負/範囲外ならその行のweight readを飛ばしzero outputとなる防御を持つ。これは不正production mapの許可ではない。bindは全host IDsをhelperで先に範囲検査し、初めてmap allocation/uploadを行う。不正file count/shape/IDをGPU mapとして公開しない。CPU helperは実productionでshape/preflight bytes/host-ID validationに使用する。

## MTP lifecycle追跡

旧sub=dhead_!=nullptrに依存した箇所はhas_draft_subset()へ集約した。bindの既存subset有無、bind_bytes、setup_coupledのnvとdinv、record_forwardのMMVQ/nv、coupled sampleへのdvocab/dinv、greedy sample/probability後map_idsが同じsubsetを使う。KV-onlyのif(!full)returnより後にだけ新head entryを呼ぶ。prompt K/V pathを変更しない。

新SYCL preflight overloadは実head typeを受け、generate.cppのauto-cache reservationがeligible時だけdhead bytesを除外する。元の2引数bind_bytes APIは残し、実type不明の旧callerは保守的にlegacy head分も予約する。CUDA/HIPのpreprocessed call/bodyは旧通り。head_logitsは従来full-vocab allocationのまま。vram_にはnew modeでmap bytesだけを追加するが、従来通りこのcounterがMTP全allocationを網羅するとは主張しない。

indexed bind完了後のsame-object/same immutable weights/WT/window pointer再bindはその状態を保持し、fileを再読してgraphのmapを差し替えない。初回subset無しlegacy fallbackという判断もeligible opt-in再bindでは保持する。異なるhead/WT/windowへの再bindや途中までindexed bindされたobjectの再利用は、graph/ownerを退役してdrafterを作り直すよう明示拒否する。borrowed NativeHead full weightsはMTP graphsより長生きしなければならない。実generateではMtpDrafterがNativeHeadより先に宣言されるため、そのままではNativeHead destructorが先となる点をsource reviewで発見した。SYCLのNativeHead直後にscope leaseを置き、index modeだけchecked stream drain/full decode graph destroyを先行してからNativeHeadを破棄する。通常OFFではこのleaseはAPI/waitを行わない。退役済みindex modeは再bind/full-forwardを拒否する。ほかのcallerも新public retire_indexed_head_readersをNativeHead破棄前に呼ぶ必要がある。

map upload、new mode coupled ring memset/inverse upload/setup syncのAPI failureはその場でcode/detailを保持したexpert_failstopへ進み、unknown DMA ownersやhost upload vectorをunwind/freeしない。新modeのborrowed-head scope leaseとdestructorはcs drainとgraph destroy戻り値をchecked化し、失敗ならownersを保持して_Exit。dvocabは全graph readerのchecked retirement後だけfreeする。copy/free/graph failuresを成功へ変えたりfallbackへ再投入したりしない。通常modeの旧upload/error挙動をまとめて改修するscopeではない。

## 再構成、fixture、Main gate

source.diffとreconstruction.jsonに全変更fileの厳密base/output SHA256を保存した。大きなgenerate.cpp完全copyはpatchで再構成できるため省略。それ以外のchanged files/new helpersはfiles/にある。MtpDrafter class layoutが変わるため、実classを所有するgenerate TUも同じoverlayで必ずrebuildする。core mtp、kernel mmvq、SYCL compat registry function、関連archives/link mapと実source resolverをMainがpinし直す。既存b1 objectへ新layoutだけを混ぜない。CMake全面変更は不要。

cpu_contract_fixture.cppは実helperのshape/bytes/type gate/ID範囲/overflowを検査するsource。subset1/17/tail、非単調/duplicate/first-last、unsupported type、bad shape、negative/too-large IDsを含む。GPU fixtureはEXACT0|1を別owned processで実行する設計で、各process内ではexplicit exact test parameterをruntime前に一度だけ設定する。process env切替はない。各ncols1..8×7maps=56casesで旧gather_rows+native_mmvqとindexed graphの全output/guardsをbitwise照合し、入力weights/map/float/Q8 payloadを検査する。subset1、17、33 tail、37、51 duplicates、first/last/nonmonotonicを含む。indexed graph/map/weightsをgraph retirementまで保持する。shape/misalignment/overlap/short registered full-weight/map/x/y span/unregistered host/null streamのnegativeもある。device invalid IDはzero防御だけを別検査し、legacy gatherへ不正IDを渡さない。

これらは実行していない。GPU fixtureはhelper/entry parityとcapture lifetimeの設計であり、実MTP bind/再bind、coupled logits/token remap、model selected logits/IDs、cancel/restore、actual head typeやrequested bytesのnative gateを代替しない。Mainはassert有効のCPU/sanitizer、未対応fallback、厳密upload failure/owner保持、IR、native/coupled/actual same-work correctnessを順に確認する。必要な個別標本・finite予算・所有closure/fresh faultsを保存する。性能/品質/採用、物理262144/RESTOREの主張はない。logical requested bytesの削減をphysical VRAM差や速度へ読み替えない。
