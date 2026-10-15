# 追加headroomのsource確認と重複draft head候補

2026-10-15。IQ ONの同じexec10041を直接pollしてrunningを確認している。root src/includeとb1 nativeは
変更せず、テスト・build・別GPU job・cleanupを重ねない。新たな性能合格や採用判断はまだない。

R657はMTPのprompt K/V-only readerを調べ、dense_/arena_を丸ごと外せないと確認した。
head_logits_とoptional dhead_/dvocab_はdecode側の出力headで、prompt K/V-only returnより後の
readerである。ただしsource/accountingだけで実free容量を決められず、subset IDsを元ファイルから
再読するなら同一性/owner/queueの検証も必要になる。この棚卸しを実detachの合格とは扱わない。

Mainは閉じたfull OFFの同じhash済みnative logから、startupの
`draft head over 106299 tokens (212.9 MiB)`を確認した。これは初期subset headのrequested重みサイズを
丸めて表示した行で、新しいfree差の測定ではない。現在bindは、すでにdevice-residentな
NativeHead.weights()からgather_rowsで別dhead_を作る。この複製を作らず、MMVQでdvocabの行を
直接参照する方策を次のsource実装候補に選んだ。数学/量子化/還元順序を変えず、同じsubset順と
token remapを維持できることを実検証する。type14/Q6_Kに限定し、実head type条件を確認して
unsupported typeは従来方式に戻す。head本体を外す、subsetを小さくする、精度を落とす変更ではない。

Solには新sycl-mtp-indexed-head-v1/だけへのdefaultOFF/SYCL opt-in proposalを依頼した。
MainがCPU fixture、isolated build、GPU全配列bitwise比較とspan/shape/host-ID拒否、実MTP/graph/
coupled/RESTORE/長文品質・clean性能を後で直列管理する。今の測定中にcodeを適用しない。
既存のdhead_!=nullptr判定を全て追い、indexed subsetのNv/dinv/remap/rebind/accountingを維持する
必要がある。新kernelはfull matrixとsubset行数を分け、元allocationとmappingの寿命/範囲を守る。
head重複を省いたrequested Bを、実free差や速度改善と先取りしない。現R_GPU4096固定では、追加
空きがそのままprefillのprefixを増やすわけでもない。これは次の容量・速度実測を可能にする候補。

## R658をMainが訂正依頼した理由

v1のQSA sel_ids幅32771と512MiB計算は現sourceに一致しなかった。Mainはqsa.hppの
min(n_kv,idx_top_k+idx_block-1)、real shapes2048/4から2051と再確認し、公開fullOFFの限定keyから
STRATA_PREFILL_RING=8も確認した。またlayer_major.incは全512expertをupload/drain後にresident公開し、
run_implはlayer_major_activeならstream_allを使わない。普通のstreamed ringと今回のrouteが混ざっていた。
FINAL後にv1を保存して別v2を依頼し、訂正版をレビューした。調査途中への介入ではない。

訂正後はC4096でsel_ids33603584 B、stage_dev8slotsで21299200 requested Bとなる。
このlayer-majorではresident expertはcache pointerを使い、fallbackのstage bufferも読まない。
少量の未使用領域はあるが、half-GiB以上の安全なdefer/alias候補は現sourceから確立できなかった。
shared QSA/GDN/MoE regionは既にmax-sizedの同一領域を再利用しており、三つを足して削減余地と
してはならない。旧v1の512MiB/384→96 ringを今回の容量根拠から外し、原bytes/hashは保存する。

新しいLuna R659/R660は、作成中Sol proposalを読まず、現rootのsubset状態契約とQ6_K kernelの
bitwise mapping条件を別々に調べる。既存report/indexを共有し、双方のFINALまで介入しない。
Mainは同じGPU suiteと実receiptを監視する。目的1000 PP/70 TG、PR6性能回復、品質、
物理262144/RESTOREとupstream/defaultの残作業は引き続き未完了である。

## R659/R660の独立source契約

両FINALを受領してMainがレビューした。R659はbind_bytes/setup_coupled二箇所/bind/record_forwardの
subset意味とcopied storageの判定を分離する必要、再bind時のcaptured pointer、NativeHeadの借用寿命を
確認した。新indexed modeでもNv、dvocab/dinv、sample/remapとaccountingを維持する。提案はまだ
作成中なので、この調査を新codeのレビュー済み証拠にはしない。R659の非Q6拒否という推奨に対し、
Mainが実装へ要求した仕様は非Q6/真正no-subsetなら従来modeへの明示fallbackである。新経路が実際に
有効だったことは将来の実route証拠で区別する。不正なIDを新経路から読むことは認めない。

R660は現在のQ6_K型14がgeneric traitsであり、実full MTPはT1、N2560ではROWS1/NW4/subgroup32と
確認した。indexed化はcompact出力幅を保ち、元重みのrow addressだけを変える。量子化、dot、warp
partial merge、XOR還元を変えない。ncols>1のmulti_exact別layoutも同じmode同士で全配列比較する。
負IDは旧gatherのzero-fill仕様だが、新production bindでは範囲拒否を先に行う。正の範囲外IDを
旧gatherの未定義readに合わせることはしない。full weightとselected ID、activation/outputの各span、
no-overlap、明示queue/enqueue-only/graph寿命、capture/replayの実検証はMainの後続gateである。
両reportは作成中proposalを読まず現rootだけを確認している。Solへの途中介入はしていない。

R657原reportにEOF空行一件のgit whitespace warningがある。原受領bytes/hashを維持するため保存し、
そのreport以外の変更のdiff --checkと実test/build資格を区別する。
