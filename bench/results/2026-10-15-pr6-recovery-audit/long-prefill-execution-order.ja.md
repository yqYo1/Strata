# 長文prefillの現在の処理順と次の優先順位

2026-10-15。Mainが現rootの実装を読み、独立したLuna R667/R668と照合した。
sourceの説明であり、新しいprofile/性能/品質/物理256K合格ではない。
直前のgoal turnは同じexec10041のlive pollを伴うverified waitだった。

## 作業方針

長文prefillを繰り返し実測して600 tok/sを超えるまで、以後の新規最適化はprefillに集中する。
目標1000 tok/sは維持する。短い入力をdecode用配置で処理する経路は残す。
短入力の経路選択閾値は後回しにするが、長文のchunkサイズの調整は後回しにしない。
decodeの新規高速化、CPU profile/cacheの新しい速度探索はこの区切りでは進めない。
MTP head共有案とNLL案は未適用sourceとして保存し、prefill容量/品質評価に必要な範囲で使う。
既に走っているIQ比較の既得結果と元のFAIL/PASSを保存する。短い入力から採用を決めない。

上流確認はproject-only strata-upstream-syncに従い、おおむね48時間ごとに行う。
確認・上流単体の評価・機能移植・推論部品採否は別々に記録する。
直近確認として参照できた記録は2026-10-14T01:12:13Zのupstream-observed.jsonであり、
今回そのファイルを読んだことを新しいremote確認とは数えない。そこでは296b59d2を観測し、
評価/全機能取り込みは未完だった。次の目安は2026-10-16T01:12Z。常駐タイマーは作らない。
defaultへの反映と既存の上流機能移植残作業も完了扱いにしない。

## 三つの入口

1. serveの短い区間はgenerate.cppのwindows_ok/read_windowsを通り、Verifierのwindowを使う。
   decode用の重み/cacheとCPU/GPU expert処理で入力を消費し、head samplingは止めている。
   この機能は削除していない。現在の比較recipeのshort-read=0は測定条件である。
2. 普通のbatched prefillはPrefill::run_implで、chunkを外側、model layerを内側に回す。
   常駐expertと必要なexpertの転送を使う。layer-major指定でも一chunk以下ならこの入口となる。
3. 長文専用のPrefill::run_layer_majorはlayerを外側、chunkを内側に回す。
   現在の比較はこの入口であり、既に独立したorchestrationがある。
   kernel/model演算はrun_implを共有するが、重みの置き方とループ順は短入力と別である。

長文選択はexact STRATA_PREFILL_LAYER_MAJOR=1かつn>chunk。
今後閾値を調整するまでは、この現状を新しい最適閾値とは呼ばない。

## 長文経路の外側

単一GPU、48層、512 experts/層、top-10、同じin-order compute/copy queueを前提にする。
現実装は全層の重みをFP16で常駐させるものではない。

```text
必要なMTP routed expertsを一時退去（このoptionが有効な場合）
prefill用の1層分quantized expert cacheと残差bankを確保
for layer = 0..47:
    前層のqueueを完了確認
    この層の512個の量子化expertをsourceから所有RAM slabへコピー
    512個のH2Dを同じqueueへ発行し、まとめて完了確認
    512個を常駐済みとして公開
    for chunk = 先頭から末尾まで:
        layer 0: token embeddingを4本の残差へ展開
        layer 1以降: 前層の残差bankからこのchunkをGPU作業領域へ読む
        この層のattention/GDNとMoEを実行
        layer 0..46: 結果を残差bankへ保存、完了確認
        layer 47: 最終callback/MTP KV処理へ渡す
    全chunkとcallbackの完了を確認して次の層へ
prefillの一時cache/残差bankを退去
MTP routed expertsを復元してdecodeへ戻す（退去していた場合）
```

一つの層の全chunkが終わるまで、512個の量子化重みは同じdevice cacheに残る。
次の層ではそのcacheを上書きする。decode側の常駐表は一時的に借りた別表から元に戻す。
この経路は通常のstream_all/ring issuerを使わず、host Stagerにもjobを発行しない。
fill_slotは非同期copyを発行するAPIであり、expert一個ごとにblocking待ちする説明は誤り。
ここでの待ちは512個のupload後にまとめて行われる。

## 一つの層・chunkの内側

1. **入力**。layer 0だけembeddingを作る。後続層は保存残差を作業用m.Rへ戻す。
   残差は4×2560=10240個のFP32/row。layer 1ではPLEを入力し、historyをtoken順に進める。
   PLEの次chunkのtable読み出しは別host threadで先読みする。
2. **attention側GR read**。4本の残差を正規化・射影・gateし、attention用入力を作る。
3. **attentionまたはGDN**。12層はQSA、36層はGDN。
   QSAはQ/K/Vとindexerを計算し、KV/indexerを追加、参照先を選択してattentionを計算する。
   mode-1 KVでは既存historyをRAMからstage poolへ用意する。GDNはconv/recurrence stateを更新する。
   層ごとにchunkは先頭からの順を守る必要がある。
4. **attention側GR write**で残差を更新し、**FFN側GR read**でMoE入力を作る。
5. **router/shared expert**。top-10のID/重みを求め、shared expertとそのgateも計算する。
6. **CPU grouping**。router IDをGPUからCPUへ渡してqueueを待ち、expert別の行数・offset・
   token対応表をCPUで作る。その表をGPUへ戻す。この待ちは重みが常駐していても発生する。
7. **routed experts**。このchunkで使うexpertをID順に処理する。各expertで
   量子化GU/DownをFP16 scratchへ展開→対応tokenの行をgather→GU GEMM→SwiGLU→Down GEMM。
   二組のscratchを使い回し、結果を元のtop-10 slotに対応する領域へ書く。
8. **combine/FFN側GR write**。元のtop-10順の重み付き合成にshared expertを加え、残差を更新する。
   次層へ渡すための保存はこの後。中間層のcallbackを最終結果として外へ公開しない。

最終層の各chunkではcaller callbackに渡す前にqueueを待つ。通常MTP有効ならdraft KVを更新し、
適用できなければmtp.prefillへfallbackする。target headの全語彙logitsを毎入力位置で出す処理は
通常の速度測定には入っていない。selected logits/NLLは別のdefault-off診断である。
standaloneは通常n_prompt-1行をbatched prefillし、最後の入力tokenはtoken/verify経路に残す。

## 「一度読めば済むもの」と繰り返すもの

| 対象 | 現在の長文経路 |
| --- | --- |
| 各層の量子化expertのRAM→VRAM | その層で一回、全chunkが再利用 |
| expertの量子化→FP16展開 | 使われるexpertごと、**各chunkで再実行** |
| expertのGU/Down GEMM | 各chunkのそのexpertに対応する行ごと |
| routerのCPU groupingと待機 | 層×chunkごと |
| RAM側残差のD2H保存/H2D再読込 | 全47の層間境界 |
| GR/attention/GDN/shared expert | モデルに必要な各層・各入力行 |

したがって現状は「重みの転送を層内で再利用」までであり、全入力に対してexpertを一度だけ
FP16へ展開して使い切るexpert-major方式は未実装。R621/R622の検討は実装済みと数えない。
全512個のGU/DownをFP16で展開すると約4.6875GiBとなり、今の追加cache/残差/既存decode領域と
共存できると未確認のまま仮定しない。decoded expertを限定して保持する、複数chunkを束ねる等は
容量・転送・GEMMの実測で選ぶ。既存のtop-10合成順を変える場合は別の数値変更として評価する。

## 現在の残差配置が生む転送

実比較recipeはchunk4096、R_GPU4096、FP32。最初のchunkだけがGPU bank、残りがRAM bankである。
実入力33214 tokenのlong-codeならbatched33213行、GPU160MiB、RAM1192632320 B。
このRAM bankを47境界で保存・再読込するため、要求byteの合計は
56053719040 B D2H + 56053719040 B H2D = **112107438080 B（約112.1GB）**。
これはsourceからの要求量の計算で、DMA実測値やcritical-path所要時間ではない。
重み、KV、PLEの転送はこの量に含まない。

同じ4096-row GPU prefixを固定した場合、65536-token入力ではbatched65535行で約236.6GB、
262144-token入力ではbatched262143行で約993.5GBになる。256Kの実行合格を意味しない。
VRAMを確保してGPU側残差を増やす、storage dtypeを変える、処理を束ねて往復回数を減らす、の
どれを優先するかはこの量と実時間を照合する。FP16の品質は別評価が必要で、無条件に採用しない。

## PR #6との差と最初の実計測

元の三標本640.34/646.46/645.85 tok/s、中央値645.85は元receipt/hashとGit履歴に存在する。
これを回復すべき実績として扱う。同時に、その条件はchunk8192、context65538、全32K残差VRAM、
全KV常駐、warm PLE、timed MTPなしであり、今のchunk4096/context262144/R_GPU4096/MTP4とは違う。
PR6と現rootは同じlayer→chunkという大枠であり、単にその順に戻すだけで解決済みとはしない。
差から原因を断定せず、旧kernel/設定/配置の寄与をprefillに絞って再現・測定する。

次は長文prefillのphase実時間と、8K chunkの容量/admissionを確認し、残差配置と展開の再利用を
個別に比較する。小tokenのdispatch閾値やdecode速度探索はしない。
注意としてR_GPU4096のままchunkを8192にするとplannerがGPU prefixを0へ丸める。
同じ配置のchunk-only比較には両armで0行にそろえるか、8192の倍数を両armで実際にadmitする必要がある。
chunk8192の追加scratchを含む実空きも確認し、admissionの失敗をGPUへ無理に通さない。
診断の小入力は安全確認に限り、性能採否は実32K以上または64Kの複数標本で判断する。

計時範囲もそろえる。layer-majorのstats.ms_totalはadmission/一時allocationより前から全層までで、
関数return時のlocal owner破棄と外側MTP復元より前に記録される。serveのprompt_msは外側request timerから
prompt処理と貸したcache slotのrefill完了までで、MTP退去/復元もその中に入る。TTFTはさらに最初の出力を
待つ指標である。内側のkernel合計、PrefillStats、公式benchのprompt速度、TTFTを同じ時間と扱わない。

## 独立調査へのMain補足

R667の末尾にあるchunk tuningもdeferという記述は今回の指示と異なる。後回しは短入力の閾値だけ。
R668のactive OFF記述は古く、OFFは正常終了済みでactiveはIQ ON exec10041である。
両reportのsource-order説明をこれらの誤記と切り分け、原bytes/hashは維持する。
進行中ONはagent六turn×三回とnominal32K三回を終え、long generationへ進んでいる。
nominal32Kの実入力32089、PP216.8/216.0/216.1は途中の観測で、最終入力照合/正常退去前の採用値ではない。

## 確認したsource

- src/prefill/layer_major.inc SHA256 1a79f1d66fdddb6ad3715fc784f2d15fecbf4288fd6cabc335005287d8d1c36c
- src/prefill/prefill.cpp SHA256 aca7a33b1019e4637ffc6738a957c0eac4d051ce8a9cf957a446524b45df51e4
- include/strata/prefill/layer_major_plan.hpp SHA256 f067cd631a6900864723b2049a95cb207ea5ac03055e2691b7d67f2706ef3721
- src/program/generate.cpp SHA256 d4048082aae24f4afc6c841309b0d0f63f2303085880d92da1cc7ebe4e83e80c
- src/core/expert_cache.cpp fill_slot、src/sycl/compat.cpp cudaMemcpyAsync、src/prefill/sycl/gemm.cpp
- PR6原測定の再検証: research-pr6-measurement-provenance-round560-v1.txt
- 独立source報告: research-current-long-prefill-execution-order-round667-v1.txt、research-pr6-current-prefill-order-delta-round668-v1.txt

今回root src/include、稼働binary、model/profile/envを変更していない。別test/build/GPU/cleanupを重ねていない。
変更は作業方針とsource調査報告のみ。600/1000 PP、70 TG、品質、物理262144/RESTOREは未達である。
